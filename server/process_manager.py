"""
process_manager.py — Zallama Process Manager

Manages llama-server subprocess lifecycle:
  - Spawning a llama-server instance per model
  - Health-checking until ready
  - Port assignment
  - LRU eviction when idle_timeout is reached
  - Graceful shutdown
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("zallama.process_manager")


# ---------------------------------------------------------------------------
# Data model for a running model instance
# ---------------------------------------------------------------------------
class ModelInstance:
    def __init__(
        self,
        name: str,
        port: int,
        process: asyncio.subprocess.Process,
        log_file: Path,
        entry: dict,
    ):
        self.name = name
        self.port = port
        self.process = process
        self.log_file = log_file
        self.entry = entry
        self.started_at = time.time()
        self.last_used = time.time()
        self.base_url = f"http://127.0.0.1:{port}"

    def touch(self):
        self.last_used = time.time()

    def is_alive(self) -> bool:
        return self.process.returncode is None


# ---------------------------------------------------------------------------
# Process Manager
# ---------------------------------------------------------------------------
class ProcessManager:
    def __init__(self, cfg: dict, binary: str, logs_dir: str):
        self.cfg = cfg
        self.binary = binary
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._instances: OrderedDict[str, ModelInstance] = OrderedDict()
        self._lock = asyncio.Lock()
        self._port_counter = cfg["llama_server"]["port_start"]
        self._idle_timeout: int = cfg["llama_server"].get("idle_timeout", 300)
        self._startup_timeout: int = cfg["llama_server"].get("startup_timeout", 60)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_or_start(self, model_name: str, entry: dict, model_path: Path) -> ModelInstance:
        """Return running instance for model, starting it if necessary."""
        async with self._lock:
            if model_name in self._instances:
                inst = self._instances[model_name]
                if inst.is_alive():
                    inst.touch()
                    # Move to end (LRU)
                    self._instances.move_to_end(model_name)
                    return inst
                else:
                    logger.warning(f"Instance {model_name} died unexpectedly, restarting...")
                    del self._instances[model_name]

            inst = await self._spawn(model_name, entry, model_path)
            self._instances[model_name] = inst
            return inst

    async def stop(self, model_name: str) -> bool:
        """Stop a running model instance."""
        async with self._lock:
            if model_name not in self._instances:
                return False
            inst = self._instances.pop(model_name)
            await self._kill_instance(inst)
            return True

    def list_running(self) -> list[dict]:
        """Return info about all running instances."""
        result = []
        for name, inst in self._instances.items():
            result.append({
                "name": name,
                "port": inst.port,
                "base_url": inst.base_url,
                "started_at": inst.started_at,
                "last_used": inst.last_used,
                "alive": inst.is_alive(),
            })
        return result

    async def shutdown_all(self):
        """Gracefully stop all running instances."""
        async with self._lock:
            for inst in list(self._instances.values()):
                await self._kill_instance(inst)
            self._instances.clear()

    async def sweep_idle(self):
        """Background task: evict models idle longer than idle_timeout."""
        if self._idle_timeout <= 0:
            return
        async with self._lock:
            now = time.time()
            to_evict = [
                name for name, inst in self._instances.items()
                if (now - inst.last_used) > self._idle_timeout
            ]
            for name in to_evict:
                inst = self._instances.pop(name)
                logger.info(f"Evicting idle model: {name}")
                await self._kill_instance(inst)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _next_port(self) -> int:
        used_ports = {inst.port for inst in self._instances.values()}
        port = self._port_counter
        while port in used_ports:
            port += 1
        self._port_counter = port + 1
        return port

    def _build_args(self, port: int, model_path: Path, entry: dict) -> list[str]:
        """Build llama-server CLI arguments from config + model params."""
        default_params = self.cfg["llama_server"]["default_params"].copy()
        model_params = entry.get("params", {})
        merged = {**default_params, **model_params}

        args = [
            self.binary,
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        param_map = {
            "ctx_size": "--ctx-size",
            "n_gpu_layers": "--n-gpu-layers",
            "threads": "--threads",
            "parallel": "--parallel",
        }
        flag_map = {
            "cont_batching": "--cont-batching",
            "mlock": "--mlock",
            "no_mmap": "--no-mmap",
            "embedding": "--embedding",
        }

        for key, flag in param_map.items():
            if key in merged:
                args += [flag, str(merged[key])]

        # Handle flash_attn (takes an option value in newer llama.cpp)
        if "flash_attn" in merged:
            val = merged["flash_attn"]
            if val is True:
                args += ["--flash-attn", "on"]
            elif val is False:
                args += ["--flash-attn", "off"]
            elif isinstance(val, str) and val in ("on", "off", "auto"):
                args += ["--flash-attn", val]

        for key, flag in flag_map.items():
            if merged.get(key):
                args.append(flag)

        if "chat_template" in merged:
            args += ["--chat-template", str(merged["chat_template"])]

        return args

    async def _spawn(self, model_name: str, entry: dict, model_path: Path) -> ModelInstance:
        """Spawn a new llama-server process."""
        port = self._next_port()
        log_path = self.logs_dir / f"{model_name.replace(':', '_')}.log"
        args = self._build_args(port, model_path, entry)

        logger.info(f"Spawning llama-server for '{model_name}' on port {port}")
        logger.debug(f"Command: {' '.join(args)}")

        log_file = open(log_path, "ab")
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=log_file,
            stderr=log_file,
            preexec_fn=os.setsid,
        )

        inst = ModelInstance(
            name=model_name,
            port=port,
            process=process,
            log_file=log_path,
            entry=entry,
        )

        # Wait for llama-server to become healthy
        await self._wait_healthy(inst)
        logger.info(f"Model '{model_name}' is ready on port {port}")
        return inst

    async def _wait_healthy(self, inst: ModelInstance):
        """Poll /health until llama-server is ready."""
        deadline = time.time() + self._startup_timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not inst.is_alive():
                    raise RuntimeError(
                        f"llama-server for '{inst.name}' died during startup. "
                        f"Check logs: {inst.log_file}"
                    )
                try:
                    r = await client.get(f"{inst.base_url}/health")
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        raise TimeoutError(
            f"llama-server for '{inst.name}' did not become healthy within "
            f"{self._startup_timeout}s. Check logs: {inst.log_file}"
        )

    async def _kill_instance(self, inst: ModelInstance):
        """Gracefully terminate a process."""
        try:
            if inst.process.returncode is None:
                os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(inst.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        logger.info(f"Stopped model '{inst.name}'")
