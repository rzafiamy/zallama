"""
process_manager.py — Zallama Process Manager

Manages inference subprocess lifecycle (backend-agnostic):
  - Spawning a server instance per model via its Backend
  - Health-checking until ready
  - Port assignment (with OS bind-check)
  - LRU eviction on idle timeout and on max-loaded cap
  - Graceful shutdown

What it does NOT know: how to build the command line or which binary to run for
a given model. That lives in backends.py, keyed off the model's `backend` field,
so adding TTS/ASR/image backends does not touch this file.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from collections import OrderedDict
from pathlib import Path

import httpx

from .backends import Backend, get_backend
from .config import resolve_binary

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
        backend: Backend,
        mem_gb: float = 0.0,
    ):
        self.name = name
        self.port = port
        self.process = process
        self.log_file = log_file
        self.entry = entry
        self.backend = backend
        self.mem_gb = mem_gb  # declared/estimated memory cost
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
    def __init__(self, cfg: dict, logs_dir: str):
        self.cfg = cfg
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._instances: OrderedDict[str, ModelInstance] = OrderedDict()
        self._global_lock = asyncio.Lock()
        # Per-model locks so booting one model never blocks requests to another.
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._binary_cache: dict[str, str] = {}

        ls = cfg["llama_server"]
        self._port_counter = ls["port_start"]
        self._idle_timeout: int = ls.get("idle_timeout", 300)
        self._startup_timeout: int = ls.get("startup_timeout", 60)
        self._max_loaded: int = ls.get("max_loaded_models", 0)  # 0 = unlimited
        self._mem_budget_gb: float = float(ls.get("mem_budget_gb", 0))  # 0 = unlimited
        self._mem_init_gb: float = float(ls.get("mem_init_gb", 2))      # fallback cost

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def _lock_for(self, model_name: str) -> asyncio.Lock:
        lock = self._model_locks.get(model_name)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_name] = lock
        return lock

    async def get_or_start(self, model_name: str, entry: dict, model_path: Path) -> ModelInstance:
        """Return running instance for model, starting it if necessary.

        Held lock is per-model: concurrent loads of *different* models run in
        parallel, and a slow startup never blocks unrelated requests.
        """
        # Fast path: already running.
        async with self._global_lock:
            inst = self._instances.get(model_name)
            if inst is not None and inst.is_alive():
                inst.touch()
                self._instances.move_to_end(model_name)
                return inst

        # Slow path: serialize starts of *this* model only.
        async with self._lock_for(model_name):
            async with self._global_lock:
                inst = self._instances.get(model_name)
                if inst is not None:
                    if inst.is_alive():
                        inst.touch()
                        self._instances.move_to_end(model_name)
                        return inst
                    logger.warning(f"Instance {model_name} died unexpectedly, restarting...")
                    del self._instances[model_name]

            # Make room *before* spawning: evict LRU until the incoming model
            # fits within the count and memory budgets.
            incoming_cost = self._estimate_cost(entry, model_path)
            async with self._global_lock:
                await self._make_room_locked(incoming_cost)

            inst = await self._spawn(model_name, entry, model_path, incoming_cost)

            async with self._global_lock:
                self._instances[model_name] = inst
            return inst

    async def stop(self, model_name: str) -> bool:
        """Stop a running model instance."""
        async with self._global_lock:
            if model_name not in self._instances:
                return False
            inst = self._instances.pop(model_name)
        await self._kill_instance(inst)
        return True

    def is_running(self, model_name: str) -> bool:
        """True if the model has a live instance."""
        inst = self._instances.get(model_name)
        return inst is not None and inst.is_alive()

    def list_running(self) -> list[dict]:
        """Return info about all running instances."""
        result = []
        for name, inst in self._instances.items():
            result.append({
                "name": name,
                "port": inst.port,
                "base_url": inst.base_url,
                "modality": inst.entry.get("modality", "text"),
                "backend": inst.backend.name,
                "mem_gb": round(inst.mem_gb, 2),
                "started_at": inst.started_at,
                "last_used": inst.last_used,
                "alive": inst.is_alive(),
            })
        return result

    def memory_status(self) -> dict:
        """Loaded memory vs. configured budget (GB)."""
        used = round(self._loaded_mem_gb(), 2)
        return {
            "loaded_gb": used,
            "budget_gb": self._mem_budget_gb,
            "headroom_gb": round(self._mem_budget_gb - used, 2) if self._mem_budget_gb > 0 else None,
            "max_loaded_models": self._max_loaded,
            "loaded_count": len(self._instances),
        }

    async def shutdown_all(self):
        """Gracefully stop all running instances."""
        async with self._global_lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for inst in instances:
            await self._kill_instance(inst)

    async def sweep_idle(self):
        """Background task: evict models idle longer than idle_timeout."""
        if self._idle_timeout <= 0:
            return
        async with self._global_lock:
            now = time.time()
            to_evict = [
                name for name, inst in self._instances.items()
                if (now - inst.last_used) > self._idle_timeout
            ]
            evicted = [self._instances.pop(name) for name in to_evict]
        for inst in evicted:
            logger.info(f"Evicting idle model: {inst.name}")
            await self._kill_instance(inst)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _estimate_cost(self, entry: dict, model_path: Path) -> float:
        """Estimated memory cost (GB) of running a model.

        Prefers the declared `mem_gb`; otherwise approximates from the GGUF file
        size (a loaded GGUF roughly occupies its on-disk size plus KV-cache
        overhead, so file_size * 1.2 is a reasonable default); falls back to
        the configured mem_init_gb when the size is unknown.
        """
        declared = entry.get("mem_gb")
        if declared:
            try:
                return float(declared)
            except (TypeError, ValueError):
                pass
        try:
            size_gb = model_path.stat().st_size / 1e9
            if size_gb > 0:
                return round(size_gb * 1.2, 2)
        except OSError:
            pass
        return self._mem_init_gb

    def _loaded_mem_gb(self) -> float:
        return sum(inst.mem_gb for inst in self._instances.values())

    async def _make_room_locked(self, incoming_cost: float):
        """Evict LRU instances until an incoming model fits both budgets.

        Caller holds the global lock. Count budget: keep loaded count below
        max_loaded. Memory budget: keep loaded + incoming within mem_budget_gb.
        Eviction always targets the least-recently-used instance first.
        """
        def over_count() -> bool:
            return self._max_loaded > 0 and len(self._instances) >= self._max_loaded

        def over_mem() -> bool:
            return (
                self._mem_budget_gb > 0
                and (self._loaded_mem_gb() + incoming_cost) > self._mem_budget_gb
            )

        while self._instances and (over_count() or over_mem()):
            victim_name = next(iter(self._instances))  # LRU is first
            victim = self._instances.pop(victim_name)
            reason = "count" if over_count() else "memory"
            logger.info(
                f"Capacity ({reason}) reached — evicting LRU model "
                f"'{victim_name}' ({victim.mem_gb:.1f}GB) to make room "
                f"for incoming {incoming_cost:.1f}GB"
            )
            await self._kill_instance(victim)

    def _binary_for(self, backend: Backend) -> str:
        cached = self._binary_cache.get(backend.name)
        if cached:
            return cached
        binary = resolve_binary(self.cfg, backend.binary_name)
        self._binary_cache[backend.name] = binary
        return binary

    def _next_port(self) -> int:
        """Pick the next free port, skipping ours and anything the OS holds."""
        used_ports = {inst.port for inst in self._instances.values()}
        port = self._port_counter
        while port in used_ports or not self._port_free(port):
            port += 1
        self._port_counter = port + 1
        return port

    @staticmethod
    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _merged_params(self, entry: dict) -> dict:
        default_params = self.cfg["llama_server"]["default_params"].copy()
        model_params = entry.get("params", {})
        return {**default_params, **model_params}

    async def _spawn(
        self, model_name: str, entry: dict, model_path: Path, mem_gb: float = 0.0
    ) -> ModelInstance:
        """Spawn a new inference server process for the model's backend."""
        from .dependencies import get_registry  # local import to avoid cycle

        backend = get_backend(entry.get("backend"))
        binary = self._binary_for(backend)
        artifacts = get_registry().resolve_artifacts(entry)
        merged = self._merged_params(entry)

        port = self._next_port()
        log_path = self.logs_dir / f"{model_name.replace(':', '_')}.log"
        args = backend.build_args(binary, port, model_path, entry, merged, artifacts)

        logger.info(f"Spawning {backend.name} for '{model_name}' on port {port}")
        logger.debug(f"Command: {' '.join(args)}")

        log_file = open(log_path, "ab")
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid,
            )
        finally:
            log_file.close()

        inst = ModelInstance(
            name=model_name,
            port=port,
            process=process,
            log_file=log_path,
            entry=entry,
            backend=backend,
            mem_gb=mem_gb,
        )

        await self._wait_healthy(inst)
        logger.info(f"Model '{model_name}' is ready on port {port}")
        return inst

    async def _wait_healthy(self, inst: ModelInstance):
        """Poll the backend's health path until the server is ready."""
        health_url = f"{inst.base_url}{inst.backend.health_path()}"
        deadline = time.time() + self._startup_timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() < deadline:
                if not inst.is_alive():
                    raise RuntimeError(
                        f"{inst.backend.name} for '{inst.name}' died during startup. "
                        f"Check logs: {inst.log_file}"
                    )
                try:
                    r = await client.get(health_url)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(0.5)
        # Time out: kill the half-started process so it doesn't linger.
        await self._kill_instance(inst)
        raise TimeoutError(
            f"{inst.backend.name} for '{inst.name}' did not become healthy within "
            f"{self._startup_timeout}s. Check logs: {inst.log_file}"
        )

    async def _kill_instance(self, inst: ModelInstance):
        """Gracefully terminate a process group."""
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
