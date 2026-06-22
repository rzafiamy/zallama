"""
download_manager.py — Zallama HuggingFace Model Downloader

Manages background model downloads from HuggingFace (or Unsloth shorthands).
Tracks downloading progress, file sizes, and registers completed downloads.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from .model_registry import ModelRegistry

logger = logging.getLogger("zallama.download_manager")

SHORTHANDS = {
    "llama3.3:70b": {
        "repo": "unsloth/Llama-3.3-70B-Instruct-GGUF",
        "file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.3 70B Instruct (Q4_K_M)"
    },
    "llama3.2:3b": {
        "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.2 3B Instruct (Q4_K_M)"
    },
    "llama3.2:1b": {
        "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "description": "Llama 3.2 1B Instruct (Q4_K_M)"
    },
    "phi4:14b": {
        "repo": "unsloth/phi-4-GGUF",
        "file": "phi-4-Q4_K_M.gguf",
        "description": "Phi-4 14B Instruct (Q4_K_M)"
    },
    "qwen2.5:7b": {
        "repo": "unsloth/Qwen2.5-7B-Instruct-GGUF",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 7B Instruct (Q4_K_M)"
    },
    "qwen2.5-coder:32b": {
        "repo": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 Coder 32B Instruct (Q4_K_M)"
    },
    "qwen2.5-coder:7b": {
        "repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen 2.5 Coder 7B Instruct (Q4_K_M)"
    },
    "deepseek-r1:32b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "description": "DeepSeek R1 Distilled Qwen 32B (Q4_K_M)"
    },
    "deepseek-r1:8b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "file": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "description": "DeepSeek R1 Distilled Llama 8B (Q4_K_M)"
    },
    "deepseek-r1:1.5b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "file": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "description": "DeepSeek R1 Distilled Qwen 1.5B (Q4_K_M)"
    }
}


class DownloadTask:
    def __init__(self, model_name: str, repo: str, filename: str, local_path: Path):
        self.model_name = model_name
        self.repo = repo
        self.filename = filename
        self.local_path = local_path
        self.total_bytes = 0
        self.completed_bytes = 0
        self.status = "queued"  # queued, downloading, completed, failed
        self.error: str | None = None
        self.speed = 0.0  # bytes/second
        self.eta = 0.0  # seconds remaining


class DownloadManager:
    def __init__(self, registry: ModelRegistry, models_dir: str):
        self.registry = registry
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, DownloadTask] = {}
        self._lock = asyncio.Lock()

    def get_shorthands(self) -> dict:
        return SHORTHANDS

    def list_tasks(self) -> list[dict]:
        """Return information about current download tasks."""
        result = []
        for name, task in self._tasks.items():
            result.append({
                "model": task.model_name,
                "repo": task.repo,
                "filename": task.filename,
                "status": task.status,
                "total_bytes": task.total_bytes,
                "completed_bytes": task.completed_bytes,
                "percent": (task.completed_bytes / task.total_bytes * 100) if task.total_bytes > 0 else 0,
                "error": task.error,
                "speed": task.speed,
                "eta": task.eta,
            })
        return result

    async def start_pull(self, model_name_or_url: str) -> str:
        """Parse HF repo, create tasks, and spin off background downloader."""
        async with self._lock:
            # 1. Resolve repo and file from shorthand or raw format
            repo = ""
            filename = ""
            model_name = model_name_or_url.strip()

            # Format check: hf://repo/path/to/file.gguf or repo/path/to/file.gguf
            cleaned = model_name_or_url
            if cleaned.startswith("hf://"):
                cleaned = cleaned[5:]

            # Is it a shorthand?
            if cleaned.lower() in SHORTHANDS:
                sh = SHORTHANDS[cleaned.lower()]
                repo = sh["repo"]
                filename = sh["file"]
                model_name = cleaned.lower()
            else:
                # Expecting format: username/repo/filename.gguf
                # Let's split by "/"
                parts = cleaned.split("/")
                if len(parts) >= 3:
                    repo = f"{parts[0]}/{parts[1]}"
                    filename = parts[-1]
                else:
                    raise ValueError(
                        f"Invalid model name. Specify a shorthand (e.g. 'llama3.2:3b') "
                        f"or full HuggingFace path (e.g. 'unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf')"
                    )

            if model_name in self._tasks and self._tasks[model_name].status in ("queued", "downloading"):
                return f"Model '{model_name}' download is already in progress."

            local_path = self.models_dir / filename
            task = DownloadTask(model_name, repo, filename, local_path)
            self._tasks[model_name] = task

            # Trigger background task
            asyncio.create_task(self._download_loop(task))
            return f"Started downloading '{model_name}' in the background."

    async def _download_loop(self, task: DownloadTask):
        """Asynchronously stream the file download in chunks."""
        url = f"https://huggingface.co/{task.repo}/resolve/main/{task.filename}"
        logger.info(f"Downloading model '{task.model_name}' from {url}")

        task.status = "downloading"
        temp_path = task.local_path.with_suffix(".download")

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP Error {resp.status_code}: {resp.reason_phrase}")

                    task.total_bytes = int(resp.headers.get("content-length", 0))
                    task.completed_bytes = 0

                    start_time = asyncio.get_event_loop().time()
                    last_time = start_time
                    last_bytes = 0

                    with open(temp_path, "wb") as f:
                        async for chunk in resp.iter_bytes(chunk_size=1024 * 128):
                            f.write(chunk)
                            task.completed_bytes += len(chunk)

                            # Calculate speed & ETA every second
                            now = asyncio.get_event_loop().time()
                            duration = now - last_time
                            if duration >= 1.0:
                                bytes_diff = task.completed_bytes - last_bytes
                                task.speed = bytes_diff / duration
                                remaining_bytes = task.total_bytes - task.completed_bytes
                                task.eta = (remaining_bytes / task.speed) if task.speed > 0 else 0
                                last_time = now
                                last_bytes = task.completed_bytes

                            # Let event loop breathe
                            await asyncio.sleep(0.001)

            # Move temp file to final location
            if temp_path.exists():
                temp_path.rename(task.local_path)

            task.status = "completed"
            task.speed = 0
            task.eta = 0

            # Register model in registry.yaml
            description = f"Downloaded from {task.repo}"
            # Match shorthand description if present
            if task.model_name in SHORTHANDS:
                description = SHORTHANDS[task.model_name]["description"]

            self.registry.add_model(
                name=task.model_name,
                file_path=str(task.local_path),
                description=description,
                params={"ctx_size": 4096, "n_gpu_layers": 99}
            )
            logger.info(f"Completed and registered model: {task.model_name}")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Download failed for '{task.model_name}': {e}")
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
