"""
download_manager.py — Zallama HuggingFace Model Downloader

Manages background model downloads from HuggingFace (or Unsloth shorthands).
Tracks downloading progress, file sizes, and registers completed downloads.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import httpx

from .model_registry import ModelRegistry

ARIA_PROGRESS_RE = re.compile(
    r"\[#(?P<hex>[0-9a-f]+)\s+"
    r"(?P<completed>[0-9.]+)(?P<c_unit>[a-zA-Z]+)/"
    r"(?P<total>[0-9.]+)(?P<t_unit>[a-zA-Z]+)\("
    r"(?P<percent>[0-9]+)%\)"
    r"(?:\s+CN:(?P<cn>\d+))?"
    r"(?:\s+DL:(?P<speed>[0-9.]+)(?P<s_unit>[a-zA-Z]+))?"
    r"(?:\s+ETA:(?P<eta>[0-9a-zA-Z]+))?"
    r"\]"
)

def parse_size_to_bytes(value: str, unit: str) -> int:
    multipliers = {
        'b': 1, 'B': 1,
        'k': 1024, 'K': 1024, 'kb': 1024, 'KB': 1024, 'kib': 1024, 'KiB': 1024,
        'm': 1024**2, 'M': 1024**2, 'mb': 1024**2, 'MB': 1024**2, 'mib': 1024**2, 'MiB': 1024**2,
        'g': 1024**3, 'G': 1024**3, 'gb': 1024**3, 'GB': 1024**3, 'gib': 1024**3, 'GiB': 1024**3,
        't': 1024**4, 'T': 1024**4, 'tb': 1024**4, 'TB': 1024**4, 'tib': 1024**4, 'TiB': 1024**4,
    }
    unit_clean = unit.strip().lower()
    mult = multipliers.get(unit_clean, 1)
    try:
        return int(float(value) * mult)
    except ValueError:
        return 0

def parse_eta(eta_str: str) -> float:
    if not eta_str:
        return 0.0
    total_seconds = 0.0
    current = ""
    for char in eta_str:
        if char.isdigit():
            current += char
        elif char == 'h':
            total_seconds += int(current or 0) * 3600
            current = ""
        elif char == 'm':
            total_seconds += int(current or 0) * 60
            current = ""
        elif char == 's':
            total_seconds += int(current or 0)
            current = ""
    return total_seconds

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
    },
    # --- RAG: embeddings & reranking ----------------------------------------
    # Embedding models run llama-server in --embedding mode via the
    # embedding-server backend and serve /v1/embeddings.
    "nomic-embed:v1.5": {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "file": "nomic-embed-text-v1.5.Q4_K_M.gguf",
        "description": "Nomic Embed Text v1.5 (embeddings, Q4_K_M)",
        "modality": "embedding",
        "backend": "embedding-server",
    },
    # Rerankers are cross-encoders; they run llama-server in --reranking mode via
    # the rerank-server backend and serve /v1/rerank.
    "bge-reranker:v2-m3": {
        "repo": "gpustack/bge-reranker-v2-m3-GGUF",
        "file": "bge-reranker-v2-m3-Q4_K_M.gguf",
        "description": "BGE Reranker v2-m3 (rerank, multilingual, Q4_K_M)",
        "modality": "rerank",
        "backend": "rerank-server",
    },
    # --- ASR / speech-to-text (parakeet.cpp) --------------------------------
    # These run on parakeet-server, not llama-server, so they carry an explicit
    # modality/backend that the downloader propagates into the registry entry.
    "parakeet:0.6b": {
        "repo": "mudler/parakeet-cpp-gguf",
        "file": "tdt-0.6b-v2-q8_0.gguf",
        "description": "NVIDIA Parakeet TDT 0.6B v2 (ASR, Q8_0)",
        "modality": "asr",
        "backend": "parakeet-server",
    },
    "parakeet:1.1b": {
        "repo": "mudler/parakeet-cpp-gguf",
        "file": "tdt-1.1b-q8_0.gguf",
        "description": "NVIDIA Parakeet TDT 1.1B (ASR, Q8_0)",
        "modality": "asr",
        "backend": "parakeet-server",
    },
    # --- TTS / text-to-speech (kokoro.cpp) ----------------------------------
    # kokoro-server loads a *directory* of files (two ONNX models + a voice
    # pack) named by content hash, so this entry declares `files` (a list) and
    # `dir` (the per-model subdirectory they're downloaded into) instead of a
    # single `file`. The registry entry's `file` becomes that directory.
    "kokoro:82m": {
        "repo": "rleo/kokoro-onnx",
        "dir": "kokoro-82m",
        "files": [
            "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb",
            "67f7dd6fed1742193e475a2fe9d060df315d9a6f434b966b15aec69fc2ed966c",
            "c3bf79648d4d8b7874b992e1cc6275688e7881f0818ec6ed8e7255c32d05ba11",
        ],
        "description": "Kokoro-82M TTS (ONNX, 54 voices)",
        "modality": "tts",
        "backend": "kokoro-server",
    },
}


class DownloadTask:
    def __init__(self, model_name: str, repo: str, filename: str, local_path: Path,
                 mmproj_filename: str | None = None,
                 modality: str | None = None, backend: str | None = None,
                 multi_files: list[str] | None = None, dest_dir: Path | None = None):
        self.model_name = model_name
        self.repo = repo
        self.filename = filename
        self.local_path = local_path
        # Multi-file models (e.g. kokoro TTS) download a *set* of files into a
        # single directory rather than one weights file. When `multi_files` is
        # set, `dest_dir` is that directory and the registry entry's `file`
        # points at it; `filename`/`local_path` are unused for the actual
        # download in that case.
        self.multi_files = multi_files
        self.dest_dir = dest_dir
        # Non-text models (e.g. ASR/parakeet) carry an explicit modality+backend
        # so registration wires them to the right engine instead of the default
        # llama-server text path.
        self.modality = modality
        self.backend = backend
        # Optional multimodal projector (vision). When set, it lives in the same
        # repo and is downloaded alongside the main gguf, then registered as the
        # `mmproj` artifact so the model is vision-capable with no extra steps.
        #
        # `mmproj_filename` is the *remote* name (used to build the download URL);
        # it is often a generic name like "mmproj-F16.gguf" shared across repos.
        # The *local* file is namespaced by the model name so two vision models
        # never overwrite each other's projector in models_dir.
        self.mmproj_filename = mmproj_filename
        self.mmproj_path: Path | None = (
            local_path.parent / self._local_mmproj_name(model_name, mmproj_filename)
            if mmproj_filename else None
        )
        self.total_bytes = 0
        self.completed_bytes = 0
        self.status = "queued"  # queued, downloading, completed, failed
        self.error: str | None = None
        self.speed = 0.0  # bytes/second
        self.eta = 0.0  # seconds remaining

    @staticmethod
    def _local_mmproj_name(model_name: str, remote_filename: str) -> str:
        """Namespace the projector by model so generic names don't collide.

        e.g. model "qwen3-vl-7b" + remote "mmproj-F16.gguf"
             -> "qwen3-vl-7b.mmproj-F16.gguf"
        The model_name is the unique registry key, so this is collision-free.
        """
        safe = model_name.replace("/", "_").replace(":", "_")
        return f"{safe}.{remote_filename}"


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

    async def _list_gguf_siblings(self, repo: str) -> list[str]:
        """Return all .gguf filenames in a HuggingFace repo."""
        url = f"https://huggingface.co/api/models/{repo}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"HuggingFace repository '{repo}' not found or inaccessible (HTTP {resp.status_code}).")
            data = resp.json()
            siblings = data.get("siblings", [])
            return [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]

    async def _auto_detect_gguf(self, repo: str) -> str:
        files = await self._list_gguf_siblings(repo)
        if not files:
            raise RuntimeError(f"No GGUF files found in repository '{repo}'.")

        # Skip projector files — those are picked separately as the mmproj artifact.
        main_files = [f for f in files if not self._is_mmproj(f)] or files

        # Prefer standard medium quantizations first
        pref_keywords = ["q4_k_m", "q4_0", "q4_k_s", "q4_1", "q5_k_m", "q5_0", "q8_0", "q3_k_m", "q3_k_s"]
        for keyword in pref_keywords:
            for f in main_files:
                if keyword in f.lower():
                    return f
        return main_files[0]

    @staticmethod
    def _is_mmproj(filename: str) -> bool:
        """A multimodal projector (vision) gguf, by HF naming convention."""
        return "mmproj" in filename.lower()

    @classmethod
    def _backend_for_modality(cls, modality: str) -> tuple[str | None, str | None]:
        """Map an explicit modality to (modality, backend).

        Delegates to the canonical map in backends.py. Text returns (None, None)
        so the registry entry stays clean and runs on llama-server; a known
        non-text modality selects its backend; an unknown/unsupported value is
        rejected with a clear error.
        """
        from .backends import TEXT, default_backend_for

        m = (modality or "").strip().lower()
        if m in ("", TEXT):
            return None, None
        backend = default_backend_for(m)  # raises ValueError if unknown/unsupported
        return m, backend

    @classmethod
    def _infer_modality_backend(cls, repo: str) -> tuple[str | None, str | None]:
        """Guess (modality, backend) from a raw HF repo id.

        Shorthands declare these explicitly, but a raw `owner/repo/file.gguf`
        pull has no hint, so we'd otherwise default everything to llama-server
        text. parakeet.cpp ships its ASR GGUFs in `*parakeet*` repos, so route
        those to the ASR backend. Returns (None, None) for plain text.
        """
        if "parakeet" in repo.lower():
            return cls._backend_for_modality("asr")
        return None, None

    async def _detect_mmproj(self, repo: str) -> str | None:
        """Find a vision projector in the repo, if any.

        Vision GGUF repos ship the projector alongside the weights as an
        `mmproj-*.gguf` file. Projectors are small, so we prefer the F16/BF16
        variant for best vision quality and fall back to whatever exists.
        Returns None for plain text repos (no projector present).
        """
        try:
            files = await self._list_gguf_siblings(repo)
        except RuntimeError:
            return None
        projectors = [f for f in files if self._is_mmproj(f)]
        if not projectors:
            return None
        for keyword in ("f16", "bf16", "f32"):
            for f in projectors:
                if keyword in f.lower():
                    return f
        return projectors[0]

    async def start_pull(
        self, model_name_or_url: str, modality_override: str | None = None
    ) -> tuple[str, str]:
        """Parse HF repo, create tasks, and spin off background downloader.

        `modality` is an optional explicit override (text | asr | tts | image),
        e.g. from `zallama pull <repo>/<file> --type asr`. When given it wins
        over the shorthand's declared modality and the repo-name heuristic, and
        selects the matching backend. When omitted, modality is inferred.
        """
        async with self._lock:
            # 1. Resolve repo and file from shorthand or raw format
            repo = ""
            filename = ""
            modality: str | None = None
            backend: str | None = None
            multi_files: list[str] | None = None
            multi_dir: str | None = None
            model_name = model_name_or_url.strip()

            # Format check: hf://repo/path/to/file.gguf or repo/path/to/file.gguf
            cleaned = model_name_or_url
            if cleaned.startswith("hf://"):
                cleaned = cleaned[5:]

            # Is it a shorthand?
            if cleaned.lower() in SHORTHANDS:
                sh = SHORTHANDS[cleaned.lower()]
                repo = sh["repo"]
                modality = sh.get("modality")
                backend = sh.get("backend")
                model_name = cleaned.lower()
                # Multi-file shorthands (e.g. kokoro TTS) declare `files` + `dir`
                # instead of a single `file`: a set of files downloaded into one
                # directory that the backend loads as its --model.
                if "files" in sh:
                    multi_files = list(sh["files"])
                    multi_dir = sh.get("dir", model_name.replace(":", "_"))
                else:
                    filename = sh["file"]
            else:
                # Expecting format: username/repo/filename.gguf or username/repo
                parts = cleaned.split("/")
                if len(parts) >= 3:
                    repo = f"{parts[0]}/{parts[1]}"
                    filename = parts[-1]
                elif len(parts) == 2:
                    repo = f"{parts[0]}/{parts[1]}"
                    filename = await self._auto_detect_gguf(repo)
                else:
                    raise ValueError(
                        f"Invalid model name. Specify a shorthand (e.g. 'llama3.2:3b') "
                        f"or full HuggingFace path (e.g. 'unsloth/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf')"
                    )
                # Register under a clean model name based on filename
                model_name = filename.replace(".gguf", "").lower()
                # Raw HF paths carry no modality hint the way shorthands do, so
                # infer it from the repo name (parakeet.cpp repos are ASR).
                modality, backend = self._infer_modality_backend(repo)

            # An explicit --type override wins over both the shorthand and the
            # repo heuristic, and selects the backend that serves that modality.
            if modality_override:
                modality, backend = self._backend_for_modality(modality_override)

            if model_name in self._tasks and self._tasks[model_name].status in ("queued", "downloading"):
                return model_name, f"Model '{model_name}' download is already in progress."

            # Vision (text) models ship a projector in the same repo; grab it too
            # so the pulled model is vision-capable with no manual registry edit.
            # Non-text modalities (ASR/TTS/…) never carry an mmproj, so skip the
            # probe for them.
            mmproj_filename = None
            if modality in (None, "text"):
                mmproj_filename = await self._detect_mmproj(repo)

            if multi_files is not None:
                # Multi-file model: a directory of files, no single weights path.
                dest_dir = self.models_dir / multi_dir
                local_path = dest_dir  # registry `file` is the directory itself
                task = DownloadTask(
                    model_name, repo, "", local_path,
                    modality=modality, backend=backend,
                    multi_files=multi_files, dest_dir=dest_dir,
                )
            else:
                local_path = self.models_dir / filename
                task = DownloadTask(
                    model_name, repo, filename, local_path, mmproj_filename,
                    modality=modality, backend=backend
                )
            self._tasks[model_name] = task

            # Trigger background task
            asyncio.create_task(self._download_loop(task))
            msg = f"Started downloading '{model_name}' in the background."
            if multi_files is not None:
                msg += f" ({len(multi_files)} files → {multi_dir}/)"
            elif mmproj_filename:
                msg += f" (vision projector '{mmproj_filename}' included)"
            return model_name, msg

    async def _download_loop(self, task: DownloadTask):
        """Asynchronously download the model file using aria2c or concurrent Python requests."""
        # Multi-file models (kokoro TTS) download a set of files into a directory
        # and register that directory, on a separate, simpler path.
        if task.multi_files is not None:
            await self._download_multi(task)
            return

        url = f"https://huggingface.co/{task.repo}/resolve/main/{task.filename}"
        logger.info(f"Downloading model '{task.model_name}' from {url}")

        task.status = "downloading"
        temp_path = task.local_path.with_suffix(".download")

        try:
            # Step 1: Try aria2c first
            success = await self._download_with_aria2c(task, url, temp_path)
            if not success:
                logger.warning("aria2c failed or was not found. Falling back to python downloader.")
                # Step 2: Fallback to range or single download
                total_bytes, range_ok = await self._check_range_support(url)
                if range_ok and total_bytes > 10 * 1024 * 1024:
                    await self._download_parallel(task, url, temp_path, total_bytes, num_connections=4)
                else:
                    await self._download_single(task, url, temp_path)

            # Move temp file to final location
            if temp_path.exists():
                temp_path.rename(task.local_path)

            # Vision projector: small, single-stream fetch alongside the weights.
            artifacts = None
            if task.mmproj_filename and task.mmproj_path is not None:
                await self._fetch_mmproj(task)
                artifacts = {"mmproj": str(task.mmproj_path)}

            task.status = "completed"
            task.speed = 0
            task.eta = 0

            # Register model in registry.yaml
            description = f"Downloaded from {task.repo}"
            if task.model_name in SHORTHANDS:
                description = SHORTHANDS[task.model_name]["description"]

            # Params are backend-specific. The llama-server-based backends
            # (text, embedding-server, rerank-server) take ctx/gpu flags; the
            # ASR/TTS backends do not accept those flags. Pick defaults from the
            # backend rather than the modality so all llama-server variants match.
            _llama_backends = {"llama-server", "embedding-server", "rerank-server"}
            if (task.backend or "llama-server") in _llama_backends:
                params = {"ctx_size": 4096, "n_gpu_layers": 99}
            else:
                params = {"threads": 4}

            # A shorthand may declare its own params (e.g. an embedding model that
            # must launch with `embedding: true`). These override the defaults.
            if task.model_name in SHORTHANDS:
                params.update(SHORTHANDS[task.model_name].get("params") or {})

            self.registry.add_model(
                name=task.model_name,
                file_path=str(task.local_path),
                description=description,
                modality=task.modality,
                backend=task.backend,
                params=params,
                artifacts=artifacts,
            )
            logger.info(f"Completed and registered model: {task.model_name}")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Download failed for '{task.model_name}': {e}")
            # Cleanup temp files (main weights + any partial projector)
            cleanup = [temp_path, temp_path.with_suffix(".download.aria2"),
                       temp_path.parent / (temp_path.name + ".aria2")]
            if task.mmproj_path is not None:
                cleanup.append(task.mmproj_path.with_suffix(task.mmproj_path.suffix + ".download"))
            for path in cleanup:
                if path.exists():
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    async def _download_multi(self, task: DownloadTask):
        """Download a set of files into one directory and register the directory.

        Used by multi-file models (kokoro TTS): the backend's --model is a
        directory of files (named by content hash), so we fetch each file with a
        plain streaming download, accumulate progress across them, then register
        the directory as the model's `file`.
        """
        task.status = "downloading"
        dest_dir = task.dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Sum the sizes up-front so the progress bar reflects the whole set.
            total = 0
            sizes: dict[str, int] = {}
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                for name in task.multi_files:
                    url = f"https://huggingface.co/{task.repo}/resolve/main/{name}"
                    head = await client.head(url)
                    sz = int(head.headers.get("content-length", 0))
                    sizes[name] = sz
                    total += sz
            task.total_bytes = total
            task.completed_bytes = 0

            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                for name in task.multi_files:
                    dest = dest_dir / name
                    # Skip files already present at the expected size (resumable
                    # across re-pulls without re-downloading the big ONNX models).
                    if dest.exists() and sizes.get(name) and dest.stat().st_size == sizes[name]:
                        task.completed_bytes += sizes[name]
                        continue
                    url = f"https://huggingface.co/{task.repo}/resolve/main/{name}"
                    logger.info(f"Downloading '{name}' from {url}")
                    temp = dest.with_suffix(dest.suffix + ".download")
                    async with client.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            raise RuntimeError(
                                f"Failed to download '{name}': HTTP {resp.status_code}"
                            )
                        with open(temp, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                                f.write(chunk)
                                task.completed_bytes += len(chunk)
                    temp.rename(dest)

            task.status = "completed"
            task.speed = 0
            task.eta = 0

            description = f"Downloaded from {task.repo}"
            if task.model_name in SHORTHANDS:
                description = SHORTHANDS[task.model_name]["description"]

            # kokoro takes no launch flags; `voice`/`speed` are per-request
            # synthesis knobs the /v1/audio/speech route injects from these
            # params when the client omits them. speed=1.0 is normal rate.
            # Pin TTS by default: kokoro's ONNX load is a slow CPU operation, so
            # keeping it warm turns a tens-of-seconds first request into a
            # sub-second one. The user can unpin via the registry if they'd
            # rather reclaim the memory.
            self.registry.add_model(
                name=task.model_name,
                file_path=str(dest_dir),
                description=description,
                modality=task.modality,
                backend=task.backend,
                params={"voice": "af_heart", "speed": 1.0},
                pinned=(task.modality == "tts"),
            )
            logger.info(f"Completed and registered model: {task.model_name}")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Download failed for '{task.model_name}': {e}")
            # Leave any complete files in place (re-pull resumes); clean partials.
            for name in task.multi_files or []:
                partial = dest_dir / (name + ".download")
                if partial.exists():
                    try:
                        os.remove(partial)
                    except Exception:
                        pass

    async def _fetch_mmproj(self, task: DownloadTask):
        """Download the vision projector into models_dir (single stream).

        Projectors are small (tens to a few hundred MB), so a plain streaming
        download is enough and keeps the main task's progress untouched.
        """
        url = f"https://huggingface.co/{task.repo}/resolve/main/{task.mmproj_filename}"
        dest = task.mmproj_path
        logger.info(f"Downloading vision projector '{task.mmproj_filename}' from {url}")
        temp = dest.with_suffix(dest.suffix + ".download")
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download projector '{task.mmproj_filename}': "
                        f"HTTP {resp.status_code}"
                    )
                with open(temp, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
        temp.rename(dest)
        logger.info(f"Vision projector ready: {dest}")

    async def _check_range_support(self, url: str) -> tuple[int, bool]:
        """Check if target server supports HTTP Range requests and get content length."""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                async with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as resp:
                    if resp.status_code == 206:
                        content_range = resp.headers.get("content-range")
                        if content_range and "/" in content_range:
                            total_bytes = int(content_range.split("/")[-1])
                            return total_bytes, True
                    total_bytes = int(resp.headers.get("content-length", 0))
                    return total_bytes, False
        except Exception as e:
            logger.warning(f"Error checking Range support for {url}: {e}")
            return 0, False

    async def _download_with_aria2c(self, task: DownloadTask, url: str, temp_path: Path) -> bool:
        """Download using aria2c if available."""
        aria2c_path = shutil.which("aria2c")
        if not aria2c_path:
            return False

        logger.info("Starting download with aria2c...")
        args = [
            "aria2c",
            "--console-log-level=warn",
            "--summary-interval=1",
            "-x", "8",
            "-s", "8",
            "-k", "1M",
            "--allow-overwrite=true",
            "-d", str(temp_path.parent),
            "-o", temp_path.name,
            url
        ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                match = ARIA_PROGRESS_RE.search(line)
                if match:
                    gd = match.groupdict()
                    completed = gd.get("completed")
                    c_unit = gd.get("c_unit")
                    total = gd.get("total")
                    t_unit = gd.get("t_unit")
                    speed = gd.get("speed")
                    s_unit = gd.get("s_unit")
                    eta = gd.get("eta")

                    if completed and c_unit:
                        task.completed_bytes = parse_size_to_bytes(completed, c_unit)
                    if total and t_unit:
                        task.total_bytes = parse_size_to_bytes(total, t_unit)
                    if speed and s_unit:
                        task.speed = parse_size_to_bytes(speed, s_unit)
                    if eta:
                        task.eta = parse_eta(eta)

            await proc.wait()
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"aria2c download exception: {e}")
            if proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return False

    async def _download_parallel(self, task: DownloadTask, url: str, temp_path: Path, total_bytes: int, num_connections: int = 4):
        """Download concurrently using HTTP range requests."""
        logger.info(f"Starting concurrent range download with {num_connections} connections...")
        part_size = total_bytes // num_connections
        parts = []
        for i in range(num_connections):
            start = i * part_size
            end = (start + part_size - 1) if i < num_connections - 1 else total_bytes - 1
            parts.append((start, end))

        task.total_bytes = total_bytes
        task.completed_bytes = 0

        # Pre-allocate
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.truncate(total_bytes)

        last_time = asyncio.get_event_loop().time()
        last_bytes = 0

        async def download_part(part_idx: int, start: int, end: int):
            current_offset = start
            headers = {"Range": f"bytes={start}-{end}"}
            timeout = httpx.Timeout(30.0, connect=10.0)
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"Part {part_idx} download failed: HTTP {resp.status_code}")
                    
                    with open(temp_path, "r+b") as f:
                        f.seek(current_offset)
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                            f.write(chunk)
                            chunk_len = len(chunk)
                            current_offset += chunk_len
                            task.completed_bytes += chunk_len

        monitor_running = True
        async def monitor_progress():
            nonlocal last_time, last_bytes
            while monitor_running:
                await asyncio.sleep(1.0)
                now = asyncio.get_event_loop().time()
                duration = now - last_time
                if duration >= 0.5:
                    bytes_diff = task.completed_bytes - last_bytes
                    task.speed = bytes_diff / duration
                    remaining_bytes = task.total_bytes - task.completed_bytes
                    task.eta = (remaining_bytes / task.speed) if task.speed > 0 else 0
                    last_time = now
                    last_bytes = task.completed_bytes

        monitor_task = asyncio.create_task(monitor_progress())
        pending_tasks = [asyncio.create_task(download_part(i, start, end)) for i, (start, end) in enumerate(parts)]
        try:
            await asyncio.gather(*pending_tasks)
        except Exception as e:
            for t in pending_tasks:
                if not t.done():
                    t.cancel()
            raise e
        finally:
            monitor_running = False
            monitor_task.cancel()

    async def _download_single(self, task: DownloadTask, url: str, temp_path: Path):
        """Single-connection streaming fallback downloader."""
        logger.info("Starting single-connection streaming download...")
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP Error {resp.status_code}: {resp.reason_phrase}")

                task.total_bytes = int(resp.headers.get("content-length", 0))
                task.completed_bytes = 0

                last_time = asyncio.get_event_loop().time()
                last_bytes = 0

                with open(temp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        task.completed_bytes += len(chunk)

                        now = asyncio.get_event_loop().time()
                        duration = now - last_time
                        if duration >= 1.0:
                            bytes_diff = task.completed_bytes - last_bytes
                            task.speed = bytes_diff / duration
                            remaining_bytes = task.total_bytes - task.completed_bytes
                            task.eta = (remaining_bytes / task.speed) if task.speed > 0 else 0
                            last_time = now
                            last_bytes = task.completed_bytes
