"""
backends.py — Zallama Backend Abstraction

A *backend* knows how to turn a registered model entry into a runnable
subprocess and how to talk to it once running. The generic process lifecycle
(spawn, health-check, port assignment, LRU eviction, kill) lives in
process_manager.py and is backend-agnostic; everything that is specific to a
particular inference engine lives here.

This is the seam that lets new modalities (TTS, ASR, image generation) be added
as *new Backend subclasses* rather than as cross-cutting changes:

  - LlamaServerBackend  → text / chat / embeddings / vision   (llama-server)
  - (future) WhisperBackend   → ASR        (whisper-server)
  - (future) TtsBackend       → TTS        (llama-tts / a tts server)
  - (future) DiffusionBackend → image gen  (sd-server / llama-diffusion)

Each backend declares:
  - binary_name:   the executable to look for (./bin/<name>, ~/.zallama/bin, PATH)
  - modalities:    the set of modalities it can serve
  - build_args():  full argv for the subprocess
  - health_path(): the HTTP path to poll for readiness
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------------
# A model's `modality` declares what it can do. "text" is the default and
# covers chat / completions / embeddings (the classic llama-server case).
# Vision is modelled as "text" + an mmproj artifact rather than a distinct
# modality, because it still runs on llama-server and serves /v1/chat/completions.
TEXT = "text"
ASR = "asr"
TTS = "tts"
IMAGE = "image"

ALL_MODALITIES = {TEXT, ASR, TTS, IMAGE}

# Maps an OpenAI-style endpoint to the modality required to serve it.
# Used by the routes layer to reject mismatches with a clear error.
ENDPOINT_MODALITY = {
    "chat/completions": TEXT,
    "completions": TEXT,
    "embeddings": TEXT,
    "audio/transcriptions": ASR,
    "audio/speech": TTS,
    "images/generations": IMAGE,
}


class Backend(Protocol):
    """Per-engine behaviour. Stateless; one instance is shared by all models."""

    name: str
    binary_name: str
    modalities: set[str]

    def build_args(
        self,
        binary: str,
        port: int,
        model_path: Path,
        entry: dict,
        merged_params: dict,
        artifacts: dict[str, Path],
    ) -> list[str]:
        """Return the full argv (including binary) to launch the subprocess."""
        ...

    def health_path(self) -> str:
        """HTTP path (relative to base_url) to poll until the server is ready."""
        ...


# ---------------------------------------------------------------------------
# llama-server backend (text / chat / embeddings / vision)
# ---------------------------------------------------------------------------
class LlamaServerBackend:
    name = "llama-server"
    binary_name = "llama-server"
    # Vision is served here too (chat/completions with image content), so this
    # backend covers the whole TEXT modality. mmproj is supplied via artifacts.
    modalities = {TEXT}

    # Params that take a value: registry/config key -> CLI flag.
    _PARAM_MAP = {
        "ctx_size": "--ctx-size",
        "n_gpu_layers": "--n-gpu-layers",
        "threads": "--threads",
        "parallel": "--parallel",
    }
    # Boolean flags: present-if-truthy.
    _FLAG_MAP = {
        "cont_batching": "--cont-batching",
        "mlock": "--mlock",
        "no_mmap": "--no-mmap",
        "embedding": "--embedding",
    }
    # Tri-state options that take on/off/auto in recent llama.cpp.
    _TRISTATE = {
        "flash_attn": "--flash-attn",
        "reasoning": "--reasoning",
    }

    def build_args(
        self,
        binary: str,
        port: int,
        model_path: Path,
        entry: dict,
        merged_params: dict,
        artifacts: dict[str, Path],
    ) -> list[str]:
        args = [
            binary,
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

        # Multimodal projector (vision). Present whenever the model declares an
        # mmproj artifact — this is what enables image input on llama-server.
        mmproj = artifacts.get("mmproj")
        if mmproj is not None:
            args += ["--mmproj", str(mmproj)]

        for key, flag in self._PARAM_MAP.items():
            if key in merged_params:
                args += [flag, str(merged_params[key])]

        for key, flag in self._TRISTATE.items():
            if key in merged_params:
                args += self._tristate(flag, merged_params[key])

        for key, flag in self._FLAG_MAP.items():
            if merged_params.get(key):
                args.append(flag)

        if "chat_template" in merged_params:
            args += ["--chat-template", str(merged_params["chat_template"])]

        return args

    @staticmethod
    def _tristate(flag: str, val) -> list[str]:
        """Render an on/off/auto option, accepting bool or string."""
        if val is True:
            return [flag, "on"]
        if val is False:
            return [flag, "off"]
        if isinstance(val, str) and val in ("on", "off", "auto"):
            return [flag, val]
        return []

    def health_path(self) -> str:
        return "/health"


# ---------------------------------------------------------------------------
# Registry of backends
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, Backend] = {
    LlamaServerBackend.name: LlamaServerBackend(),
}

DEFAULT_BACKEND = LlamaServerBackend.name


def get_backend(name: str | None) -> Backend:
    """Resolve a backend by name, falling back to the default (llama-server)."""
    key = (name or DEFAULT_BACKEND).strip()
    backend = _BACKENDS.get(key)
    if backend is None:
        raise ValueError(
            f"Unknown backend '{key}'. Registered backends: {sorted(_BACKENDS)}"
        )
    return backend


def register_backend(backend: Backend) -> None:
    """Register a new backend. Future modality backends call this at import."""
    _BACKENDS[backend.name] = backend
