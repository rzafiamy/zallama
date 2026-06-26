"""
backends.py — Zallama Backend Abstraction

A *backend* knows how to turn a registered model entry into a runnable
subprocess and how to talk to it once running. The generic process lifecycle
(spawn, health-check, port assignment, LRU eviction, kill) lives in
process_manager.py and is backend-agnostic; everything that is specific to a
particular inference engine lives here.

This is the seam that lets new modalities (TTS, ASR, image generation) be added
as *new Backend subclasses* rather than as cross-cutting changes:

  - LlamaServerBackend    → text / chat / vision      (llama-server)
  - EmbeddingServerBackend→ embeddings (llama-server --embedding)
  - RerankServerBackend   → rerank     (llama-server --reranking)
  - ParakeetServerBackend → ASR        (parakeet-server)
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
# Reranking (cross-encoder relevance scoring). Runs on llama-server in
# --reranking mode, which is a *different launch mode* than a chat/embedding
# model, so it is modelled as its own modality (a reranker model can serve
# /v1/rerank but not /v1/chat/completions, and vice-versa).
RERANK = "rerank"
# Embeddings. Runs on llama-server with --embedding, which (like --reranking) is
# a distinct launch mode: an embedding model serves /v1/embeddings but not
# /v1/chat/completions. Modelling it as its own modality keeps it symmetric with
# rerank and lets the modality guard protect the endpoint.
EMBEDDING = "embedding"

ALL_MODALITIES = {TEXT, ASR, TTS, IMAGE, RERANK, EMBEDDING}

# Default backend for each modality. This is the canonical "which engine serves
# this modality" map; the downloader and the model-management API both resolve
# through default_backend_for() so a modality is wired to its backend in exactly
# one place. TEXT maps to None so a text entry stays clean and uses the registry
# default (llama-server). IMAGE has no backend yet (awaiting a diffusion backend).
MODALITY_BACKEND: dict[str, str | None] = {
    TEXT: None,                       # llama-server default (also vision)
    EMBEDDING: "embedding-server",
    RERANK: "rerank-server",
    ASR: "parakeet-server",
    TTS: "kokoro-server",
    IMAGE: None,                      # no backend implemented yet
}


def default_backend_for(modality: str | None) -> str | None:
    """Return the default backend name for a modality, validating the modality.

    Returns None for text (let the registry default apply). Raises ValueError
    for an unknown modality, or one whose backend is not implemented yet.
    """
    m = (modality or TEXT).strip().lower()
    if m not in ALL_MODALITIES:
        raise ValueError(
            f"Unknown modality '{m}'. Supported: {', '.join(sorted(ALL_MODALITIES))}."
        )
    if m not in MODALITY_BACKEND or (m != TEXT and MODALITY_BACKEND[m] is None):
        raise ValueError(f"Modality '{m}' has no backend implemented yet.")
    return MODALITY_BACKEND[m]


def validate_backend(name: str | None) -> str:
    """Validate a backend name against the registry, returning the resolved name."""
    return get_backend(name).name


# Maps an OpenAI-style endpoint to the modality required to serve it.
# Used by the routes layer to reject mismatches with a clear error.
ENDPOINT_MODALITY = {
    "chat/completions": TEXT,
    "completions": TEXT,
    "embeddings": EMBEDDING,
    "audio/transcriptions": ASR,
    "audio/speech": TTS,
    "images/generations": IMAGE,
    "rerank": RERANK,
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
        # MTP / speculative decoding. Requires an MTP GGUF variant whose draft
        # head is baked in (e.g. unsloth *-MTP-GGUF). spec_type "draft-mtp"
        # activates it; spec_draft_n_max tunes lookahead (2 is a good start).
        "spec_type": "--spec-type",
        "spec_draft_n_max": "--spec-draft-n-max",
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
# llama-server backend in reranking mode (cross-encoder relevance scoring)
# ---------------------------------------------------------------------------
class RerankServerBackend(LlamaServerBackend):
    """llama-server launched in reranking mode (--reranking / --pooling rank).

    Reuses the llama-server binary and all of LlamaServerBackend's param/flag
    handling, but is a distinct backend so it appears separately in `ps`, slots
    into the LRU/eviction lifecycle on its own, and is matched to the RERANK
    modality. It exposes POST /v1/rerank (and /rerank) upstream.

    A reranker GGUF (e.g. bge-reranker-v2-m3) must be a cross-encoder model;
    --reranking enables the rank pooling head and the /rerank route.
    """
    name = "rerank-server"
    binary_name = "llama-server"
    modalities = {RERANK}

    def build_args(
        self,
        binary: str,
        port: int,
        model_path: Path,
        entry: dict,
        merged_params: dict,
        artifacts: dict[str, Path],
    ) -> list[str]:
        args = super().build_args(
            binary, port, model_path, entry, merged_params, artifacts
        )
        # Enable reranking. --reranking turns on rank pooling and the /rerank
        # endpoint; it is mutually exclusive with --embedding upstream, so the
        # registry should not set `embedding` on a rerank model.
        if "--reranking" not in args:
            args.append("--reranking")
        return args


# ---------------------------------------------------------------------------
# llama-server backend in embedding mode (vector embeddings)
# ---------------------------------------------------------------------------
class EmbeddingServerBackend(LlamaServerBackend):
    """llama-server launched in embedding mode (--embedding).

    Like RerankServerBackend, this reuses the llama-server binary and param
    handling but is a distinct backend tied to the EMBEDDING modality, so it
    shows up separately in `ps` and the modality guard can keep chat traffic off
    an embedding model (and vice-versa). It serves POST /v1/embeddings.

    --embedding is forced on here, so an embedding model needs no `embedding:
    true` param (legacy entries that still set it are harmless — the flag is
    de-duplicated below).
    """
    name = "embedding-server"
    binary_name = "llama-server"
    modalities = {EMBEDDING}

    def build_args(
        self,
        binary: str,
        port: int,
        model_path: Path,
        entry: dict,
        merged_params: dict,
        artifacts: dict[str, Path],
    ) -> list[str]:
        args = super().build_args(
            binary, port, model_path, entry, merged_params, artifacts
        )
        # --embedding enables the embeddings endpoint; mutually exclusive with
        # --reranking upstream. De-dup in case a legacy entry set embedding=true.
        if "--embedding" not in args:
            args.append("--embedding")
        return args


# ---------------------------------------------------------------------------
# parakeet-server backend (ASR / speech-to-text)
# ---------------------------------------------------------------------------
class ParakeetServerBackend:
    """parakeet.cpp's example server — OpenAI-compatible transcription.

    Exposes POST /v1/audio/transcriptions (multipart, WAV input) and GET
    /health. It is single-model and serves one request at a time; zallama's
    per-model lifecycle (one process per model) maps onto that cleanly, and the
    routes layer simply proxies each request through.

    Build it with build-ggml-parakeet.cpp.sh, which installs `parakeet-server`
    into ./bin/.
    """
    name = "parakeet-server"
    binary_name = "parakeet-server"
    modalities = {ASR}

    # Params that take a value: registry/config key -> CLI flag.
    _PARAM_MAP = {
        "threads": "--threads",
        "cache_dir": "--cache-dir",
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
        for key, flag in self._PARAM_MAP.items():
            if key in merged_params:
                args += [flag, str(merged_params[key])]
        return args

    def health_path(self) -> str:
        return "/health"


# ---------------------------------------------------------------------------
# kokoro-server backend (TTS / text-to-speech)
# ---------------------------------------------------------------------------
class KokoroServerBackend:
    """kokoro.cpp's server — OpenAI-compatible text-to-speech.

    Exposes POST /v1/audio/speech (JSON in, WAV out) and GET /health. Build it
    with build-ggml-kokoro.cpp.sh, which installs `kokoro-server` into ./bin/.

    Note: kokoro's `--model` is a *resource directory* (not a single weights
    file), because the model loads several files (two ONNX models + a voice
    pack) by name. The downloader fetches them into one directory, and the
    registry's `file` for a kokoro model points at that directory.
    """
    name = "kokoro-server"
    binary_name = "kokoro-server"
    modalities = {TTS}

    def build_args(
        self,
        binary: str,
        port: int,
        model_path: Path,
        entry: dict,
        merged_params: dict,
        artifacts: dict[str, Path],
    ) -> list[str]:
        # kokoro-server accepts only --model/--host/--port; it has no launch-time
        # synthesis flags. `voice`/`speed` are per-request fields handled by the
        # /v1/audio/speech route (which also applies registry-param defaults), so
        # there is nothing else to forward here.
        #
        # kokoro-server wants the resource *directory*. The registry stores the
        # directory as this model's `file`, so use it directly; only fall back to
        # the parent if a file path was registered by mistake (and it isn't an
        # existing directory).
        resource_dir = model_path
        if model_path.suffix and not model_path.is_dir():
            resource_dir = model_path.parent
        return [
            binary,
            "--model", str(resource_dir),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]

    def health_path(self) -> str:
        return "/health"


# ---------------------------------------------------------------------------
# Registry of backends
# ---------------------------------------------------------------------------
_BACKENDS: dict[str, Backend] = {
    LlamaServerBackend.name: LlamaServerBackend(),
    EmbeddingServerBackend.name: EmbeddingServerBackend(),
    RerankServerBackend.name: RerankServerBackend(),
    ParakeetServerBackend.name: ParakeetServerBackend(),
    KokoroServerBackend.name: KokoroServerBackend(),
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
