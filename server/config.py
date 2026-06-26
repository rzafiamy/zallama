"""
config.py — Zallama Configuration Loader

Loads and merges:
  1. config/config.yaml  (or ~/.zallama/config.yaml)
  2. Environment variables (ZALLAMA_HOST, ZALLAMA_PORT, ...)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Default configuration (merged with user config)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "zallama": {
        "host": "127.0.0.1",   # localhost by default; set 0.0.0.0 to expose
        "port": 11435,
        "models_dir": "~/.zallama/models",
        "logs_dir": "~/.zallama/logs",
        "webui": True,
        "log_level": "info",
        "api_key": "",          # if set, required as Bearer token on /v1 and /api
        "request_timeout": 600, # seconds for non-streaming upstream proxy calls
    },
    "llama_server": {
        "binary": "",
        "port_start": 8100,
        "startup_timeout": 60,
        "idle_timeout": 300,
        "max_loaded_models": 0,  # 0 = unlimited; cap concurrent loaded models
        "mem_budget_gb": 0,      # 0 = unlimited; evict LRU to fit declared mem_gb
        "mem_init_gb": 2,        # fallback per-model cost when mem_gb is undeclared
        "default_params": {
            "ctx_size": 4096,
            "n_gpu_layers": 99,
            "threads": 8,
            "flash_attn": True,
            "cont_batching": True,
            "parallel": 4,
            "mlock": False,
            "no_mmap": False,
        },
    },
    # Retrieval-Augmented Generation: the built-in "zvec" vector store and the
    # default models it leans on. zvec embeds/queries by calling Zallama's own
    # /v1/embeddings (embedding_model) and can rerank with rerank_model.
    "rag": {
        "embedding_model": "",   # default model for zvec embed/query (registry name)
        "rerank_model": "",      # default model for /v1/rerank and zvec rerank
        "zvec_dir": "~/.zallama/zvec",  # SQLite vector store location
        "default_top_k": 5,      # default candidates returned by a zvec query
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _expand_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).expanduser().resolve()


def _find_config_file() -> Path | None:
    """Search for config.yaml in common locations."""
    candidates = [
        Path(__file__).parent.parent / "config" / "config.yaml",
        Path.home() / ".zallama" / "config.yaml",
        Path("/etc/zallama/config.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_config() -> dict[str, Any]:
    """Load and return the merged Zallama configuration."""
    cfg = _deep_merge({}, DEFAULTS)

    config_file = _find_config_file()
    if config_file:
        with open(config_file) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    # Environment variable overrides
    env_map = {
        "ZALLAMA_HOST": ("zallama", "host"),
        "ZALLAMA_PORT": ("zallama", "port"),
        "ZALLAMA_MODELS_DIR": ("zallama", "models_dir"),
        "ZALLAMA_LOGS_DIR": ("zallama", "logs_dir"),
        "ZALLAMA_LOG_LEVEL": ("zallama", "log_level"),
        "LLAMA_SERVER_BINARY": ("llama_server", "binary"),
        "LLAMA_GPU_LAYERS": ("llama_server", "default_params", "n_gpu_layers"),
        "ZALLAMA_EMBEDDING_MODEL": ("rag", "embedding_model"),
        "ZALLAMA_RERANK_MODEL": ("rag", "rerank_model"),
        "ZALLAMA_ZVEC_DIR": ("rag", "zvec_dir"),
        "ZALLAMA_RAG_TOP_K": ("rag", "default_top_k"),
    }
    for env_key, path in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            node = cfg
            for p in path[:-1]:
                node = node[p]
            # Try numeric coercion
            try:
                node[path[-1]] = int(val)
            except ValueError:
                node[path[-1]] = val

    # Resolve paths
    cfg["zallama"]["models_dir"] = str(_expand_path(cfg["zallama"]["models_dir"]))
    cfg["zallama"]["logs_dir"] = str(_expand_path(cfg["zallama"]["logs_dir"]))
    cfg["rag"]["zvec_dir"] = str(_expand_path(cfg["rag"]["zvec_dir"]))

    # Ensure directories exist
    Path(cfg["zallama"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["zallama"]["logs_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["rag"]["zvec_dir"]).mkdir(parents=True, exist_ok=True)

    return cfg


def resolve_binary(cfg: dict, binary_name: str = "llama-server") -> str:
    """Find an inference backend binary path.

    Lookup order: explicit config override (llama-server only) → ./bin/<name> →
    ~/.zallama/bin/<name> → PATH. The name is per-backend so TTS/ASR/image
    backends resolve their own executables (e.g. whisper-server, sd-server).
    """
    # The `binary` config key historically pins llama-server specifically.
    if binary_name == "llama-server":
        explicit = cfg["llama_server"].get("binary", "").strip()
        if explicit and Path(explicit).is_file():
            return explicit

    root = Path(__file__).parent.parent
    candidates = [
        root / "bin" / binary_name,
        Path.home() / ".zallama" / "bin" / binary_name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    import shutil
    found = shutil.which(binary_name)
    if found:
        return found

    raise FileNotFoundError(
        f"'{binary_name}' binary not found. Place it in ./bin/{binary_name}, "
        f"~/.zallama/bin/{binary_name}, or on your PATH."
    )
