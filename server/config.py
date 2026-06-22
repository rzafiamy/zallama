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
        "host": "0.0.0.0",
        "port": 11435,
        "models_dir": "~/.zallama/models",
        "logs_dir": "~/.zallama/logs",
        "webui": True,
        "log_level": "info",
    },
    "llama_server": {
        "binary": "",
        "port_start": 8100,
        "startup_timeout": 60,
        "idle_timeout": 300,
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

    # Ensure directories exist
    Path(cfg["zallama"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["zallama"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    return cfg


def resolve_binary(cfg: dict) -> str:
    """Find llama-server binary path."""
    explicit = cfg["llama_server"].get("binary", "").strip()
    if explicit and Path(explicit).is_file():
        return explicit

    # Search relative to zallama root
    root = Path(__file__).parent.parent
    candidates = [
        root / "bin" / "llama-server",
        Path.home() / ".zallama" / "bin" / "llama-server",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Fall back to PATH
    import shutil
    found = shutil.which("llama-server")
    if found:
        return found

    raise FileNotFoundError(
        "llama-server binary not found. Set 'llama_server.binary' in config.yaml "
        "or place it in ./bin/llama-server"
    )
