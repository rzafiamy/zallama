"""
model_registry.py — Zallama Model Registry

Reads models/registry.yaml and provides model resolution utilities.

A registry entry looks like:

    name: my-model
    file: model.gguf            # primary artifact (relative to models_dir or absolute)
    modality: text              # text | asr | tts | image   (default: text)
    backend: llama-server       # which Backend runs it       (default: llama-server)
    artifacts:                  # OPTIONAL extra files (mmproj, vocoder, ...)
      mmproj: clip.gguf
    params: { ctx_size: 4096, ... }
    aliases: [ ... ]
    description: "..."

`modality`, `backend`, and `artifacts` are all optional and default to the
classic single-gguf llama-server text model, so existing registries keep working
unchanged.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from .backends import DEFAULT_BACKEND, TEXT


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self, registry_path: Path, models_dir: str):
        self.registry_path = registry_path
        self.models_dir = Path(models_dir)
        self._models: dict[str, dict] = {}
        self._mtime: float = -1.0
        self._write_lock = threading.Lock()
        self.reload(force=True)

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------
    def reload(self, force: bool = False):
        """(Re)load the registry from disk, skipping the parse if unchanged.

        The hot path (every inference request) calls get() which calls this; an
        mtime check keeps that from re-parsing the YAML on every request.
        """
        if not self.registry_path.exists():
            self._models = {}
            self._mtime = -1.0
            return

        mtime = self.registry_path.stat().st_mtime
        if not force and mtime == self._mtime:
            return

        with open(self.registry_path) as f:
            data = yaml.safe_load(f) or {}

        models: dict[str, dict] = {}
        for entry in data.get("models", []):
            name = entry.get("name", "").strip()
            if not name:
                continue
            models[name] = entry
            for alias in entry.get("aliases", []):
                models[alias.strip()] = entry

        self._models = models
        self._mtime = mtime

    def list_models(self) -> list[dict]:
        """Return deduplicated list of model entries."""
        seen = set()
        result = []
        for entry in self._models.values():
            n = entry["name"]
            if n not in seen:
                seen.add(n)
                result.append(entry)
        return result

    def get(self, name: str) -> dict:
        """Resolve a model by name or alias. Raises ModelNotFoundError if not found."""
        self.reload()  # cheap: mtime-gated, only re-parses on change
        entry = self._models.get(name)
        if not entry:
            raise ModelNotFoundError(f"Model '{name}' not found in registry")
        return entry

    # -----------------------------------------------------------------------
    # Entry introspection helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def modality_of(entry: dict) -> str:
        return (entry.get("modality") or TEXT).strip()

    @staticmethod
    def backend_of(entry: dict) -> str:
        return (entry.get("backend") or DEFAULT_BACKEND).strip()

    # -----------------------------------------------------------------------
    # Path resolution
    # -----------------------------------------------------------------------
    def _resolve_one(self, raw: str, *, kind: str) -> Path:
        p = Path(raw)
        resolved = p if p.is_absolute() else self.models_dir / p
        if not resolved.exists():
            raise FileNotFoundError(
                f"{kind} file not found: {resolved}\n"
                f"Update the registry entry or run: zallama add <name> <path>"
            )
        return resolved

    def resolve_path(self, entry: dict) -> Path:
        """Return the absolute path to the model's primary artifact (the gguf)."""
        return self._resolve_one(entry["file"], kind="Model")

    def resolve_artifacts(self, entry: dict) -> dict[str, Path]:
        """Resolve all secondary artifacts (mmproj, vocoder, ...) to abs paths."""
        artifacts: dict[str, Path] = {}
        for key, raw in (entry.get("artifacts") or {}).items():
            if not raw:
                continue
            artifacts[key] = self._resolve_one(str(raw), kind=f"Artifact '{key}'")
        return artifacts

    # -----------------------------------------------------------------------
    # Mutation (write-locked; read-modify-write of the YAML)
    # -----------------------------------------------------------------------
    def add_model(
        self,
        name: str,
        file_path: str,
        params: dict | None = None,
        description: str = "",
        modality: str | None = None,
        backend: str | None = None,
        artifacts: dict | None = None,
        aliases: list[str] | None = None,
        mem_gb: float | None = None,
        pinned: bool = False,
    ) -> dict:
        """Register (or replace) a model in registry.yaml."""
        entry: dict[str, Any] = {
            "name": name,
            "file": str(file_path),
            "description": description,
        }
        if aliases:
            entry["aliases"] = aliases
        if modality and modality != TEXT:
            entry["modality"] = modality
        if backend and backend != DEFAULT_BACKEND:
            entry["backend"] = backend
        if artifacts:
            entry["artifacts"] = artifacts
        if mem_gb:
            entry["mem_gb"] = mem_gb
        if pinned:
            entry["pinned"] = True
        if params:
            entry["params"] = params

        with self._write_lock:
            data = self._read_data()
            data["models"] = [m for m in data["models"] if m.get("name") != name]
            data["models"].append(entry)
            self._write_data(data)
            self.reload(force=True)
        return entry

    def remove_model(self, name: str) -> bool:
        """Remove a model from registry.yaml."""
        if name not in self._models:
            return False
        canonical = self._models[name]["name"]
        with self._write_lock:
            data = self._read_data()
            data["models"] = [m for m in data["models"] if m.get("name") != canonical]
            self._write_data(data)
            self.reload(force=True)
        return True

    # -----------------------------------------------------------------------
    # YAML I/O
    # -----------------------------------------------------------------------
    def _read_data(self) -> dict:
        data: dict = {"models": []}
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = yaml.safe_load(f) or {"models": []}
        if not data.get("models"):
            data["models"] = []
        return data

    def _write_data(self, data: dict):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
