"""
model_registry.py — Zallama Model Registry

Reads models/registry.yaml and provides model resolution utilities.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    def __init__(self, registry_path: Path, models_dir: str):
        self.registry_path = registry_path
        self.models_dir = Path(models_dir)
        self._models: dict[str, dict] = {}
        self.reload()

    def reload(self):
        """(Re)load the registry from disk."""
        self._models = {}
        if not self.registry_path.exists():
            return
        with open(self.registry_path) as f:
            data = yaml.safe_load(f) or {}
        for entry in data.get("models", []):
            name = entry.get("name", "").strip()
            if not name:
                continue
            self._models[name] = entry
            for alias in entry.get("aliases", []):
                self._models[alias.strip()] = entry

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
        self.reload()  # Always fresh read for live updates
        entry = self._models.get(name)
        if not entry:
            raise ModelNotFoundError(f"Model '{name}' not found in registry")
        return entry

    def resolve_path(self, entry: dict) -> Path:
        """Return the absolute path to the model GGUF file."""
        file_path = Path(entry["file"])
        if file_path.is_absolute():
            resolved = file_path
        else:
            resolved = self.models_dir / file_path
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model file not found: {resolved}\n"
                f"Update 'file' in models/registry.yaml or run: zallama add <name> <path>"
            )
        return resolved

    def add_model(self, name: str, file_path: str, params: dict | None = None, description: str = "") -> dict:
        """Register a new model in registry.yaml."""
        entry: dict[str, Any] = {
            "name": name,
            "file": str(file_path),
            "description": description,
        }
        if params:
            entry["params"] = params

        # Load existing registry
        data: dict = {"models": []}
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = yaml.safe_load(f) or {"models": []}
        if "models" not in data or data["models"] is None:
            data["models"] = []

        # Remove existing entry with same name
        data["models"] = [m for m in data["models"] if m.get("name") != name]
        data["models"].append(entry)

        with open(self.registry_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        self.reload()
        return entry

    def remove_model(self, name: str) -> bool:
        """Remove a model from registry.yaml."""
        if name not in self._models:
            return False
        data: dict = {"models": []}
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = yaml.safe_load(f) or {"models": []}
        if "models" not in data:
            return False
        canonical = self._models[name]["name"]
        data["models"] = [m for m in data["models"] if m.get("name") != canonical]
        with open(self.registry_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        self.reload()
        return True
