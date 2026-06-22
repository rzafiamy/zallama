"""
dependencies.py — FastAPI dependency injection for Zallama

Provides shared singletons: ProcessManager, ModelRegistry
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .process_manager import ProcessManager
    from .model_registry import ModelRegistry
    from .download_manager import DownloadManager

# These are set at startup by main.py
_pm: "ProcessManager | None" = None
_registry: "ModelRegistry | None" = None
_dm: "DownloadManager | None" = None


def set_pm(pm: "ProcessManager"):
    global _pm
    _pm = pm


def set_registry(registry: "ModelRegistry"):
    global _registry
    _registry = registry


def set_dm(dm: "DownloadManager"):
    global _dm
    _dm = dm


def get_pm() -> "ProcessManager":
    if _pm is None:
        raise RuntimeError("ProcessManager not initialized")
    return _pm


def get_registry() -> "ModelRegistry":
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialized")
    return _registry


def get_dm() -> "DownloadManager":
    if _dm is None:
        raise RuntimeError("DownloadManager not initialized")
    return _dm
