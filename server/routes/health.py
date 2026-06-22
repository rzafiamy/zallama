"""
routes/health.py — Zallama health and system info
"""
from __future__ import annotations

import platform
import time

from fastapi import APIRouter, Depends

from ..dependencies import get_pm, get_registry

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "zallama", "time": time.time()}


@router.get("/api/health")
async def api_health(pm=Depends(get_pm), registry=Depends(get_registry)):
    models = registry.list_models()
    running = pm.list_running()
    return {
        "status": "ok",
        "service": "zallama",
        "version": "1.0.0",
        "time": time.time(),
        "platform": platform.platform(),
        "models_registered": len(models),
        "models_running": len(running),
    }
