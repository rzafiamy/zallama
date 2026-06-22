"""
routes/models.py — Zallama model management API

Endpoints:
  GET    /api/models          — list all registered models
  POST   /api/models/add      — register a new model
  DELETE /api/models/{name}   — remove model from registry
  GET    /api/ps              — list running llama-server processes
  POST   /api/models/{name}/load    — pre-load a model
  POST   /api/models/{name}/unload  — stop a running model
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies import get_pm, get_registry, get_dm

router = APIRouter(prefix="/api")


class PullModelRequest(BaseModel):
    model: str



class AddModelRequest(BaseModel):
    name: str
    file: str                   # Absolute or relative path to .gguf
    description: str = ""
    aliases: list[str] = []
    params: dict = {}


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models(registry=Depends(get_registry), pm=Depends(get_pm)):
    models = registry.list_models()
    running_map = {r["name"]: r for r in pm.list_running()}
    result = []
    for m in models:
        name = m["name"]
        file_path = m.get("file", "")
        # Check if file exists
        try:
            abs_path = str(registry.resolve_path(m))
            file_ok = True
            file_size = os.path.getsize(abs_path)
        except Exception:
            abs_path = file_path
            file_ok = False
            file_size = 0

        result.append({
            "name": name,
            "description": m.get("description", ""),
            "aliases": m.get("aliases", []),
            "file": abs_path,
            "file_size": file_size,
            "file_ok": file_ok,
            "params": m.get("params", {}),
            "running": name in running_map,
            "port": running_map.get(name, {}).get("port"),
        })
    return {"models": result}


# ---------------------------------------------------------------------------
# POST /api/models/add
# ---------------------------------------------------------------------------
@router.post("/models/add")
async def add_model(req: AddModelRequest, registry=Depends(get_registry)):
    file_path = Path(req.file).expanduser().resolve()
    if not file_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {file_path}. Provide an absolute path to an existing .gguf file."
        )
    entry = registry.add_model(
        name=req.name,
        file_path=str(file_path),
        params=req.params or None,
        description=req.description,
    )
    return {"status": "added", "model": entry}


# ---------------------------------------------------------------------------
# DELETE /api/models/{name}
# ---------------------------------------------------------------------------
@router.delete("/models/{name:path}")
async def remove_model(name: str, registry=Depends(get_registry), pm=Depends(get_pm)):
    # Stop if running
    await pm.stop(name)
    ok = registry.remove_model(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not in registry")
    return {"status": "removed", "name": name}


# ---------------------------------------------------------------------------
# GET /api/ps  (running processes)
# ---------------------------------------------------------------------------
@router.get("/ps")
async def list_running(pm=Depends(get_pm)):
    return {"processes": pm.list_running()}


# ---------------------------------------------------------------------------
# POST /api/models/{name}/load
# ---------------------------------------------------------------------------
@router.post("/models/{name:path}/load")
async def load_model(name: str, pm=Depends(get_pm), registry=Depends(get_registry)):
    try:
        entry = registry.get(name)
        model_path = registry.resolve_path(entry)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        inst = await pm.get_or_start(name, entry, model_path)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "status": "running",
        "name": name,
        "port": inst.port,
        "base_url": inst.base_url,
    }


# ---------------------------------------------------------------------------
# POST /api/models/{name}/unload
# ---------------------------------------------------------------------------
@router.post("/models/{name:path}/unload")
async def unload_model(name: str, pm=Depends(get_pm)):
    stopped = await pm.stop(name)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"Model '{name}' is not running")
    return {"status": "stopped", "name": name}


# ---------------------------------------------------------------------------
# POST /api/models/pull
# ---------------------------------------------------------------------------
@router.post("/models/pull")
async def pull_model(req: PullModelRequest, dm=Depends(get_dm)):
    try:
        msg = await dm.start_pull(req.model)
        return {"status": "started", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/models/pull/status
# ---------------------------------------------------------------------------
@router.get("/models/pull/status")
async def pull_status(dm=Depends(get_dm)):
    return {"downloads": dm.list_tasks()}


# ---------------------------------------------------------------------------
# GET /api/models/shorthands
# ---------------------------------------------------------------------------
@router.get("/models/shorthands")
async def get_shorthands(dm=Depends(get_dm)):
    return {"shorthands": dm.get_shorthands()}

