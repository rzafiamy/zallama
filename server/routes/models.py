"""
routes/models.py — Zallama model management API

Endpoints:
  GET    /api/models          — list all registered models
  POST   /api/models/add      — register a new model
  DELETE /api/models/{name}   — remove model from registry
  GET    /api/ps              — list running llama-server processes
  POST   /api/models/{name}/load    — pre-load a model
  POST   /api/models/{name}/unload  — stop a running model
  POST   /api/models/{name}/reload  — restart a running model to apply param changes
"""
from __future__ import annotations

import os
from pathlib import Path
import httpx

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
    modality: str = "text"      # text | asr | tts | image
    backend: str = "llama-server"
    artifacts: dict = {}        # e.g. {"mmproj": "/path/to/clip.gguf"} for vision
    mem_gb: float = 0           # declared memory cost; 0 = infer/fallback


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
            "modality": m.get("modality", "text"),
            "backend": m.get("backend", "llama-server"),
            "artifacts": m.get("artifacts", {}),
            "mem_gb": m.get("mem_gb", 0),
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
        modality=req.modality,
        backend=req.backend,
        artifacts=req.artifacts or None,
        aliases=req.aliases or None,
        mem_gb=req.mem_gb or None,
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
    return {"processes": pm.list_running(), "memory": pm.memory_status()}


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
# POST /api/models/{name}/reload
# ---------------------------------------------------------------------------
@router.post("/models/{name:path}/reload")
async def reload_model(name: str, pm=Depends(get_pm), registry=Depends(get_registry)):
    """Restart a running backend so it picks up the latest registry params.

    No-op if the model isn't running: live params already reload from disk, so a
    stopped model applies the new params on its next load.
    """
    try:
        entry = registry.get(name)
        model_path = registry.resolve_path(entry)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not pm.is_running(name):
        return {"status": "not_running", "name": name}
    await pm.stop(name)
    try:
        inst = await pm.get_or_start(name, entry, model_path)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "status": "reloaded",
        "name": name,
        "port": inst.port,
        "base_url": inst.base_url,
    }


# ---------------------------------------------------------------------------
# POST /api/models/pull
# ---------------------------------------------------------------------------
@router.post("/models/pull")
async def pull_model(req: PullModelRequest, dm=Depends(get_dm)):
    try:
        resolved_name, msg = await dm.start_pull(req.model)
        return {"status": "started", "message": msg, "model": resolved_name}
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


# ---------------------------------------------------------------------------
# GET /api/models/search
# ---------------------------------------------------------------------------
@router.get("/models/search")
async def search_huggingface(q: str):
    q = q.strip()
    if not q:
        return {"results": [], "type": "repo_list"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If the query contains a slash, we assume it's a specific repository lookup
        if "/" in q:
            url = f"https://huggingface.co/api/models/{q}"
            resp = await client.get(url)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"HuggingFace API error: {resp.reason_phrase}")
            
            data = resp.json()
            siblings = data.get("siblings", [])
            # Filter files ending in .gguf
            files = [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]
            
            return {
                "type": "file_list",
                "repo": q,
                "files": files,
                "downloads": data.get("downloads", 0),
                "likes": data.get("likes", 0)
            }
        else:
            # Query for repositories
            url = "https://huggingface.co/api/models"
            params = {
                "search": q,
                "filter": "gguf",
                "limit": 15,
                "sort": "downloads",
                "direction": "-1"
            }
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"HuggingFace API error: {resp.reason_phrase}")
            
            data = resp.json()
            results = []
            for item in data:
                results.append({
                    "id": item.get("id"),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": item.get("tags", []),
                })
            
            return {
                "type": "repo_list",
                "query": q,
                "results": results
            }

