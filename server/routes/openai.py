"""
routes/openai.py — Full OpenAI /v1 API proxy for Zallama

Implements:
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions   (streaming + non-streaming)
  POST /v1/completions        (streaming + non-streaming)
  POST /v1/embeddings
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..dependencies import get_pm, get_registry
from ..backends import ENDPOINT_MODALITY
from ..model_registry import ModelRegistry

router = APIRouter(prefix="/v1")


def _request_timeout(request: Request) -> float:
    """Non-streaming upstream timeout, from config (default 600s)."""
    try:
        return float(request.app.state.cfg["zallama"].get("request_timeout", 600))
    except Exception:
        return 600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_id_from_body(body: dict) -> str:
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")
    return model


async def _resolve_instance(model_name: str, pm, registry, endpoint: str | None = None):
    """Look up model in registry, enforce modality, and ensure it is running."""
    try:
        entry = registry.get(model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Modality guard: reject e.g. a tts model on /chat/completions with a clear
    # error instead of a confusing upstream failure.
    if endpoint is not None:
        required = ENDPOINT_MODALITY.get(endpoint)
        actual = ModelRegistry.modality_of(entry)
        if required is not None and actual != required:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' has modality '{actual}', "
                       f"which cannot serve /v1/{endpoint} (requires '{required}').",
            )
    try:
        model_path = registry.resolve_path(entry)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        inst = await pm.get_or_start(model_name, entry, model_path)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to start model: {e}")
    return inst


async def _stream_proxy(upstream_url: str, body: dict) -> AsyncIterator[bytes]:
    """Stream SSE from llama-server back to client."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", upstream_url, json=body) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                yield error_body
                return
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models(registry=Depends(get_registry), pm=Depends(get_pm)):
    models = registry.list_models()
    running = {r["name"] for r in pm.list_running()}
    data = []
    for m in models:
        name = m["name"]
        data.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zallama",
            "description": m.get("description", ""),
            "status": "running" if name in running else "available",
        })
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# GET /v1/models/{model_id}
# ---------------------------------------------------------------------------
@router.get("/models/{model_id:path}")
async def get_model(model_id: str, registry=Depends(get_registry), pm=Depends(get_pm)):
    try:
        entry = registry.get(model_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    running = {r["name"] for r in pm.list_running()}
    return {
        "id": entry["name"],
        "object": "model",
        "created": int(time.time()),
        "owned_by": "zallama",
        "description": entry.get("description", ""),
        "status": "running" if entry["name"] in running else "available",
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------
@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    pm=Depends(get_pm),
    registry=Depends(get_registry),
):
    body = await request.json()
    model_name = _model_id_from_body(body)
    inst = await _resolve_instance(model_name, pm, registry, endpoint="chat/completions")
    inst.touch()

    upstream_url = f"{inst.base_url}/v1/chat/completions"
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_proxy(upstream_url, body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        async with httpx.AsyncClient(timeout=_request_timeout(request)) as client:
            try:
                resp = await client.post(upstream_url, json=body)
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"llama-server error: {e}")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# POST /v1/completions
# ---------------------------------------------------------------------------
@router.post("/completions")
async def completions(
    request: Request,
    pm=Depends(get_pm),
    registry=Depends(get_registry),
):
    body = await request.json()
    model_name = _model_id_from_body(body)
    inst = await _resolve_instance(model_name, pm, registry, endpoint="completions")
    inst.touch()

    upstream_url = f"{inst.base_url}/v1/completions"
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            _stream_proxy(upstream_url, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        async with httpx.AsyncClient(timeout=_request_timeout(request)) as client:
            try:
                resp = await client.post(upstream_url, json=body)
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"llama-server error: {e}")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# POST /v1/embeddings
# ---------------------------------------------------------------------------
@router.post("/embeddings")
async def embeddings(
    request: Request,
    pm=Depends(get_pm),
    registry=Depends(get_registry),
):
    body = await request.json()
    model_name = _model_id_from_body(body)
    inst = await _resolve_instance(model_name, pm, registry, endpoint="embeddings")
    inst.touch()

    upstream_url = f"{inst.base_url}/v1/embeddings"
    async with httpx.AsyncClient(timeout=_request_timeout(request)) as client:
        try:
            resp = await client.post(upstream_url, json=body)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"llama-server error: {e}")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)
