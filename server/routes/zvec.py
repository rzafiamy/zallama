"""
routes/zvec.py — Zallama's built-in vector store (RAG retrieval).

Endpoints (all under /v1/zvec):

  POST   /v1/zvec/collections                 create a collection
  GET    /v1/zvec/collections                 list collections
  GET    /v1/zvec/collections/{name}          collection info
  DELETE /v1/zvec/collections/{name}          delete a collection
  POST   /v1/zvec/{collection}/upsert         add/replace documents (auto-embeds)
  POST   /v1/zvec/{collection}/query          semantic search (+ optional rerank)
  POST   /v1/zvec/{collection}/delete         delete documents by id

The store itself (server/zvec/store.py) only holds vectors; embedding and
reranking are done by calling Zallama's own /v1/embeddings and /v1/rerank
through the in-process proxy helpers, so zvec reuses the exact models, lifecycle
and eviction that everything else uses.
"""
from __future__ import annotations

import uuid

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from ..dependencies import get_pm, get_registry
from ..zvec import get_store
from ..zvec.store import CollectionExistsError, CollectionNotFoundError, Document
from .openai import _resolve_instance, _request_timeout

router = APIRouter(prefix="/v1/zvec")


def _rag_cfg(request: Request) -> dict:
    return request.app.state.cfg.get("rag", {})


async def _embed(request: Request, pm, registry, model: str, texts: list[str]) -> np.ndarray:
    """Embed texts via the embedding model's llama-server instance.

    Returns an (N, dim) float32 array. Raises HTTPException on failure.
    """
    inst = await _resolve_instance(model, pm, registry, endpoint="embeddings")
    inst.touch()
    upstream_url = f"{inst.base_url}/v1/embeddings"
    async with httpx.AsyncClient(timeout=_request_timeout(request)) as client:
        try:
            resp = await client.post(upstream_url, json={"model": model, "input": texts})
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"embedding error: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])
    data = resp.json().get("data", [])
    # OpenAI embeddings responses are not guaranteed ordered; sort by index.
    data.sort(key=lambda d: d.get("index", 0))
    vecs = [d["embedding"] for d in data]
    if not vecs:
        raise HTTPException(status_code=502, detail="embedding model returned no vectors")
    return np.asarray(vecs, dtype=np.float32)


def _embedding_model(body: dict, request: Request, fallback: str | None = None) -> str:
    model = (body.get("embedding_model") or fallback
             or _rag_cfg(request).get("embedding_model") or "").strip()
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No embedding model. Set 'embedding_model' in the request, "
                   "rag.embedding_model in config, or ZALLAMA_EMBEDDING_MODEL.",
        )
    return model


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------
@router.post("/collections")
async def create_collection(request: Request, pm=Depends(get_pm), registry=Depends(get_registry)):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="'name' is required")
    model = _embedding_model(body, request)

    # Determine the embedding dimension by embedding a probe string, unless the
    # caller supplied `dim` explicitly.
    dim = body.get("dim")
    if not dim:
        probe = await _embed(request, pm, registry, model, ["dimension probe"])
        dim = int(probe.shape[1])

    try:
        coll = get_store().create_collection(name, model, int(dim))
    except CollectionExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _coll_json(coll)


@router.get("/collections")
async def list_collections():
    return {"object": "list", "data": [_coll_json(c) for c in get_store().list_collections()]}


@router.get("/collections/{name}")
async def get_collection(name: str):
    try:
        return _coll_json(get_store().get_collection(name))
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    deleted = get_store().delete_collection(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"collection '{name}' not found")
    return {"deleted": True, "name": name}


def _coll_json(c) -> dict:
    return {
        "name": c.name,
        "embedding_model": c.embedding_model,
        "dim": c.dim,
        "created": c.created,
        "count": c.count,
    }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@router.post("/{collection}/upsert")
async def upsert(collection: str, request: Request, pm=Depends(get_pm), registry=Depends(get_registry)):
    body = await request.json()
    raw = body.get("documents")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="'documents' must be a non-empty list")

    docs: list[Document] = []
    for d in raw:
        if isinstance(d, str):
            docs.append(Document(doc_id=str(uuid.uuid4()), text=d))
        elif isinstance(d, dict):
            text = d.get("text")
            if not text:
                raise HTTPException(status_code=400, detail="each document needs 'text'")
            docs.append(Document(
                doc_id=str(d.get("id") or uuid.uuid4()),
                text=text,
                metadata=d.get("metadata") or {},
            ))
        else:
            raise HTTPException(status_code=400, detail="documents must be strings or objects")

    try:
        coll = get_store().get_collection(collection)
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    embeddings = await _embed(request, pm, registry, coll.embedding_model, [d.text for d in docs])
    n = get_store().upsert(collection, docs, embeddings)
    return {"upserted": n, "ids": [d.doc_id for d in docs]}


@router.post("/{collection}/query")
async def query(collection: str, request: Request, pm=Depends(get_pm), registry=Depends(get_registry)):
    body = await request.json()
    q = body.get("query")
    if not q or not isinstance(q, str):
        raise HTTPException(status_code=400, detail="'query' (string) is required")

    try:
        coll = get_store().get_collection(collection)
    except CollectionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    default_k = int(_rag_cfg(request).get("default_top_k", 5))
    top_k = int(body.get("top_k", default_k))
    where = body.get("filter") or body.get("where")

    qvec = await _embed(request, pm, registry, coll.embedding_model, [q])
    # When reranking, over-fetch candidates so the reranker has room to reorder.
    rerank_model = (body.get("rerank_model") or _rag_cfg(request).get("rerank_model") or "").strip()
    fetch_k = max(top_k, top_k * 4) if rerank_model else top_k
    hits = get_store().query(collection, qvec[0], top_k=fetch_k, where=where)

    if rerank_model and hits:
        hits = await _rerank(request, pm, registry, rerank_model, q, hits, top_k)

    return {
        "collection": collection,
        "results": [
            {"id": h.doc_id, "text": h.text, "metadata": h.metadata, "score": h.score}
            for h in hits[:top_k]
        ],
    }


@router.post("/{collection}/delete")
async def delete_documents(collection: str, request: Request):
    body = await request.json()
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="'ids' must be a non-empty list")
    n = get_store().delete_documents(collection, [str(i) for i in ids])
    return {"deleted": n}


async def _rerank(request, pm, registry, model, query, hits, top_k):
    """Reorder `hits` with a cross-encoder reranker, returning the top_k."""
    inst = await _resolve_instance(model, pm, registry, endpoint="rerank")
    inst.touch()
    upstream_url = f"{inst.base_url}/v1/rerank"
    payload = {"model": model, "query": query, "documents": [h.text for h in hits]}
    async with httpx.AsyncClient(timeout=_request_timeout(request)) as client:
        try:
            resp = await client.post(upstream_url, json=payload)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"rerank error: {e}")
    if resp.status_code != 200:
        # Reranking is best-effort; fall back to the vector-similarity order.
        return hits[:top_k]
    results = resp.json().get("results", [])
    reordered = []
    for r in sorted(results, key=lambda x: x.get("relevance_score", x.get("score", 0.0)), reverse=True):
        h = hits[r["index"]]
        h.score = r.get("relevance_score", r.get("score", h.score))
        reordered.append(h)
    return reordered[:top_k]
