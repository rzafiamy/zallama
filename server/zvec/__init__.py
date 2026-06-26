"""
zvec — Zallama's vector store for RAG.

Backed by the `zvec` library (Alibaba, Apache-2.0): an in-process, HNSW-indexed
vector database that embeds directly into the daemon (no server to run). The
store lives under `rag.zvec_dir` — one collection directory per collection plus a
small `collections.json` manifest.

Collections hold documents (id, text, metadata, embedding). Embeddings are not
computed here — the routes layer calls Zallama's own /v1/embeddings and hands
the vectors to the store, so the store never needs a model of its own.

The VectorStore / Collection / Document API in store.py is engine-agnostic, so
the underlying library is an implementation detail of that module.
"""
from .store import VectorStore, Collection, Document, get_store

__all__ = ["VectorStore", "Collection", "Document", "get_store"]
