"""
zvec/store.py — vector store backed by the `zvec` library (Alibaba, Apache-2.0).

`zvec` is an in-process, HNSW-indexed vector database (compiled extension). This
module wraps it behind the same `VectorStore` / `Collection` / `Document` API the
routes and CLI already use, so the engine swap is invisible above this layer.

Layout under `rag.zvec_dir`:
  collections.json          manifest: name -> {dir, embedding_model, dim, created}
  <name>/                   one zvec collection directory per collection

Why a manifest: a zvec collection is a self-contained directory opened by path;
the library has no global "list every collection" call, and we also need to
remember each collection's embedding model and creation time. The manifest is
the small bit of bookkeeping that gives us `list_collections()` and stable
metadata across restarts.

Scores: zvec returns a COSINE *distance* (0.0 == identical, larger == less
similar). We convert to a similarity (`1 - distance`) on the way out so callers
keep the "higher is better" convention.

Metadata filtering: the route-level `where` filter is an arbitrary exact-match
dict whose keys aren't known at schema-creation time, so it is applied in Python
after the vector search (over-fetching to compensate). The HNSW vector search
itself runs natively in zvec.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zvec


# A zvec collection name must pass an identifier-style regex (no leading digit,
# no '-' / ':' / '.'). Our public collection names are looser, so we keep the
# user-facing name in the manifest and use a sanitized name inside zvec.
_VECTOR_FIELD = "embedding"
_TEXT_FIELD = "text"
_META_FIELD = "metadata"


class CollectionNotFoundError(Exception):
    pass


class CollectionExistsError(Exception):
    pass


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


@dataclass
class Collection:
    name: str
    embedding_model: str
    dim: int
    created: float
    count: int = 0


def _safe_name(name: str) -> str:
    """Map a user collection name to a zvec-legal schema name.

    zvec requires a C-identifier-ish name; prefix and replace illegal chars so
    e.g. "my-notes:v1" -> "c_my_notes_v1". The user name is preserved in the
    manifest, so this only affects the on-disk schema name.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    return f"c_{cleaned}"


class VectorStore:
    def __init__(self, zvec_dir: str | Path):
        self.root = Path(zvec_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "collections.json"
        self._lock = threading.Lock()
        self._open: dict[str, Any] = {}  # name -> open zvec.Collection
        self._manifest: dict[str, dict] = self._load_manifest()

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------
    def _load_manifest(self) -> dict[str, dict]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=2)
        tmp.replace(self.manifest_path)

    def _collection_dir(self, name: str) -> Path:
        return self.root / _safe_name(name)

    def _open_collection(self, name: str):
        """Open (and cache) the underlying zvec collection for `name`."""
        if name in self._open:
            return self._open[name]
        if name not in self._manifest:
            raise CollectionNotFoundError(f"collection '{name}' not found")
        coll = zvec.open(str(self._collection_dir(name)))
        self._open[name] = coll
        return coll

    # -----------------------------------------------------------------------
    # Collections
    # -----------------------------------------------------------------------
    def create_collection(self, name: str, embedding_model: str, dim: int) -> Collection:
        name = name.strip()
        if not name:
            raise ValueError("collection name is required")
        with self._lock:
            if name in self._manifest:
                raise CollectionExistsError(f"collection '{name}' already exists")
            schema = zvec.CollectionSchema(
                name=_safe_name(name),
                fields=[
                    zvec.FieldSchema(_TEXT_FIELD, zvec.DataType.STRING),
                    zvec.FieldSchema(_META_FIELD, zvec.DataType.STRING, nullable=True),
                ],
                vectors=zvec.VectorSchema(
                    _VECTOR_FIELD,
                    zvec.DataType.VECTOR_FP32,
                    dimension=int(dim),
                    index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
                ),
            )
            coll = zvec.create_and_open(path=str(self._collection_dir(name)), schema=schema)
            self._open[name] = coll
            created = time.time()
            self._manifest[name] = {
                "dir": _safe_name(name),
                "embedding_model": embedding_model,
                "dim": int(dim),
                "created": created,
            }
            self._save_manifest()
        return Collection(name, embedding_model, int(dim), created, 0)

    def get_collection(self, name: str) -> Collection:
        if name not in self._manifest:
            raise CollectionNotFoundError(f"collection '{name}' not found")
        m = self._manifest[name]
        return Collection(name, m["embedding_model"], m["dim"], m["created"], self._count(name))

    def list_collections(self) -> list[Collection]:
        out = []
        for name, m in sorted(self._manifest.items(), key=lambda kv: kv[1]["created"]):
            out.append(Collection(name, m["embedding_model"], m["dim"], m["created"], self._count(name)))
        return out

    def delete_collection(self, name: str) -> bool:
        with self._lock:
            if name not in self._manifest:
                return False
            coll = self._open.pop(name, None)
            try:
                if coll is None:
                    coll = zvec.open(str(self._collection_dir(name)))
                coll.destroy()
            except Exception:
                pass  # best-effort; manifest removal below is what makes it "gone"
            self._manifest.pop(name, None)
            self._save_manifest()
        return True

    def _count(self, name: str) -> int:
        try:
            stats = self._open_collection(name).stats
            # stats exposes a doc count; tolerate either attribute or dict form.
            raw = getattr(stats, "num_docs", None)
            if raw is None:
                data = json.loads(str(stats)) if not isinstance(stats, dict) else stats
                raw = data.get("doc_count", 0)
            return int(raw)
        except Exception:
            return 0

    # -----------------------------------------------------------------------
    # Documents
    # -----------------------------------------------------------------------
    def upsert(self, collection: str, docs: list[Document], embeddings: np.ndarray) -> int:
        coll = self.get_collection(collection)  # validates existence
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(docs):
            raise ValueError("embeddings must be (len(docs), dim)")
        if embeddings.shape[1] != coll.dim:
            raise ValueError(f"embedding dim {embeddings.shape[1]} != collection dim {coll.dim}")
        zdocs = [
            zvec.Doc(
                id=d.doc_id,
                vectors={_VECTOR_FIELD: embeddings[i].tolist()},
                fields={
                    _TEXT_FIELD: d.text,
                    _META_FIELD: json.dumps(d.metadata or {}, ensure_ascii=False),
                },
            )
            for i, d in enumerate(docs)
        ]
        c = self._open_collection(collection)
        with self._lock:
            c.upsert(zdocs)
            c.flush()
        return len(zdocs)

    def delete_documents(self, collection: str, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        self.get_collection(collection)
        c = self._open_collection(collection)
        with self._lock:
            c.delete(doc_ids)
            c.flush()
        # zvec.delete does not report a count; report the number requested.
        return len(doc_ids)

    def query(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        coll = self.get_collection(collection)
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != coll.dim:
            raise ValueError(f"query dim {q.shape[0]} != collection dim {coll.dim}")

        # Over-fetch when a metadata filter is present, since it's applied in
        # Python after the native vector search.
        fetch_k = max(top_k * 5, top_k) if where else top_k
        c = self._open_collection(collection)
        hits = c.query(
            queries=zvec.Query(field_name=_VECTOR_FIELD, vector=q.tolist()),
            topk=max(1, fetch_k),
            output_fields=[_TEXT_FIELD, _META_FIELD],
        )

        results: list[Document] = []
        for h in hits:
            fields = h.fields or {}
            meta = {}
            raw = fields.get(_META_FIELD)
            if raw:
                try:
                    meta = json.loads(raw)
                except (ValueError, TypeError):
                    meta = {}
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            # zvec score is a COSINE distance (0 == identical); expose similarity.
            distance = float(h.score) if h.score is not None else 0.0
            results.append(Document(
                doc_id=h.id,
                text=fields.get(_TEXT_FIELD, ""),
                metadata=meta,
                score=1.0 - distance,
            ))
            if len(results) >= top_k:
                break
        return results


# ---------------------------------------------------------------------------
# Process-wide singleton (set up at startup by main.py)
# ---------------------------------------------------------------------------
_store: VectorStore | None = None


def get_store() -> VectorStore:
    if _store is None:
        raise RuntimeError("zvec store not initialized")
    return _store


def init_store(zvec_dir: str | Path) -> VectorStore:
    global _store
    _store = VectorStore(zvec_dir)
    return _store
