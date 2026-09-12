from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from app.core.config import get_settings
from app.db.session import pgvector_ready, postgres_ready
from app.models.schemas import ChunkRecord
from app.services.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)


class HybridVectorStore(VectorStore):
    """pgvector for new indexes; FAISS files for sessions created before the switch."""

    name = "pgvector"
    persists_chunks = True

    def __init__(self, primary: VectorStore, fallback: VectorStore) -> None:
        self.primary = primary
        self.fallback = fallback

    async def upsert(self, tenant_id: str, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        await self.primary.upsert(tenant_id, chunks, vectors)

    async def search(self, tenant_id: str, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if await self.primary.exists(tenant_id):
            return await self.primary.search(tenant_id, query_vector, top_k)
        logger.info("No pgvector embeddings for tenant %s — using FAISS fallback", tenant_id)
        return await self.fallback.search(tenant_id, query_vector, top_k)

    async def delete_tenant(self, tenant_id: str) -> None:
        await self.primary.delete_tenant(tenant_id)
        await self.fallback.delete_tenant(tenant_id)

    async def exists(self, tenant_id: str) -> bool:
        return await self.primary.exists(tenant_id) or await self.fallback.exists(tenant_id)

    async def backfill_missing_embeddings(self, tenant_id: str) -> bool:
        fn = getattr(self.primary, "backfill_missing_embeddings", None)
        if callable(fn):
            return await fn(tenant_id)
        return False

    async def preload(self, tenant_id: str) -> None:
        await self.backfill_missing_embeddings(tenant_id)


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    prefer = (settings.vector_store or "auto").strip().lower()
    if prefer == "faiss":
        from app.services.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore()
    if prefer == "pgvector" or (prefer == "auto" and settings.postgres_enabled):
        if postgres_ready() and pgvector_ready():
            from app.services.vectorstore.faiss_store import FaissVectorStore
            from app.services.vectorstore.pgvector_store import PgVectorStore

            return HybridVectorStore(PgVectorStore(), FaissVectorStore())
        if prefer == "pgvector":
            from app.services.vectorstore.pgvector_store import PgVectorStore

            return PgVectorStore()
    from app.services.vectorstore.faiss_store import FaissVectorStore

    return FaissVectorStore()


def reset_vector_store() -> None:
    get_vector_store.cache_clear()


def vector_store_name() -> str:
    store = get_vector_store()
    return getattr(store, "name", store.__class__.__name__.lower())
