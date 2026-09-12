from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import ChunkRow
from app.db.repository import db_clear_tenant_knowledge, db_save_document_with_chunks
from app.db.session import postgres_ready, session_scope
from app.models.schemas import ChunkRecord
from app.services.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class PgVectorStore(VectorStore):
    name = "pgvector"
    persists_chunks = True

    def __init__(self) -> None:
        self.settings = get_settings()

    async def upsert(self, tenant_id: str, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        if not chunks:
            raise ValueError("No chunks to index")
        vectors = _l2_normalize(vectors)
        if vectors.shape[0] != len(chunks):
            raise ValueError("Vector/chunk length mismatch")
        expected = self.settings.embedding_dimensions
        if vectors.shape[1] != expected:
            raise ValueError(f"Expected {expected}-d embeddings, got {vectors.shape[1]}")
        if not postgres_ready():
            raise RuntimeError("PostgreSQL is not connected — cannot use pgvector")

        await db_save_document_with_chunks(
            tenant_id=tenant_id,
            doc_id=chunks[0].doc_id,
            original_name=chunks[0].document_name,
            safe_name=chunks[0].document_name,
            page_count=max(c.page for c in chunks),
            chunks=chunks,
            vectors=vectors,
        )

    async def search(self, tenant_id: str, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not postgres_ready():
            return []
        query = _l2_normalize(query_vector)[0].tolist()
        distance = ChunkRow.embedding.cosine_distance(query)
        async with session_scope() as db:
            rows = (
                await db.execute(
                    select(ChunkRow, distance.label("dist"))
                    .where(ChunkRow.tenant_id == tenant_id)
                    .where(ChunkRow.embedding.is_not(None))
                    .order_by(distance)
                    .limit(top_k)
                )
            ).all()
        results: list[ScoredChunk] = []
        for row, dist in rows:
            score = 1.0 - float(dist)
            results.append(
                ScoredChunk(
                    chunk=ChunkRecord(
                        chunk_id=row.id,
                        tenant_id=row.tenant_id,
                        doc_id=row.document_id,
                        document_name=row.document_name,
                        page=row.page,
                        text=row.content,
                        token_estimate=row.token_estimate,
                    ),
                    score=score,
                )
            )
        return results

    async def delete_tenant(self, tenant_id: str) -> None:
        await db_clear_tenant_knowledge(tenant_id)

    async def exists(self, tenant_id: str) -> bool:
        if not postgres_ready():
            return False
        async with session_scope() as db:
            found = await db.scalar(
                select(ChunkRow.id)
                .where(ChunkRow.tenant_id == tenant_id)
                .where(ChunkRow.embedding.is_not(None))
                .limit(1)
            )
        return found is not None

    async def backfill_missing_embeddings(self, tenant_id: str) -> bool:
        """Embed existing chunk text that was saved before pgvector."""
        if not postgres_ready():
            return False
        async with session_scope() as db:
            rows = (
                await db.execute(
                    select(ChunkRow.id, ChunkRow.content)
                    .where(ChunkRow.tenant_id == tenant_id)
                    .where(ChunkRow.embedding.is_(None))
                )
            ).all()
        if not rows:
            return await self.exists(tenant_id)

        from app.services.rag.embeddings import get_embedding_provider

        vectors = _l2_normalize(
            await get_embedding_provider().embed_documents([content for _, content in rows])
        )
        async with session_scope() as db:
            for (chunk_id, _content), vec in zip(rows, vectors, strict=True):
                row = await db.get(ChunkRow, chunk_id)
                if row is not None:
                    row.embedding = [float(x) for x in vec]
        logger.info("Backfilled %s pgvector embeddings for tenant %s", len(rows), tenant_id)
        return True

    async def preload(self, tenant_id: str) -> None:
        if not await self.exists(tenant_id):
            await self.backfill_missing_embeddings(tenant_id)
