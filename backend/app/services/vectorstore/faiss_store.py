from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import asyncio

import faiss
import numpy as np

from app.core.config import get_settings
from app.models.schemas import ChunkRecord
from app.services.vectorstore.base import ScoredChunk, VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self) -> None:
        self.settings = get_settings()

    def _index_dir(self, tenant_id: str) -> Path:
        path = self.settings.data_path / tenant_id / "index"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _index_path(self, tenant_id: str) -> Path:
        return self._index_dir(tenant_id) / "faiss.index"

    def _meta_path(self, tenant_id: str) -> Path:
        return self._index_dir(tenant_id) / "chunks.jsonl"

    def exists(self, tenant_id: str) -> bool:
        return self._index_path(tenant_id).exists() and self._meta_path(tenant_id).exists()

    async def upsert(self, tenant_id: str, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            raise ValueError("No chunks to index")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Vector/chunk length mismatch")

        vectors = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        index_path = self._index_path(tenant_id)
        meta_path = self._meta_path(tenant_id)
        faiss.write_index(index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    def _load(self, tenant_id: str) -> tuple[faiss.Index, list[ChunkRecord]]:
        if not self.exists(tenant_id):
            raise FileNotFoundError(f"No index for tenant {tenant_id}")
        index_stat = self._index_path(tenant_id).stat()
        meta_stat = self._meta_path(tenant_id).stat()
        return self._load_version(tenant_id, (index_stat.st_mtime_ns, index_stat.st_size,
                                             meta_stat.st_mtime_ns, meta_stat.st_size))

    async def preload(self, tenant_id: str) -> None:
        await asyncio.to_thread(self._load, tenant_id)

    @lru_cache(maxsize=16)
    def _load_version(self, tenant_id: str, version: tuple[int, ...]) -> tuple[faiss.Index, list[ChunkRecord]]:
        index = faiss.read_index(str(self._index_path(tenant_id)))
        chunks: list[ChunkRecord] = []
        with self._meta_path(tenant_id).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(ChunkRecord.model_validate_json(line))
        return index, chunks

    async def search(self, tenant_id: str, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        index, chunks = self._load(tenant_id)
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k = min(top_k, len(chunks))
        scores, idxs = index.search(q, k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            chunk = chunks[int(idx)]
            if chunk.tenant_id != tenant_id:
                continue
            results.append(ScoredChunk(chunk=chunk, score=float(score)))
        return results

    async def delete_tenant(self, tenant_id: str) -> None:
        self._load_version.cache_clear()
        for path in (self._index_path(tenant_id), self._meta_path(tenant_id)):
            if path.exists():
                path.unlink()
