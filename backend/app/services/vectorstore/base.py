from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.models.schemas import ChunkRecord


@dataclass
class ScoredChunk:
    chunk: ChunkRecord
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, tenant_id: str, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(self, tenant_id: str, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        raise NotImplementedError

    @abstractmethod
    async def delete_tenant(self, tenant_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, tenant_id: str) -> bool:
        raise NotImplementedError
