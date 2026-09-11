from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    name: str

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbeddingProvider:
    name = "local"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        return await self.embed_documents([text])


class VoyageEmbeddingProvider:
    name = "voyage"

    def __init__(self, api_key: str, model: str) -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        # voyage SDK is sync; run in batches
        batch_size = 128
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.embed(batch, model=self.model, input_type="document")
            all_vecs.extend(result.embeddings)
        return np.asarray(all_vecs, dtype=np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        result = self.client.embed([text], model=self.model, input_type="query")
        return np.asarray(result.embeddings, dtype=np.float32)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "voyage" or (provider == "auto" and settings.voyage_api_key):
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY required when EMBEDDING_PROVIDER=voyage")
        return VoyageEmbeddingProvider(settings.voyage_api_key, settings.voyage_model)
    return LocalEmbeddingProvider(settings.local_embedding_model)
