from __future__ import annotations

import asyncio
import logging
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_ready = False
_ready_lock = threading.Lock()


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
        self.model_name = model_name

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = await asyncio.to_thread(
            self.model.encode, texts, normalize_embeddings=False, show_progress_bar=False
        )
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
        batch_size = 128
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = await asyncio.to_thread(
                self.client.embed, batch, model=self.model, input_type="document"
            )
            all_vecs.extend(result.embeddings)
        return np.asarray(all_vecs, dtype=np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        result = await asyncio.to_thread(
            self.client.embed, [text], model=self.model, input_type="query"
        )
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


def embeddings_ready() -> bool:
    return _ready


def warm_embeddings() -> None:
    """Load the embedding model and run one forward so retrieve is not paying 8s+."""
    global _ready
    if _ready:
        return
    with _ready_lock:
        if _ready:
            return
        provider = get_embedding_provider()
        if isinstance(provider, LocalEmbeddingProvider):
            provider.model.encode(["bonjour"], normalize_embeddings=False, show_progress_bar=False)
        _ready = True
        logger.info("Embedding provider ready (%s)", provider.name)
