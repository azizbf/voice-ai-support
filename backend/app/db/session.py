from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_pgvector_ready = False


def postgres_ready() -> bool:
    return get_settings().postgres_enabled and _session_factory is not None


def pgvector_ready() -> bool:
    return _pgvector_ready


async def init_db() -> bool:
    """Initialize engine and create tables. Returns True if Postgres is active."""
    global _engine, _session_factory, _pgvector_ready
    settings = get_settings()
    if not settings.database_url.strip():
        logger.warning("DATABASE_URL not set — running without PostgreSQL (file/FAISS only)")
        return False

    url = settings.database_url.strip()
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)

    dim = int(settings.embedding_dimensions or 384)
    async with _engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            _pgvector_ready = True
        except Exception:
            _pgvector_ready = False
            logger.exception("pgvector extension is not installed — RAG will fall back to FAISS")

        await conn.run_sync(Base.metadata.create_all)

        if _pgvector_ready:
            try:
                await conn.execute(text(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector({dim})"))
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS chunks_embedding_cosine_idx "
                        "ON chunks USING hnsw (embedding vector_cosine_ops)"
                    )
                )
            except Exception:
                logger.exception("pgvector column/index setup failed")

    logger.info("PostgreSQL connected and schema ready (pgvector=%s)", _pgvector_ready)
    return True


async def close_db() -> None:
    global _engine, _session_factory, _pgvector_ready
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
    _pgvector_ready = False


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("PostgreSQL is not configured")
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
