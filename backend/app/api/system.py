from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.logging_metrics import latency_store
from app.db.session import postgres_ready
from app.models.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    embedding = (
        "voyage"
        if (settings.voyage_api_key and settings.embedding_provider != "local")
        else "local"
    )
    if settings.deepgram_api_key:
        stt = "deepgram"
    else:
        stt = "client-speech"
    tts = "edge-tts"
    return HealthResponse(
        status="ok",
        embedding_provider=embedding,
        stt_provider=stt,
        tts_provider=tts,
    )


@router.get("/api/v1/system/storage")
async def storage_status():
    settings = get_settings()
    return {
        "postgres_configured": settings.postgres_enabled,
        "postgres_connected": postgres_ready(),
        "vector_store": "faiss",
        "data_dir": settings.data_dir,
    }


@router.get("/api/v1/metrics/recent")
async def recent_metrics():
    settings = get_settings()
    if settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Not found")
    return {"samples": latency_store.recent(50), "percentiles": latency_store.percentiles()}
