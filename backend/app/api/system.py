from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.logging_metrics import latency_store
from app.db.session import postgres_ready
from app.models.schemas import HealthResponse
from app.services.vectorstore.factory import vector_store_name

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    embedding = (
        "voyage"
        if (settings.voyage_api_key and settings.embedding_provider != "local")
        else "local"
    )
    from app.services.voice.stt import describe_server_stt, stt_ready
    from app.services.rag.embeddings import embeddings_ready

    info = describe_server_stt()
    tts = "edge-tts"
    emb_ok = embeddings_ready()
    stt_ok = stt_ready()
    voice_ok = emb_ok and stt_ok
    return HealthResponse(
        status="ok" if voice_ok else "starting",
        embedding_provider=embedding,
        stt_provider=info["stt"],
        stt_device=info.get("stt_device"),
        tts_provider=tts,
        embeddings_ready=emb_ok,
        stt_ready=stt_ok,
        voice_ready=voice_ok,
        stt_impl=info.get("stt_impl"),
        whisper_keepalive=bool(settings.whisper_keepalive),
    )


@router.get("/api/v1/system/storage")
async def storage_status():
    settings = get_settings()
    return {
        "postgres_configured": settings.postgres_enabled,
        "postgres_connected": postgres_ready(),
        "vector_store": vector_store_name(),
        "data_dir": settings.data_dir,
    }


@router.get("/api/v1/metrics/recent")
async def recent_metrics():
    settings = get_settings()
    if settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Not found")
    return {"samples": latency_store.recent(50), "percentiles": latency_store.percentiles()}
