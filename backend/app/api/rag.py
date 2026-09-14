# Keep runtime annotations: SlowAPI wraps these endpoints, and postponed
# request-model annotations are otherwise resolved in the wrapper's module.

import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import verify_tenant_token
from app.models.schemas import ChatRequest, ChatResponse, RetrieveRequest, RetrieveResponse, TtsRequest
from app.services.llm.claude import claude_service
from app.services.rag.service import rag_service

router = APIRouter(prefix="/api/v1", tags=["rag"])


def require_tenant_auth(tenant_id: str, x_tenant_token: str | None) -> None:
    if not x_tenant_token or not verify_tenant_token(x_tenant_token, tenant_id):
        raise HTTPException(status_code=401, detail="Token tenant invalide ou expiré.")


@router.post("/rag/retrieve", response_model=RetrieveResponse)
@limiter.limit(get_settings().rate_limit_chat)
async def retrieve(
    request: Request,
    body: RetrieveRequest,
    x_tenant_token: str | None = Header(default=None),
) -> RetrieveResponse:
    require_tenant_auth(body.tenant_id, x_tenant_token)
    try:
        chunks, latency_ms = await rag_service.retrieve(body.tenant_id, body.query, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrieveResponse(chunks=chunks, latency_ms=latency_ms)


@router.post("/chat")
@limiter.limit(get_settings().rate_limit_chat)
async def chat(
    request: Request,
    body: ChatRequest,
    x_tenant_token: str | None = Header(default=None),
):
    require_tenant_auth(body.tenant_id, x_tenant_token)
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message vide.")

    if not body.stream:
        try:
            return await claude_service.answer(body.tenant_id, body.message.strip())
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_gen():
        try:
            async for item in claude_service.stream_answer(body.tenant_id, body.message.strip()):
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/tts")
@limiter.limit(get_settings().rate_limit_voice)
async def synthesize_speech(
    request: Request,
    body: TtsRequest,
    x_tenant_token: str | None = Header(default=None),
):
    """French speech synthesis using the configured TTS provider."""
    require_tenant_auth(body.tenant_id, x_tenant_token)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texte vide.")
    if len(text) > 2500:
        raise HTTPException(status_code=400, detail="Texte trop long pour la synthèse.")

    from app.services.voice.tts import synthesize_with_fallback

    try:
        audio, provider_name = await synthesize_with_fallback(text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Synthèse vocale échouée: {exc}") from exc
    if not audio:
        raise HTTPException(status_code=502, detail="Aucun audio généré.")

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Provider": provider_name,
        },
    )
