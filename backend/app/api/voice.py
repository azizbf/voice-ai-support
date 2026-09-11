from __future__ import annotations

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.core.config import get_settings
from app.core.security import verify_tenant_token
from app.services.session.service import session_service
from app.services.voice.session import voice_websocket_loop

router = APIRouter(tags=["voice"])


@router.websocket("/api/v1/voice/{tenant_id}")
async def voice_ws(
    websocket: WebSocket,
    tenant_id: str,
    token: str = Query(default=""),
) -> None:
    await websocket.accept()
    # Basic abuse protection: reject unauthenticated / empty tenants early
    if not token or not verify_tenant_token(token, tenant_id):
        await websocket.send_json({"type": "error", "message": "Authentification invalide."})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    meta = await session_service.load_meta(tenant_id)
    if meta is None or meta.status != "ready":
        await websocket.send_json(
            {"type": "error", "message": "La base de connaissances n'est pas prête."}
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        await voice_websocket_loop(websocket, tenant_id)
    except WebSocketDisconnect:
        return
