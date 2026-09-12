from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException

from app.core.config import get_settings
from app.db.repository import db_list_tenants, db_overview, db_tenant_detail
from app.db.session import postgres_ready
from app.services.vectorstore.factory import vector_store_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


def require_dashboard_key(x_dashboard_key: str | None) -> None:
    expected = get_settings().dashboard_api_key
    if not expected or x_dashboard_key != expected:
        raise HTTPException(status_code=401, detail="Clé dashboard invalide.")


@router.get("/status")
async def dashboard_status():
    settings = get_settings()
    return {
        "postgres_configured": settings.postgres_enabled,
        "postgres_connected": postgres_ready(),
        "vector_store": vector_store_name(),
        "note": "Embeddings live in PostgreSQL via pgvector when DATABASE_URL is set; otherwise FAISS files.",
    }


@router.get("/overview")
async def overview(x_dashboard_key: str | None = Header(default=None)):
    require_dashboard_key(x_dashboard_key)
    if not postgres_ready():
        return {
            "enabled": False,
            "message": "Configure DATABASE_URL in backend/.env and restart the API.",
        }
    try:
        return await db_overview()
    except Exception as exc:
        logger.exception("dashboard overview failed")
        # Return JSON (with CORS via middleware) instead of opaque 500
        return {
            "enabled": True,
            "error": str(exc),
            "tenants": 0,
            "tenants_ready": 0,
            "documents": 0,
            "chunks": 0,
            "conversations": 0,
            "messages": 0,
        }


@router.get("/tenants")
async def tenants(x_dashboard_key: str | None = Header(default=None)):
    require_dashboard_key(x_dashboard_key)
    if not postgres_ready():
        raise HTTPException(status_code=503, detail="PostgreSQL non connecté.")
    try:
        return {"tenants": await db_list_tenants()}
    except Exception as exc:
        logger.exception("dashboard tenants failed")
        raise HTTPException(status_code=500, detail=f"Dashboard tenants error: {exc}") from exc


@router.get("/tenants/{tenant_id}")
async def tenant_detail(tenant_id: str, x_dashboard_key: str | None = Header(default=None)):
    require_dashboard_key(x_dashboard_key)
    if not postgres_ready():
        raise HTTPException(status_code=503, detail="PostgreSQL non connecté.")
    try:
        detail = await db_tenant_detail(tenant_id)
    except Exception as exc:
        logger.exception("dashboard tenant detail failed")
        raise HTTPException(status_code=500, detail=f"Dashboard detail error: {exc}") from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="Tenant introuvable.")
    return detail
