from __future__ import annotations

import asyncio
from datetime import timezone
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import issue_tenant_token, verify_tenant_token
from app.models.schemas import (
    DocumentUploadResponse,
    OkResponse,
    PipelineStatus,
    TenantCreateResponse,
    TenantStatusResponse,
)
from app.services.pdf.extractor import PdfValidationError, extract_pdf, validate_pdf_bytes
from app.services.rag.service import rag_service
from app.services.session.service import session_service
from app.services.vectorstore.factory import get_vector_store

router = APIRouter(prefix="/api/v1/tenants", tags=["tenants"])


def require_tenant_auth(tenant_id: str, x_tenant_token: str | None) -> None:
    if not x_tenant_token or not verify_tenant_token(x_tenant_token, tenant_id):
        raise HTTPException(status_code=401, detail="Token tenant invalide ou expiré.")


@router.post("", response_model=TenantCreateResponse)
@limiter.limit(get_settings().rate_limit_upload)
async def create_tenant(request: Request) -> TenantCreateResponse:
    meta = await session_service.create_tenant()
    exp = meta.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    token = issue_tenant_token(meta.tenant_id, exp.timestamp())

    # Demo tenants always get the built-in FAQ — no PDF upload required.
    await session_service.set_status(
        meta.tenant_id,
        "processing",
        detail="Chargement de la base démo",
        pipeline=PipelineStatus(extracting=False, creating_knowledge_base=True, ready=False),
    )

    async def _seed() -> None:
        try:
            await rag_service.ingest_demo_knowledge(meta.tenant_id)
        except Exception as exc:  # noqa: BLE001
            await session_service.set_status(
                meta.tenant_id,
                "error",
                detail=f"Erreur démo: {exc}",
                pipeline=PipelineStatus(),
            )

    asyncio.create_task(_seed())
    return TenantCreateResponse(tenant_id=meta.tenant_id, token=token, expires_at=meta.expires_at)


@router.get("/{tenant_id}/status", response_model=TenantStatusResponse)
async def get_tenant_status(
    tenant_id: str,
    x_tenant_token: str | None = Header(default=None),
) -> TenantStatusResponse:
    require_tenant_auth(tenant_id, x_tenant_token)
    meta = await session_service.load_meta(tenant_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Tenant introuvable.")
    return TenantStatusResponse(
        tenant_id=meta.tenant_id,
        status=meta.status,
        pipeline=meta.pipeline,
        document=meta.document,
        error=meta.status_detail if meta.status == "error" else None,
    )


@router.post("/{tenant_id}/documents", response_model=DocumentUploadResponse)
@limiter.limit(get_settings().rate_limit_upload)
async def upload_document(
    request: Request,
    tenant_id: str,
    file: Annotated[UploadFile, File(...)],
    x_tenant_token: str | None = Header(default=None),
) -> DocumentUploadResponse:
    require_tenant_auth(tenant_id, x_tenant_token)
    meta = await session_service.load_meta(tenant_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Tenant introuvable.")

    data = await file.read()
    filename = file.filename or "document.pdf"

    try:
        validate_pdf_bytes(data, filename)
        extracted = extract_pdf(data)
    except PdfValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await session_service.set_status(
        tenant_id,
        "processing",
        detail="Extraction du document",
        pipeline=PipelineStatus(extracting=True, creating_knowledge_base=False, ready=False),
    )

    async def _process() -> None:
        try:
            await rag_service.ingest_pdf(tenant_id, filename, data)
        except PdfValidationError as exc:
            await session_service.set_status(
                tenant_id,
                "error",
                detail=str(exc),
                pipeline=PipelineStatus(),
            )
        except Exception as exc:  # noqa: BLE001
            await session_service.set_status(
                tenant_id,
                "error",
                detail=f"Erreur de traitement: {exc}",
                pipeline=PipelineStatus(),
            )

    asyncio.create_task(_process())

    return DocumentUploadResponse(
        doc_id="pending",
        original_name=filename,
        page_count=extracted.page_count,
        status="processing",
    )


@router.post("/{tenant_id}/demo-knowledge", response_model=DocumentUploadResponse)
@limiter.limit(get_settings().rate_limit_upload)
async def load_demo_knowledge(
    request: Request,
    tenant_id: str,
    x_tenant_token: str | None = Header(default=None),
) -> DocumentUploadResponse:
    """Index the built-in Tunisian ISP FAQ so the voice/chat demo works without a PDF."""
    require_tenant_auth(tenant_id, x_tenant_token)
    meta = await session_service.load_meta(tenant_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Tenant introuvable.")

    await session_service.set_status(
        tenant_id,
        "processing",
        detail="Chargement de la FAQ démo",
        pipeline=PipelineStatus(extracting=False, creating_knowledge_base=True, ready=False),
    )

    async def _process() -> None:
        try:
            await rag_service.ingest_demo_knowledge(tenant_id)
        except Exception as exc:  # noqa: BLE001
            await session_service.set_status(
                tenant_id,
                "error",
                detail=f"Erreur démo: {exc}",
                pipeline=PipelineStatus(),
            )

    asyncio.create_task(_process())
    from app.services.rag.demo_knowledge import DEMO_DOCUMENT_NAME, DEMO_PAGES

    return DocumentUploadResponse(
        doc_id="pending",
        original_name=DEMO_DOCUMENT_NAME,
        page_count=len(DEMO_PAGES),
        status="processing",
    )


@router.delete("/{tenant_id}/knowledge", response_model=OkResponse)
async def clear_knowledge(
    tenant_id: str,
    x_tenant_token: str | None = Header(default=None),
) -> OkResponse:
    require_tenant_auth(tenant_id, x_tenant_token)
    meta = await session_service.load_meta(tenant_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Tenant introuvable.")
    store = get_vector_store()
    await store.delete_tenant(tenant_id)
    await session_service.clear_knowledge(tenant_id)
    return OkResponse(ok=True)
