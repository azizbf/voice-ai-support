from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, or_, select

from app.db.models import (
    ChunkRow,
    ConversationRow,
    DocumentRow,
    MessageRow,
    TenantRow,
    UsageEventRow,
)
from app.db.session import postgres_ready, session_scope

logger = logging.getLogger(__name__)


async def db_upsert_tenant(
    tenant_id: str,
    *,
    status: str,
    expires_at: datetime | None = None,
    status_detail: str = "",
    name: str = "Demo tenant",
) -> None:
    if not postgres_ready():
        return
    try:
        async with session_scope() as db:
            row = await db.get(TenantRow, tenant_id)
            if row is None:
                row = TenantRow(
                    id=tenant_id,
                    name=name,
                    status=status,
                    status_detail=status_detail,
                    expires_at=expires_at,
                )
                db.add(row)
            else:
                row.status = status
                row.status_detail = status_detail
                if expires_at is not None:
                    row.expires_at = expires_at
    except Exception:
        logger.exception("db_upsert_tenant failed")


async def db_save_document_with_chunks(
    *,
    tenant_id: str,
    doc_id: str,
    original_name: str,
    safe_name: str,
    page_count: int,
    chunks: list[Any],
    vectors: Any | None = None,
) -> None:
    if not postgres_ready():
        return
    try:
        async with session_scope() as db:
            # Ensure tenant exists
            if await db.get(TenantRow, tenant_id) is None:
                db.add(TenantRow(id=tenant_id, status="ready", name="Demo tenant"))

            # Replace previous docs/chunks for this demo tenant (single-doc MVP)
            await db.execute(delete(ChunkRow).where(ChunkRow.tenant_id == tenant_id))
            await db.execute(delete(DocumentRow).where(DocumentRow.tenant_id == tenant_id))

            doc = DocumentRow(
                id=doc_id,
                tenant_id=tenant_id,
                original_name=original_name,
                safe_name=safe_name,
                page_count=page_count,
                chunk_count=len(chunks),
                status="ready",
            )
            db.add(doc)
            for i, chunk in enumerate(chunks):
                embedding = None
                if vectors is not None:
                    embedding = [float(x) for x in vectors[i]]
                db.add(
                    ChunkRow(
                        id=chunk.chunk_id,
                        tenant_id=tenant_id,
                        document_id=doc_id,
                        document_name=chunk.document_name,
                        page=chunk.page,
                        content=chunk.text,
                        token_estimate=chunk.token_estimate,
                        faiss_row=i,
                        embedding=embedding,
                    )
                )
            tenant = await db.get(TenantRow, tenant_id)
            if tenant:
                tenant.status = "ready"
                tenant.status_detail = "Prêt"
    except Exception:
        logger.exception("db_save_document_with_chunks failed")
        raise


async def db_clear_tenant_knowledge(tenant_id: str) -> None:
    if not postgres_ready():
        return
    try:
        async with session_scope() as db:
            await db.execute(delete(ChunkRow).where(ChunkRow.tenant_id == tenant_id))
            await db.execute(delete(DocumentRow).where(DocumentRow.tenant_id == tenant_id))
            tenant = await db.get(TenantRow, tenant_id)
            if tenant:
                tenant.status = "empty"
                tenant.status_detail = ""
    except Exception:
        logger.exception("db_clear_tenant_knowledge failed")


async def db_log_usage(tenant_id: str, event_type: str, meta: dict[str, Any] | None = None) -> None:
    if not postgres_ready():
        return
    try:
        async with session_scope() as db:
            if await db.get(TenantRow, tenant_id) is None:
                db.add(TenantRow(id=tenant_id, status="ready"))
            db.add(
                UsageEventRow(
                    tenant_id=tenant_id,
                    event_type=event_type,
                    meta_json=json.dumps(meta or {}, ensure_ascii=False),
                )
            )
    except Exception:
        logger.exception("db_log_usage failed")


async def db_save_turn(
    *,
    tenant_id: str,
    channel: str,
    user_text: str,
    assistant_text: str,
    sources: list[dict[str, Any]],
    latency_ms: float | None = None,
) -> None:
    if not postgres_ready():
        return
    try:
        async with session_scope() as db:
            if await db.get(TenantRow, tenant_id) is None:
                db.add(TenantRow(id=tenant_id, status="ready"))
            conv = ConversationRow(tenant_id=tenant_id, channel=channel)
            db.add(conv)
            await db.flush()
            db.add(
                MessageRow(
                    conversation_id=conv.id,
                    tenant_id=tenant_id,
                    role="user",
                    content=user_text,
                )
            )
            db.add(
                MessageRow(
                    conversation_id=conv.id,
                    tenant_id=tenant_id,
                    role="assistant",
                    content=assistant_text,
                    sources_json=json.dumps(sources, ensure_ascii=False),
                    latency_ms=latency_ms,
                )
            )
            db.add(
                UsageEventRow(
                    tenant_id=tenant_id,
                    event_type=channel,
                    meta_json=json.dumps({"latency_ms": latency_ms}, ensure_ascii=False),
                )
            )
    except Exception:
        logger.exception("db_save_turn failed")


async def db_overview() -> dict[str, Any]:
    if not postgres_ready():
        return {"enabled": False}
    async with session_scope() as db:
        tenants = (await db.execute(select(func.count()).select_from(TenantRow))).scalar_one()
        docs = (await db.execute(select(func.count()).select_from(DocumentRow))).scalar_one()
        chunks = (await db.execute(select(func.count()).select_from(ChunkRow))).scalar_one()
        messages = (await db.execute(select(func.count()).select_from(MessageRow))).scalar_one()
        conversations = (
            await db.execute(select(func.count()).select_from(ConversationRow))
        ).scalar_one()
        ready = (
            await db.execute(select(func.count()).select_from(TenantRow).where(TenantRow.status == "ready"))
        ).scalar_one()
        return {
            "enabled": True,
            "tenants": tenants,
            "tenants_ready": ready,
            "documents": docs,
            "chunks": chunks,
            "conversations": conversations,
            "messages": messages,
        }


async def db_list_tenants(limit: int = 50) -> list[dict[str, Any]]:
    if not postgres_ready():
        return []
    async with session_scope() as db:
        rows = (
            await db.execute(select(TenantRow).order_by(TenantRow.created_at.desc()).limit(limit))
        ).scalars().all()
        out: list[dict[str, Any]] = []
        for t in rows:
            doc_count = (
                await db.execute(
                    select(func.count()).select_from(DocumentRow).where(DocumentRow.tenant_id == t.id)
                )
            ).scalar_one()
            msg_count = (
                await db.execute(
                    select(func.count()).select_from(MessageRow).where(MessageRow.tenant_id == t.id)
                )
            ).scalar_one()
            out.append(
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status,
                    "status_detail": t.status_detail,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "expires_at": t.expires_at.isoformat() if t.expires_at else None,
                    "document_count": doc_count,
                    "message_count": msg_count,
                }
            )
        return out


async def db_tenant_detail(tenant_id: str) -> dict[str, Any] | None:
    if not postgres_ready():
        return None
    async with session_scope() as db:
        t = await db.get(TenantRow, tenant_id)
        if t is None:
            return None
        docs = (
            await db.execute(
                select(DocumentRow).where(DocumentRow.tenant_id == tenant_id).order_by(DocumentRow.created_at.desc())
            )
        ).scalars().all()
        msgs = (
            await db.execute(
                select(MessageRow)
                .where(MessageRow.tenant_id == tenant_id)
                .order_by(MessageRow.created_at.desc())
                .limit(40)
            )
        ).scalars().all()
        return {
            "id": t.id,
            "name": t.name,
            "status": t.status,
            "status_detail": t.status_detail,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "documents": [
                {
                    "id": d.id,
                    "original_name": d.original_name,
                    "page_count": d.page_count,
                    "chunk_count": d.chunk_count,
                    "status": d.status,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in docs
            ],
            "recent_messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content[:500],
                    "sources": json.loads(m.sources_json or "[]"),
                    "latency_ms": m.latency_ms,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in msgs
            ],
        }


async def db_keyword_chunks(tenant_id: str, needles: list[str], limit: int = 6) -> list[ChunkRow]:
    if not postgres_ready() or not needles:
        return []
    safe = [n.strip() for n in needles if n and len(n.strip()) >= 2 and "%" not in n and "_" not in n]
    if not safe:
        return []
    async with session_scope() as db:
        conds = [ChunkRow.content.ilike(f"%{n}%") for n in safe]
        rows = (
            await db.execute(
                select(ChunkRow).where(ChunkRow.tenant_id == tenant_id).where(or_(*conds)).limit(limit)
            )
        ).scalars().all()
        return list(rows)
