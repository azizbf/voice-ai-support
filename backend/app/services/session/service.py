from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiofiles

from app.core.config import get_settings
from app.models.schemas import DocumentInfo, PipelineStatus, TenantMeta, TenantStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def tenant_dir(self, tenant_id: str) -> Path:
        path = self.settings.data_path / tenant_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "documents").mkdir(exist_ok=True)
        (path / "index").mkdir(exist_ok=True)
        (path / "uploads").mkdir(exist_ok=True)
        return path

    def meta_path(self, tenant_id: str) -> Path:
        return self.tenant_dir(tenant_id) / "meta.json"

    async def create_tenant(self) -> TenantMeta:
        tenant_id = str(uuid.uuid4())
        now = utcnow()
        meta = TenantMeta(
            tenant_id=tenant_id,
            created_at=now,
            expires_at=now + timedelta(hours=self.settings.tenant_ttl_hours),
            status="empty",
            pipeline=PipelineStatus(),
        )
        await self.save_meta(meta)
        from app.db.repository import db_upsert_tenant

        await db_upsert_tenant(
            tenant_id,
            status="empty",
            expires_at=meta.expires_at,
            name="Demo tenant",
        )
        return meta

    async def save_meta(self, meta: TenantMeta) -> None:
        path = self.meta_path(meta.tenant_id)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(meta.model_dump_json(indent=2))

    async def load_meta(self, tenant_id: str) -> TenantMeta | None:
        path = self.meta_path(tenant_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            raw = await f.read()
        return TenantMeta.model_validate_json(raw)

    async def set_status(
        self,
        tenant_id: str,
        status: TenantStatus,
        *,
        detail: str = "",
        pipeline: PipelineStatus | None = None,
        document: DocumentInfo | None = None,
    ) -> TenantMeta:
        meta = await self.load_meta(tenant_id)
        if meta is None:
            raise KeyError(tenant_id)
        meta.status = status
        meta.status_detail = detail
        if pipeline is not None:
            meta.pipeline = pipeline
        if document is not None:
            meta.document = document
        await self.save_meta(meta)
        from app.db.repository import db_upsert_tenant

        await db_upsert_tenant(tenant_id, status=status, status_detail=detail, expires_at=meta.expires_at)
        return meta

    async def clear_knowledge(self, tenant_id: str) -> None:
        path = self.tenant_dir(tenant_id)
        for sub in ("documents", "index", "uploads"):
            target = path / sub
            if target.exists():
                shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
        meta = await self.load_meta(tenant_id)
        if meta:
            meta.status = "empty"
            meta.status_detail = ""
            meta.document = None
            meta.pipeline = PipelineStatus()
            await self.save_meta(meta)
        from app.db.repository import db_clear_tenant_knowledge

        await db_clear_tenant_knowledge(tenant_id)

    async def delete_tenant(self, tenant_id: str) -> None:
        path = self.settings.data_path / tenant_id
        if path.exists():
            shutil.rmtree(path)

    async def cleanup_expired(self) -> int:
        removed = 0
        root = self.settings.data_path
        if not root.exists():
            return 0
        now = utcnow()
        for child in root.iterdir():
            if not child.is_dir():
                continue
            meta_file = child / "meta.json"
            if not meta_file.exists():
                continue
            try:
                raw = meta_file.read_text(encoding="utf-8")
                meta = TenantMeta.model_validate_json(raw)
            except Exception:
                continue
            if meta.expires_at.replace(tzinfo=timezone.utc) < now and not self.settings.persist_documents:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed


session_service = SessionService()
