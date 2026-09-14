from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.logging_metrics import timed
from app.core.security import safe_filename
from app.models.schemas import (
    DocumentInfo,
    PageText,
    PipelineStatus,
    RetrievedChunk,
    SourceCitation,
)
from app.services.pdf.extractor import PdfValidationError, extract_pdf, validate_pdf_bytes
from app.services.rag.chunking import chunk_pages
from app.services.rag.embeddings import get_embedding_provider
from app.services.rag.query import expand_retrieval_query, lexical_needles
from app.services.session.service import session_service
from app.services.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Tu es un représentant du service clientèle professionnel, francophone (Tunisie).
Tu réponds clairement, naturellement, poliment et de façon concise.
Tu utilises UNIQUEMENT la base de connaissances fournie comme source de vérité.
Quand un tarif, une procédure ou une politique apparaît dans le contexte, tu le donnes
directement et précisément (ex: prix en DT, délais, étapes) — ne dis PAS que tu n'as pas
l'information si elle est présente.
Questions offres / forfait / Fibre 50, 100 ou 300: si le contexte contient « Forfaits Fibre »
ou un prix en DT, liste les forfaits présents (prix mensuel et frais de mise en service).
N'escalade pas.
Pour les procédures guidées (réinitialisation routeur, diagnostic panne, etc.):
donne UNE seule étape à la fois, demande au client de confirmer quand c'est fait,
puis passe à l'étape suivante seulement après sa confirmation. Ne récite jamais toute
la procédure d'un coup.
Tu n'inventes JAMAIS de politiques, prix ou procédures absents du contexte.
Escalade vers un conseiller humain UNIQUEMENT si l'information demandée est vraiment
absente du contexte fourni.
Réponds toujours en français."""


class RagService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._demo_jobs: dict[str, asyncio.Task] = {}
        self.last_timings: dict[str, float] = {}

    @property
    def store(self):
        return get_vector_store()

    async def ingest_pdf(self, tenant_id: str, filename: str, data: bytes) -> DocumentInfo:
        validate_pdf_bytes(data, filename)
        doc_id = str(uuid.uuid4())
        safe_name = safe_filename(filename)

        await session_service.set_status(
            tenant_id,
            "processing",
            detail="Extraction du document",
            pipeline=PipelineStatus(extracting=True, creating_knowledge_base=False, ready=False),
        )

        extracted = extract_pdf(data)

        tenant_dir = session_service.tenant_dir(tenant_id)
        upload_path = tenant_dir / "uploads" / f"{doc_id}_{safe_name}"
        upload_path.write_bytes(data)
        if self.settings.persist_documents:
            dest = tenant_dir / "documents" / f"{doc_id}_{safe_name}"
            dest.write_bytes(data)

        await session_service.set_status(
            tenant_id,
            "processing",
            detail="Création de la base de connaissances",
            pipeline=PipelineStatus(extracting=False, creating_knowledge_base=True, ready=False),
        )

        info = await self._index_pages(
            tenant_id=tenant_id,
            doc_id=doc_id,
            filename=filename,
            safe_name=safe_name,
            pages=extracted.pages,
            page_count=extracted.page_count,
        )

        if not self.settings.persist_documents and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        return info

    async def ingest_demo_knowledge(self, tenant_id: str) -> DocumentInfo:
        """Index the built-in Tunisian ISP FAQ (no PDF required)."""
        from app.services.rag.demo_knowledge import DEMO_DOCUMENT_NAME, demo_page_texts

        # Replace any previous index so the demo is deterministic
        await self.store.delete_tenant(tenant_id)
        tenant_dir = session_service.tenant_dir(tenant_id)
        for sub in ("documents", "index", "uploads"):
            target = tenant_dir / sub
            if target.exists():
                import shutil

                shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)

        doc_id = str(uuid.uuid4())
        pages = demo_page_texts()
        await session_service.set_status(
            tenant_id,
            "processing",
            detail="Création de la base de connaissances démo",
            pipeline=PipelineStatus(extracting=False, creating_knowledge_base=True, ready=False),
        )
        return await self._index_pages(
            tenant_id=tenant_id,
            doc_id=doc_id,
            filename=DEMO_DOCUMENT_NAME,
            safe_name=safe_filename(DEMO_DOCUMENT_NAME),
            pages=pages,
            page_count=len(pages),
        )

    async def _index_pages(
        self,
        *,
        tenant_id: str,
        doc_id: str,
        filename: str,
        safe_name: str,
        pages: list[PageText],
        page_count: int,
    ) -> DocumentInfo:
        chunks = chunk_pages(
            pages,
            tenant_id=tenant_id,
            doc_id=doc_id,
            document_name=safe_name,
        )
        if not chunks:
            raise PdfValidationError("Aucun contenu indexable trouvé.")

        embedder = get_embedding_provider()
        vectors = await embedder.embed_documents([c.text for c in chunks])
        await self.store.upsert(tenant_id, chunks, vectors)

        info = DocumentInfo(
            doc_id=doc_id,
            original_name=filename,
            page_count=page_count,
            chunk_count=len(chunks),
        )
        await session_service.set_status(
            tenant_id,
            "ready",
            detail="Prêt",
            pipeline=PipelineStatus(extracting=False, creating_knowledge_base=False, ready=True),
            document=info,
        )
        from app.db.repository import db_log_usage, db_save_document_with_chunks

        if not getattr(self.store, "persists_chunks", False):
            await db_save_document_with_chunks(
                tenant_id=tenant_id,
                doc_id=doc_id,
                original_name=filename,
                safe_name=safe_name,
                page_count=page_count,
                chunks=chunks,
            )
        await db_log_usage(
            tenant_id,
            "upload",
            {"doc_id": doc_id, "pages": page_count, "chunks": len(chunks), "demo": filename.endswith("Demo.pdf")},
        )
        logger.info("Ingested %s chunks for tenant %s (%s)", len(chunks), tenant_id, filename)
        return info

    def start_demo_ingest(self, tenant_id: str) -> asyncio.Task:
        """Index the demo FAQ, or no-op if that tenant is already ingesting."""
        existing = self._demo_jobs.get(tenant_id)
        if existing is not None and not existing.done():
            return existing

        async def _run() -> None:
            try:
                await self.ingest_demo_knowledge(tenant_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Demo ingest failed for %s", tenant_id)
                await session_service.set_status(
                    tenant_id,
                    "error",
                    detail=f"Erreur démo: {exc}",
                    pipeline=PipelineStatus(),
                )
            finally:
                current = self._demo_jobs.get(tenant_id)
                if current is asyncio.current_task():
                    self._demo_jobs.pop(tenant_id, None)

        task = asyncio.create_task(_run())
        self._demo_jobs[tenant_id] = task
        return task

    async def retrieve(self, tenant_id: str, query: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], float]:
        meta = await session_service.load_meta(tenant_id)
        if meta is None or meta.status != "ready":
            raise ValueError("La base de connaissances n'est pas prête pour ce tenant.")
        backfill_ms = 0.0
        if not await self.store.exists(tenant_id):
            backfill = getattr(self.store, "backfill_missing_embeddings", None)
            if callable(backfill):
                started = time.perf_counter()
                await backfill(tenant_id)
                backfill_ms = (time.perf_counter() - started) * 1000
        if not await self.store.exists(tenant_id):
            raise ValueError(
                "La base de connaissances n'est pas prête pour ce tenant. "
                "Démarrez une nouvelle session pour réindexer."
            )

        user_query = query
        query = expand_retrieval_query(query)
        k = max(top_k or self.settings.rag_top_k, 4)
        load_started = time.perf_counter()
        embedder = get_embedding_provider()
        load_ms = (time.perf_counter() - load_started) * 1000
        with timed() as t:
            embed_started = time.perf_counter()
            qvec = await embedder.embed_query(query)
            embed_ms = (time.perf_counter() - embed_started) * 1000
            search_started = time.perf_counter()
            scored = await self.store.search(tenant_id, qvec[0], k)
            search_ms = (time.perf_counter() - search_started) * 1000

        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for item in scored:
            if item.score < self.settings.rag_min_score:
                continue
            seen.add(item.chunk.chunk_id)
            chunks.append(
                RetrievedChunk(
                    chunk_id=item.chunk.chunk_id,
                    document_name=item.chunk.document_name,
                    page=item.chunk.page,
                    text=item.chunk.text,
                    score=item.score,
                )
            )

        needles = lexical_needles(user_query)
        if needles:
            keyed = await self.store.keyword_chunks(tenant_id, needles)
            if not keyed:
                from app.db.repository import db_keyword_chunks

                keyed = await db_keyword_chunks(tenant_id, needles)
            for row in keyed:
                chunk_id = getattr(row, "chunk_id", None) or getattr(row, "id", "")
                if not chunk_id or chunk_id in seen:
                    continue
                text = getattr(row, "text", None) or getattr(row, "content", "") or ""
                seen.add(chunk_id)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        document_name=row.document_name,
                        page=row.page,
                        text=text,
                        score=0.62 if "forfaits fibre" in text.lower() else 0.55,
                    )
                )
        chunks.sort(key=lambda c: c.score, reverse=True)
        self.last_timings = {
            "rag_embed_load_ms": round(load_ms),
            "rag_embed_ms": round(embed_ms),
            "rag_search_ms": round(search_ms),
            "rag_backfill_ms": round(backfill_ms),
        }
        return chunks, t["elapsed_ms"]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "Aucune information pertinente trouvée dans la base de connaissances."
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i}] Document: {c.document_name} | Page: {c.page} | Score: {c.score:.3f}\n{c.text}"
            )
        return "\n\n".join(parts)

    def to_sources(self, chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        seen: set[tuple[str, int]] = set()
        sources: list[SourceCitation] = []
        for c in chunks:
            key = (c.document_name, c.page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceCitation(
                    document_name=c.document_name,
                    page=c.page,
                    score=c.score,
                    chunk_id=c.chunk_id,
                )
            )
        return sources


rag_service = RagService()
