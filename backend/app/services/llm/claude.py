from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.core.logging_metrics import LatencySample, latency_store, timed
from app.models.schemas import ChatResponse, LatencyBreakdown, SourceCitation
from app.services.rag.service import SYSTEM_PROMPT, rag_service

logger = logging.getLogger(__name__)


class ClaudeService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: AsyncAnthropic | None = None

    def refresh_settings(self) -> None:
        get_settings.cache_clear()
        self.settings = get_settings()
        # Force client rebuild if API key changed
        self._client = None

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self.refresh_settings()
            if not self.settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY manquante. Ajoutez-la dans backend/.env puis redémarrez l'API."
                )
            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _user_message(self, question: str, context: str) -> str:
        return (
            f"Base de connaissances entreprise:\n{context}\n\n"
            f"Question du client:\n{question}\n\n"
            "Réponds en te basant uniquement sur la base de connaissances ci-dessus. "
            "Si un tarif ou une procédure y figure, communique-le clairement. "
            "N'escalade vers un humain que si l'information est absente."
        )

    async def answer(self, tenant_id: str, message: str) -> ChatResponse:
        chunks, retrieval_ms = await rag_service.retrieve(tenant_id, message)
        context = rag_service.format_context(chunks)
        sources = rag_service.to_sources(chunks)

        with timed() as t:
            response = await self.client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._user_message(message, context)}],
            )
        answer = "".join(block.text for block in response.content if block.type == "text")
        total = retrieval_ms + t["elapsed_ms"]
        latency_store.add(
            LatencySample(
                tenant_id=tenant_id,
                path="chat",
                retrieval_ms=retrieval_ms,
                llm_first_token_ms=t["elapsed_ms"],
                total_ms=total,
                retrieval_scores=[c.score for c in chunks],
                chunks_retrieved=len(chunks),
                source_pages=[{"document_name": s.document_name, "page": s.page} for s in sources],
            )
        )
        from app.db.repository import db_save_turn

        await db_save_turn(
            tenant_id=tenant_id,
            channel="chat",
            user_text=message,
            assistant_text=answer,
            sources=[s.model_dump() for s in sources],
            latency_ms=total,
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
            latency=LatencyBreakdown(
                retrieval_ms=retrieval_ms,
                llm_ms=t["elapsed_ms"],
                total_ms=total,
            ),
        )

    async def stream_answer(self, tenant_id: str, message: str) -> AsyncIterator[dict[str, Any]]:
        import time

        t0 = time.perf_counter()
        chunks, retrieval_ms = await rag_service.retrieve(tenant_id, message)
        context = rag_service.format_context(chunks)
        sources = rag_service.to_sources(chunks)
        yield {
            "event": "meta",
            "data": {
                "retrieval_ms": retrieval_ms,
                "sources": [s.model_dump() for s in sources],
            },
        }

        first_token_ms: float | None = None
        answer_parts: list[str] = []
        async with self.client.messages.stream(
            model=self.settings.anthropic_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._user_message(message, context)}],
        ) as stream:
            async for text in stream.text_stream:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                answer_parts.append(text)
                yield {"event": "token", "data": {"text": text}}

        total_ms = (time.perf_counter() - t0) * 1000
        latency_store.add(
            LatencySample(
                tenant_id=tenant_id,
                path="chat_stream",
                retrieval_ms=retrieval_ms,
                llm_first_token_ms=first_token_ms,
                total_ms=total_ms,
                retrieval_scores=[c.score for c in chunks],
                chunks_retrieved=len(chunks),
                source_pages=[{"document_name": s.document_name, "page": s.page} for s in sources],
            )
        )
        from app.db.repository import db_save_turn

        await db_save_turn(
            tenant_id=tenant_id,
            channel="chat",
            user_text=message,
            assistant_text="".join(answer_parts),
            sources=[s.model_dump() for s in sources],
            latency_ms=total_ms,
        )
        yield {"event": "sources", "data": {"sources": [s.model_dump() for s in sources]}}
        yield {
            "event": "done",
            "data": {
                "latency": {
                    "retrieval_ms": retrieval_ms,
                    "llm_first_token_ms": first_token_ms,
                    "total_ms": total_ms,
                }
            },
        }


claude_service = ClaudeService()
