from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.core.config import get_settings
from app.models.schemas import ChunkRecord, PageText


@dataclass
class ChunkDraft:
    page: int
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _window_chunks(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def chunk_pages(
    pages: list[PageText],
    *,
    tenant_id: str,
    doc_id: str,
    document_name: str,
) -> list[ChunkRecord]:
    settings = get_settings()
    size = settings.chunk_size_chars
    overlap = settings.chunk_overlap_chars
    drafts: list[ChunkDraft] = []

    for page in pages:
        paragraphs = _split_paragraphs(page.text) or [page.text]
        buffer = ""
        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(candidate) <= size:
                buffer = candidate
                continue
            if buffer:
                drafts.append(ChunkDraft(page=page.page, text=buffer))
            for piece in _window_chunks(para, size, overlap):
                if len(piece) <= size and not buffer:
                    buffer = piece
                else:
                    drafts.append(ChunkDraft(page=page.page, text=piece))
                    buffer = ""
        if buffer:
            drafts.append(ChunkDraft(page=page.page, text=buffer))

    chunks: list[ChunkRecord] = []
    for draft in drafts:
        chunks.append(
            ChunkRecord(
                chunk_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                doc_id=doc_id,
                document_name=document_name,
                page=draft.page,
                text=draft.text,
                token_estimate=max(1, len(draft.text) // 4),
            )
        )
    return chunks
