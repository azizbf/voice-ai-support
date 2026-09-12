"""Smoke test: PDF extract → chunk → embed → vector store → retrieve (no Claude required)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.session import close_db, init_db
from app.services.pdf.extractor import extract_pdf
from app.services.rag.chunking import chunk_pages
from app.services.rag.embeddings import get_embedding_provider
from app.services.session.service import session_service
from app.services.vectorstore.factory import get_vector_store, reset_vector_store


async def main() -> None:
    await init_db()
    reset_vector_store()
    sample = ROOT / "samples" / "faq_internet.pdf"
    if not sample.exists():
        from scripts.make_sample_pdf import main as make_pdf

        make_pdf()

    data = sample.read_bytes()
    extracted = extract_pdf(data)
    print(f"pages={extracted.page_count} text_pages={len(extracted.pages)}")

    meta = await session_service.create_tenant()
    tenant_id = meta.tenant_id
    chunks = chunk_pages(
        extracted.pages,
        tenant_id=tenant_id,
        doc_id="doc-1",
        document_name="faq_internet.pdf",
    )
    print(f"chunks={len(chunks)}")

    embedder = get_embedding_provider()
    vectors = await embedder.embed_documents([c.text for c in chunks])
    store = get_vector_store()
    await store.upsert(tenant_id, chunks, vectors)

    q = await embedder.embed_query("Ma connexion internet ne fonctionne plus")
    hits = await store.search(tenant_id, q[0], top_k=3)
    assert hits, "expected retrieval hits"
    for h in hits:
        print(f"score={h.score:.3f} page={h.chunk.page} text={h.chunk.text[:80]}...")
    print("SMOKE_OK", tenant_id, getattr(store, "name", store.__class__.__name__))
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
