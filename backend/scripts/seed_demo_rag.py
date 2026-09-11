"""Seed the built-in demo FAQ into a new tenant and print retrieval checks.

Usage (from backend/):
  python scripts/seed_demo_rag.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.security import issue_tenant_token
from app.services.rag.demo_knowledge import demo_test_questions
from app.services.rag.service import rag_service
from app.services.session.service import session_service


async def main() -> None:
    meta = await session_service.create_tenant()
    tenant_id = meta.tenant_id
    exp = meta.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    token = issue_tenant_token(tenant_id, exp.timestamp())

    print("Indexing demo FAQ…")
    info = await rag_service.ingest_demo_knowledge(tenant_id)
    print(
        f"READY tenant={tenant_id} pages={info.page_count} chunks={info.chunk_count}"
    )
    print(f"TOKEN={token}")
    print()
    print("Sample retrievals:")
    for q in demo_test_questions()[:4]:
        chunks, ms = await rag_service.retrieve(tenant_id, q, top_k=2)
        top = chunks[0] if chunks else None
        preview = (top.text[:90] + "…") if top else "(no hit)"
        score = f"{top.score:.3f}" if top else "-"
        print(f"- [{ms:.0f}ms] {q}")
        print(f"  score={score} page={top.page if top else '-'} {preview}")
    print()
    print("In the UI: click « Charger la FAQ démo » (or paste tenant + token into localStorage).")
    print(
        'localStorage.setItem("auralis_demo", JSON.stringify({'
        f'tenant_id:"{tenant_id}",token:"{token}"'
        "}))"
    )


if __name__ == "__main__":
    asyncio.run(main())
