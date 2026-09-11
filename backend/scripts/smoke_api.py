"""End-to-end HTTP API smoke test (RAG retrieve; chat skipped if no ANTHROPIC_API_KEY)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

API = os.getenv("API_URL", "http://127.0.0.1:8000")
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "faq_internet.pdf"


def main() -> None:
    with httpx.Client(base_url=API, timeout=120) as client:
        h = client.get("/health")
        h.raise_for_status()
        print("health", h.json())

        t = client.post("/api/v1/tenants")
        t.raise_for_status()
        body = t.json()
        tid, token = body["tenant_id"], body["token"]
        headers = {"X-Tenant-Token": token}
        print("tenant", tid)

        with SAMPLE.open("rb") as f:
            up = client.post(
                f"/api/v1/tenants/{tid}/documents",
                headers=headers,
                files={"file": ("faq_internet.pdf", f, "application/pdf")},
            )
        up.raise_for_status()
        print("upload", up.json())

        for _ in range(60):
            st = client.get(f"/api/v1/tenants/{tid}/status", headers=headers)
            st.raise_for_status()
            status = st.json()
            print("status", status["status"], status.get("pipeline"))
            if status["status"] == "ready":
                break
            if status["status"] == "error":
                raise SystemExit(status)
            time.sleep(1)
        else:
            raise SystemExit("timeout waiting for ready")

        ret = client.post(
            "/api/v1/rag/retrieve",
            headers=headers,
            json={"tenant_id": tid, "query": "connexion internet ne fonctionne plus", "top_k": 3},
        )
        ret.raise_for_status()
        data = ret.json()
        assert data["chunks"], "expected chunks"
        print("retrieve", json.dumps(data, ensure_ascii=False, indent=2)[:500])

        if os.getenv("ANTHROPIC_API_KEY") or Path(__file__).resolve().parents[1].joinpath(".env").read_text(encoding="utf-8").find("ANTHROPIC_API_KEY=sk") >= 0:
            # only call chat if key looks set
            env_text = Path(__file__).resolve().parents[1].joinpath(".env").read_text(encoding="utf-8")
            key_line = [ln for ln in env_text.splitlines() if ln.startswith("ANTHROPIC_API_KEY=")]
            if key_line and len(key_line[0].split("=", 1)[1].strip()) > 10:
                chat = client.post(
                    "/api/v1/chat",
                    headers=headers,
                    json={"tenant_id": tid, "message": "Ma connexion ne marche plus, que faire ?", "stream": False},
                )
                print("chat", chat.status_code, chat.text[:400])
            else:
                print("skip chat (no ANTHROPIC_API_KEY)")
        else:
            print("skip chat (no ANTHROPIC_API_KEY)")

        cleared = client.delete(f"/api/v1/tenants/{tid}/knowledge", headers=headers)
        cleared.raise_for_status()
        print("cleared", cleared.json())
        print("API_SMOKE_OK")


if __name__ == "__main__":
    main()
