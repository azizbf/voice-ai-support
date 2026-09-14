"""Measure first audible TTS vs full Gemini generation; check phrase coverage.

Run from backend: .venv/Scripts/python.exe -m scripts.bench_tts_overlap
Uses the production VoiceSession path (RAG + Gemini stream + Edge TTS).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from types import SimpleNamespace

from starlette.websockets import WebSocketState

from app.core.config import get_settings
from app.services.voice.session import VoiceSession
from app.services.voice.tts import get_tts_provider

QUESTIONS = [
    "Combien coûte le forfait Fibre 100 ?",
    "Ma connexion internet ne fonctionne plus, que dois-je faire ?",
    "Comment réinitialiser mon routeur ?",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())


def _coverage(spoken: list[str], answer: str) -> dict:
    joined = " ".join(spoken).strip()
    spoken_words = _words(joined)
    answer_words = _words(answer)
    missing = []
    index = 0
    for word in answer_words:
        try:
            found = spoken_words.index(word, index)
        except ValueError:
            missing.append(word)
            continue
        index = found + 1
    repeated = [spoken[i] for i in range(1, len(spoken)) if spoken[i] == spoken[i - 1]]
    extra = []
    index = 0
    for word in spoken_words:
        try:
            found = answer_words.index(word, index)
        except ValueError:
            extra.append(word)
            continue
        index = found + 1
    return {
        "phrases": spoken,
        "joined": joined,
        "missing_words": missing,
        "extra_words": extra,
        "repeated_phrases": repeated,
        "complete": not missing and not extra and not repeated,
    }


async def _ready_tenant() -> str:
    from app.services.rag.service import rag_service
    from app.services.session.service import session_service

    meta = await session_service.create_tenant()
    await rag_service.start_demo_ingest(meta.tenant_id)
    for _ in range(50):
        status = await session_service.load_meta(meta.tenant_id)
        if status and status.status == "ready":
            return meta.tenant_id
        await asyncio.sleep(0.3)
    raise RuntimeError("tenant not ready")


class CaptureWS:
    def __init__(self) -> None:
        self.client_state = WebSocketState.CONNECTED
        self.events: list[dict] = []
        self.first_audio_at: float | None = None
        self.audio_bytes = 0

    async def send_text(self, text: str) -> None:
        payload = json.loads(text)
        self.events.append({"t": time.perf_counter(), **payload})

    async def send_bytes(self, data: bytes) -> None:
        now = time.perf_counter()
        if self.first_audio_at is None:
            self.first_audio_at = now
        self.audio_bytes += len(data)


async def run_turn(tenant_id: str, question: str, provider) -> dict:
    spoken: list[str] = []
    original = provider.synthesize_stream

    async def wrapped(text: str):
        spoken.append(text)
        async for audio in original(text):
            yield audio

    provider.synthesize_stream = wrapped
    ws = CaptureWS()
    session = VoiceSession(ws, tenant_id)
    t0 = time.perf_counter()
    try:
        await session._respond_text(question, t0)
    finally:
        provider.synthesize_stream = original

    timings = {
        event["stage"]: event
        for event in ws.events
        if event.get("type") == "debug_timing"
    }
    latency = next(event for event in ws.events if event.get("type") == "latency")
    transcript = next(
        event["text"]
        for event in ws.events
        if event.get("type") == "transcript" and event.get("role") == "assistant"
    )
    coverage = _coverage(spoken, transcript)
    first_audio = latency.get("tts_first_audio_ms")
    llm_stream = latency.get("llm_stream_ms")
    return {
        "question": question,
        "answer": transcript,
        "first_clause": spoken[0] if spoken else None,
        "phrase_count": len(spoken),
        "llm_first_token_ms": round(latency.get("llm_first_token_ms") or 0, 1),
        "llm_phrase_ms": round(timings["llm_phrase"]["offset_ms"] + timings["llm_phrase"]["duration_ms"], 1)
        if "llm_phrase" in timings
        else None,
        "llm_stream_ms": round(llm_stream or 0, 1),
        "tts_first_audio_ms": round(first_audio or 0, 1),
        "tts_stage_ms": round(timings["tts"]["duration_ms"], 1) if "tts" in timings else None,
        "audio_lead_ms": round((llm_stream or 0) - (first_audio or 0), 1),
        "tts_overlapped_llm": bool(latency.get("tts_overlapped_llm")),
        "tts_already_started": bool((timings.get("llm_stream") or {}).get("tts_already_started")),
        "audio_bytes": ws.audio_bytes,
        **coverage,
    }


async def main() -> int:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY missing", file=sys.stderr)
        return 2
    tenant_id = await _ready_tenant()
    provider = get_tts_provider()
    ensure = getattr(provider, "ensure_ready", None)
    if callable(ensure):
        await ensure()
    rows = []
    for question in QUESTIONS:
        row = await run_turn(tenant_id, question, provider)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True, indent=2))
    overlapped = sum(1 for row in rows if row["tts_overlapped_llm"])
    complete = sum(1 for row in rows if row["complete"])
    print(json.dumps({
        "turns": len(rows),
        "overlapped": overlapped,
        "complete_speech": complete,
        "first_audio_ms": [row["tts_first_audio_ms"] for row in rows],
        "audio_lead_ms": [row["audio_lead_ms"] for row in rows],
    }, ensure_ascii=True))
    close = getattr(provider, "close", None)
    if callable(close):
        await close()
    if complete < len(rows) or overlapped < len(rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
