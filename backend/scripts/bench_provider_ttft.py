"""Interleaved Gemini 3.5 Flash-Lite vs configured Anthropic Haiku TTFT.

Does not change app defaults. Reuses HTTP connections. Counts retries separately.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time

from anthropic import APIStatusError, AsyncAnthropic, DefaultAsyncHttpxClient, RateLimitError
import httpx

from app.core.config import get_settings
from app.services.llm.gemini import GeminiClient, _http_client
from app.services.voice.session import VOICE_SYSTEM_PROMPT, _speech_prefix
from scripts.bench_llm_ttft import QUESTIONS, load_prompts, pct, score_answer, stream_once as gemini_stream_once

MAX_TOKENS = 80
PACE_S = 0.55
MAX_ATTEMPTS = 4


def _retryable(status: int | None) -> bool:
    return status in {408, 409, 429, 500, 502, 503, 529}


async def anthropic_stream_once(client: AsyncAnthropic, model: str, messages: list[dict[str, str]]) -> dict:
    retries = 0
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        request_started = time.perf_counter()
        first_token_ms = None
        first_clause_ms = None
        first_clause = None
        buf = ""
        try:
            async with client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                temperature=0.2,
                system=VOICE_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    now = (time.perf_counter() - request_started) * 1000
                    if first_token_ms is None:
                        first_token_ms = now
                    buf += text
                    if first_clause_ms is None:
                        clause = _speech_prefix(buf)
                        if clause:
                            first_clause, first_clause_ms = clause, now
            break
        except RateLimitError as exc:
            retries += 1
            last_error = exc
            await asyncio.sleep(1.5 * (attempt + 1))
        except APIStatusError as exc:
            last_error = exc
            if _retryable(getattr(exc, "status_code", None)):
                retries += 1
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    else:
        raise last_error or RuntimeError("Anthropic stream failed")
    if first_clause_ms is None and buf.strip():
        first_clause_ms = (time.perf_counter() - request_started) * 1000
        first_clause = buf.strip()
    return {
        "ttft_ms": first_token_ms,
        "clause_ms": first_clause_ms,
        "clause": first_clause,
        "answer": buf.strip(),
        "retries": retries,
        "failed": False,
    }


async def gemini_once(client: GeminiClient, messages: list[dict[str, str]]) -> dict:
    row = await gemini_stream_once(client, messages)
    row["failed"] = False
    row.setdefault("retries", 0)
    return row


def summarize(model: str, rows: list[dict]) -> dict:
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None and not r.get("failed")]
    clause = [r["clause_ms"] for r in rows if r.get("clause_ms") is not None and not r.get("failed")]
    checks = [item for r in rows if not r.get("failed") for item in r.get("checks", [])]
    passed = sum(1 for _, ok in checks if ok)
    return {
        "model": model,
        "n": len(rows),
        "failed_requests": sum(1 for r in rows if r.get("failed")),
        "retries": sum(int(r.get("retries") or 0) for r in rows),
        "ttft_median_ms": round(statistics.median(ttft)) if ttft else None,
        "ttft_p95_ms": round(pct(ttft, 95)) if ttft else None,
        "clause_median_ms": round(statistics.median(clause)) if clause else None,
        "clause_p95_ms": round(pct(clause, 95)) if clause else None,
        "correctness": f"{passed}/{len(checks)}" if checks else "0/0",
    }


async def main() -> int:
    settings = get_settings()
    print("PROVIDERS", json.dumps({
        "active_default": settings.llm_provider,
        "gemini_model": settings.gemini_model,
        "gemini_key": bool(settings.gemini_api_key),
        "anthropic_key": bool(settings.anthropic_api_key),
        "anthropic_voice_model": settings.anthropic_voice_model,
        "anthropic_model": settings.anthropic_model,
        "max_tokens": MAX_TOKENS,
    }, ensure_ascii=True))
    if not settings.gemini_api_key:
        print("STOP: GEMINI_API_KEY required for the current default baseline.", file=sys.stderr)
        return 2
    if not settings.anthropic_api_key:
        print("STOP: alternative provider is Anthropic; ANTHROPIC_API_KEY is missing.", file=sys.stderr)
        return 3

    haiku = settings.anthropic_voice_model or "claude-haiku-4-5"
    gemini_http = _http_client(settings.gemini_api_key, 30)
    anthropic = AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        http_client=DefaultAsyncHttpxClient(limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=20, keepalive_expiry=120.0,
        )),
        max_retries=0,
    )
    gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model, gemini_http)
    try:
        try:
            models = await anthropic.models.list(limit=5)
            print("ANTHROPIC_OK", json.dumps({
                "voice_model": haiku,
                "listed": [getattr(m, "id", None) for m in getattr(models, "data", [])[:5]],
            }, ensure_ascii=True))
        except APIStatusError as exc:
            print("STOP: Anthropic credential rejected.", json.dumps({
                "status": exc.status_code,
                "needed": "valid ANTHROPIC_API_KEY",
            }, ensure_ascii=True), file=sys.stderr)
            return 4

        _tid, prompts = await load_prompts()
        dummy = [{"role": "user", "content": "Réponds: oui"}]
        await gemini_once(gemini, dummy)
        await asyncio.sleep(PACE_S)
        await anthropic_stream_once(anthropic, haiku, dummy)
        await asyncio.sleep(PACE_S)

        results = {settings.gemini_model: [], haiku: []}
        print("RUNS")
        for i in range(len(QUESTIONS) * 2):
            prompt = prompts[i % len(prompts)]
            for label, kind in (
                (settings.gemini_model, "gemini"),
                (haiku, "anthropic"),
            ):
                try:
                    if kind == "gemini":
                        row = await gemini_once(gemini, prompt["messages"])
                    else:
                        row = await anthropic_stream_once(anthropic, haiku, prompt["messages"])
                except Exception as exc:
                    row = {
                        "failed": True,
                        "retries": MAX_ATTEMPTS - 1,
                        "error": f"{type(exc).__name__}",
                        "ttft_ms": None,
                        "clause_ms": None,
                        "answer": "",
                        "checks": [],
                    }
                else:
                    row["checks"] = score_answer(prompt["question"], row["answer"], prompt["kb"])
                    row["failed"] = False
                row.update({"model": label, "n": i, "q": prompt["question"]})
                results[label].append(row)
                ok = (not row.get("failed")) and all(pass_ for _, pass_ in row.get("checks", []))
                print(json.dumps({
                    "model": label,
                    "n": i,
                    "q": prompt["question"],
                    "ttft_ms": None if row.get("ttft_ms") is None else round(row["ttft_ms"]),
                    "clause_ms": None if row.get("clause_ms") is None else round(row["clause_ms"]),
                    "retries": row.get("retries", 0),
                    "failed": bool(row.get("failed")),
                    "ok": ok,
                    "answer": (row.get("answer") or "")[:160],
                    "fail": [name for name, pass_ in row.get("checks", []) if not pass_],
                    "error": row.get("error"),
                }, ensure_ascii=True))
                await asyncio.sleep(PACE_S)

        print("SUMMARY")
        for model, rows in results.items():
            print(json.dumps(summarize(model, rows), ensure_ascii=True))
        return 0
    finally:
        await gemini_http.aclose()
        await anthropic.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
