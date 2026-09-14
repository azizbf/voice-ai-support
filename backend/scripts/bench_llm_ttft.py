"""Interleaved Gemini first-token benchmark on real RAG voice prompts."""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import httpx

from app.core.config import get_settings
from app.services.llm.claude import claude_service
from app.services.llm.gemini import GeminiClient, _http_client
from app.services.voice.session import VOICE_SYSTEM_PROMPT, VoiceSession, _speech_prefix

QUESTIONS = [
    "Quelles sont les offres que vous avez ?",
    "Combien coûte le forfait Fibre 100 ?",
    "Ma connexion internet ne fonctionne plus, que dois-je faire ?",
    "Comment réinitialiser mon routeur ?",
    "Je veux résilier mon abonnement, quelle est la procédure ?",
    "Je déménage à Sousse, comment transférer ma ligne ?",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def score_answer(question: str, answer: str, kb: str) -> list[tuple[str, bool]]:
    q, a, kb_l = question.lower(), answer.lower(), kb.lower()
    checks: list[tuple[str, bool]] = []
    if "offres" in q:
        checks.append(("39 DT", "39" in a))
        checks.append(("59 DT", "59" in a))
        checks.append(("89 DT", "89" in a))
        checks.append(("devise DT", "dt" in a))
    elif "fibre 100" in q:
        if "59" in kb:
            checks.append(("prix 59 DT", "59" in a))
        if "50 dt" in kb_l or "50 DT" in kb:
            checks.append(("frais 50 DT", "50" in a))
        checks.append(("devise DT", "dt" in a))
    elif "connexion" in q or "fonctionne plus" in q:
        checks.append(("étape A restart", "éteindre" in a or "rallumer" in a or "routeur" in a))
        dumped = sum(x in a for x in ("rj45", "smartphone", "voyant"))
        checks.append(("une étape, pas tout le diagnostic", dumped <= 1))
    elif "réinitialiser" in q:
        checks.append(("bouton reset", "reset" in a or "bouton" in a or "trombone" in a))
        dumped = sum(x in a for x in ("10 secondes", "5 minutes", "ssid", "usine"))
        checks.append(("une étape, pas toute la procédure", dumped <= 1))
    elif "résilier" in q:
        checks.append(("préavis 30 jours", "30" in a))
    elif "déménage" in q or "transférer" in q:
        checks.append(("frais 40 DT ou délai 7-10j", "40" in a or "7" in a or "10" in a))
    checks.append(("français", any(c in a for c in "éèàùçô")))
    checks.append(("pas d'escalade gratuite", "conseiller humain" not in a and "je n'ai pas" not in a))
    return checks


async def thinking_probe(http: httpx.AsyncClient, model: str, thinking: dict) -> dict:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Réponds uniquement: oui"}]}],
        "generationConfig": {"maxOutputTokens": 8, "temperature": 0, "thinkingConfig": thinking},
    }
    started = time.perf_counter()
    response = await http.post(f"models/{model}:generateContent", json=payload, timeout=20)
    elapsed = (time.perf_counter() - started) * 1000
    body = {}
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text[:200]}
    usage = body.get("usageMetadata") or {}
    err = None
    if response.status_code >= 400:
        err = (body.get("error") or {}).get("message") or response.text[:180]
    return {
        "model": model,
        "thinking": thinking,
        "status": response.status_code,
        "ms": round(elapsed),
        "thoughts": usage.get("thoughtsTokenCount"),
        "error": err,
    }


async def stream_once(client: GeminiClient, messages: list[dict[str, str]]) -> dict:
    prep_started = time.perf_counter()
    payload = client.payload(VOICE_SYSTEM_PROMPT, messages, 80)
    local_prep_ms = (time.perf_counter() - prep_started) * 1000
    last_error: Exception | None = None
    for attempt in range(4):
        request_started = time.perf_counter()
        first_token_ms = None
        first_clause_ms = None
        first_clause = None
        buf = ""
        try:
            async with client.http.stream(
                "POST",
                f"models/{client.model}:streamGenerateContent",
                params={"alt": "sse"},
                json=payload,
                timeout=client.timeout,
            ) as response:
                if response.status_code in {429, 503}:
                    await response.aread()
                    await asyncio.sleep(1.5 * (attempt + 1))
                    last_error = RuntimeError(str(response.status_code))
                    continue
                response.raise_for_status()

                async def tokens():
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data = json.loads(line[5:].strip())
                            if "error" in data:
                                raise RuntimeError("Gemini streaming request failed")
                            text = client.text(data)
                            if text:
                                yield text

                async for text in tokens():
                    now = (time.perf_counter() - request_started) * 1000
                    if first_token_ms is None:
                        first_token_ms = now
                    buf += text
                    if first_clause_ms is None:
                        clause = _speech_prefix(buf)
                        if clause:
                            first_clause, first_clause_ms = clause, now
            break
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code in {429, 503}:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    else:
        raise last_error or RuntimeError("Gemini stream failed")
    if first_clause_ms is None and buf.strip():
        first_clause_ms = (time.perf_counter() - request_started) * 1000
        first_clause = buf.strip()
    return {
        "local_prep_ms": local_prep_ms,
        "payload_bytes": len(json.dumps(payload).encode("utf-8")),
        "ttft_ms": first_token_ms,
        "clause_ms": first_clause_ms,
        "clause": first_clause,
        "answer": buf.strip(),
        "retries": attempt if last_error is None else attempt,
    }


async def load_prompts() -> tuple[str, list[dict]]:
    from app.services.rag.service import rag_service
    from app.services.session.service import session_service

    meta = await session_service.create_tenant()
    await rag_service.start_demo_ingest(meta.tenant_id)
    for _ in range(50):
        status = await session_service.load_meta(meta.tenant_id)
        if status and status.status == "ready":
            break
        await asyncio.sleep(0.3)
    else:
        raise RuntimeError("tenant not ready")
    session = VoiceSession(SimpleNamespace(client_state=None), meta.tenant_id)
    prompts: list[dict] = []
    print("PROMPTS")
    for question in QUESTIONS:
        chunks, _latency = await rag_service.retrieve(meta.tenant_id, question, top_k=2)
        context = session._voice_context(chunks)
        messages = session._claude_messages(question, context)
        print(json.dumps({
            "q": question,
            "sys_chars": len(VOICE_SYSTEM_PROMPT),
            "kb_chars": len(context),
            "msg_chars": len(messages[-1]["content"]),
            "history": len(messages) - 1,
            "prices_in_kb": {k: k in context for k in ("39", "59", "89", "49", "40", "30")},
        }, ensure_ascii=True))
        prompts.append({"question": question, "messages": messages, "kb": context})
    return meta.tenant_id, prompts


async def pick_alt(http: httpx.AsyncClient, names: list[str], current: str) -> str | None:
    preferred = (
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
    )
    candidates = [name for name in preferred if name in names and name != current]
    candidates.extend(
        n for n in names
        if "flash-lite" in n and "image" not in n and "tts" not in n and n != current and n not in candidates
    )
    for name in candidates:
        probe = await thinking_probe(
            http,
            name,
            {"thinkingLevel": "minimal"} if name.startswith("gemini-3") else {"thinkingBudget": 0},
        )
        print("ALT_PROBE", json.dumps(probe, ensure_ascii=True))
        if probe["status"] == 200:
            return name
        await asyncio.sleep(0.3)
    return None


async def main() -> int:
    settings = get_settings()
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY missing", file=sys.stderr)
        return 2
    current = settings.gemini_model
    gemini_http = _http_client(settings.gemini_api_key, 30)
    try:
        listed = await gemini_http.get("models", params={"pageSize": 100})
        listed.raise_for_status()
        print("INSPECT", json.dumps({
            "llm_provider": settings.llm_provider,
            "model": current,
            "http_version": listed.http_version,
            "http2": listed.http_version == "HTTP/2",
            "voice_system_chars": len(VOICE_SYSTEM_PROMPT),
            "history_turns": 2,
            "voice_max_tokens": 80,
            "thinking_3x": "thinkingLevel=minimal",
        }, ensure_ascii=True))
        names = [m["name"].split("/", 1)[-1] for m in listed.json().get("models", [])]
        lite = [n for n in names if "flash-lite" in n and "image" not in n and "tts" not in n]
        print("FLASH_LITE", lite)
        alt = await pick_alt(gemini_http, names, current)
        if not alt:
            print("NO_ALT_MODEL", file=sys.stderr)
            return 3
        print("CURRENT", current, "ALT", alt)

        probes = [
            await thinking_probe(gemini_http, current, {"thinkingLevel": "minimal"}),
        ]
        print("THINKING")
        for row in probes:
            print(json.dumps(row, ensure_ascii=True))

        tenant_id, prompts = await load_prompts()
        offers = next(p for p in prompts if "offres" in p["question"].lower())
        print("OFFERS_KB", json.dumps({
            "has_39": "39" in offers["kb"],
            "has_59": "59" in offers["kb"],
            "has_89": "89" in offers["kb"],
            "kb_preview": offers["kb"][:240],
        }, ensure_ascii=True))
        baseline = await claude_service.answer(tenant_id, offers["question"])
        baseline_checks = score_answer(offers["question"], baseline.answer, offers["kb"])
        print("TEXT_BASELINE", json.dumps({
            "ok": all(ok for _, ok in baseline_checks),
            "checks": baseline_checks,
            "answer": baseline.answer[:400],
        }, ensure_ascii=True))
        if not all(ok for _, ok in baseline_checks):
            print("TEXT_BASELINE_FAILED", file=sys.stderr)
            return 4

        current_client = GeminiClient(settings.gemini_api_key, current, gemini_http)
        clients = [current_client]
        if os.getenv("BENCH_ALT", "1") != "0":
            clients.append(GeminiClient(settings.gemini_api_key, alt, gemini_http))
        await current_client.models.list(limit=1)
        for client in clients:
            await stream_once(client, prompts[0]["messages"])
            await asyncio.sleep(0.45)

        results = {client.model: [] for client in clients}
        print("RUNS")
        for i in range(12):
            prompt = prompts[i % len(prompts)]
            for client in clients:
                row = await stream_once(client, prompt["messages"])
                await asyncio.sleep(0.45)
                row.update({
                    "model": client.model,
                    "n": i,
                    "q": prompt["question"],
                    "checks": score_answer(prompt["question"], row["answer"], prompt["kb"]),
                })
                results[client.model].append(row)
                ok = all(pass_ for _, pass_ in row["checks"])
                print(json.dumps({
                    "model": client.model,
                    "n": i,
                    "q": prompt["question"],
                    "local_prep_ms": round(row["local_prep_ms"], 2),
                    "payload_bytes": row["payload_bytes"],
                    "ttft_ms": None if row["ttft_ms"] is None else round(row["ttft_ms"]),
                    "clause_ms": None if row["clause_ms"] is None else round(row["clause_ms"]),
                    "ok": ok,
                    "answer": row["answer"][:180],
                    "fail": [name for name, pass_ in row["checks"] if not pass_],
                }, ensure_ascii=True))

        print("SUMMARY")
        for model, rows in results.items():
            ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
            clause = [r["clause_ms"] for r in rows if r["clause_ms"] is not None]
            prep = [r["local_prep_ms"] for r in rows]
            checks = [item for r in rows for item in r["checks"]]
            passed = sum(1 for _, ok in checks if ok)
            print(json.dumps({
                "model": model,
                "n": len(rows),
                "local_prep_median_ms": round(statistics.median(prep), 2) if prep else None,
                "ttft_median_ms": round(statistics.median(ttft)) if ttft else None,
                "ttft_p95_ms": round(pct(ttft, 95)) if ttft else None,
                "clause_median_ms": round(statistics.median(clause)) if clause else None,
                "clause_p95_ms": round(pct(clause, 95)) if clause else None,
                "correctness": f"{passed}/{len(checks)}",
            }, ensure_ascii=True))
        return 0
    finally:
        await gemini_http.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
