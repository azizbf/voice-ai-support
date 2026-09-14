"""Send one real audio clip through the running voice websocket."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import websockets

API = "http://127.0.0.1:8000"
AUDIO = Path(__file__).resolve().parents[1] / "samples" / "stt-webm-short.webm"


async def main() -> int:
    if not AUDIO.is_file() or AUDIO.stat().st_size < 100:
        print("NO_AUDIO", AUDIO, file=sys.stderr)
        return 2
    audio = AUDIO.read_bytes()
    print("AUDIO_BYTES", len(audio))
    async with httpx.AsyncClient(base_url=API, timeout=60) as http:
        health = (await http.get("/health")).json()
        print("HEALTH", json.dumps(health))
        created = (await http.post("/api/v1/tenants")).json()
        tid, token = created["tenant_id"], created["token"]
        print("TENANT", tid)
        status = None
        for _ in range(45):
            status = (
                await http.get(f"/api/v1/tenants/{tid}/status", headers={"X-Tenant-Token": token})
            ).json()
            print("STATUS", status.get("status"), status.get("pipeline"))
            if status.get("status") == "ready":
                break
            if status.get("status") == "error":
                print("TENANT_ERROR", status)
                return 3
            await asyncio.sleep(1)
        else:
            print("TENANT_NOT_READY", status)
            return 3
    uri = f"ws://127.0.0.1:8000/api/v1/voice/{tid}?token={token}"
    async with websockets.connect(uri, max_size=8_000_000) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 90))
        print("READY", json.dumps(ready, ensure_ascii=True))
        await ws.send(audio)
        await ws.send(json.dumps({"type": "audio_end", "turn_id": "probe-stt-1"}))
        outcome = "timeout"
        for _ in range(40):
            raw = await asyncio.wait_for(ws.recv(), 30)
            if isinstance(raw, bytes):
                print("AUDIO_OUT", len(raw))
                continue
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "debug_timing":
                keys = (
                    "stage",
                    "duration_ms",
                    "failed",
                    "stt_error",
                    "stt_error_type",
                    "stt_impl",
                    "stt_queue_ms",
                    "stt_decode_ms",
                    "stt_infer_ms",
                    "stt_encode_ms",
                    "stt_keep_awake",
                )
                print("DEBUG", {k: msg.get(k) for k in keys})
            elif kind == "error":
                print("ERROR", json.dumps(msg, ensure_ascii=True))
                outcome = "error"
            elif kind == "transcript":
                print("TRANSCRIPT", json.dumps(msg, ensure_ascii=True)[:500])
                if msg.get("role") == "user":
                    outcome = "transcript"
            elif kind in ("status", "tts_start", "tts_end", "latency"):
                print(kind.upper(), {k: msg.get(k) for k in msg if k != "type"})
            else:
                print("WS", kind, list(msg))
            if kind == "error":
                break
            if kind == "tts_end" or (kind == "status" and msg.get("state") == "listening" and outcome != "timeout"):
                break
        print("OUTCOME", outcome)
        return 0 if outcome == "transcript" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
