from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.config import get_settings
from app.core.logging_metrics import LatencySample, latency_store
from app.services.llm.claude import claude_service
from app.services.rag.service import SYSTEM_PROMPT, rag_service
from app.services.voice.stt import get_stt_provider
from app.services.voice.tts import get_tts_provider

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"^(.+?[.!?…])(?:\s+|$)")


def _first_sentence(buffer: str) -> str | None:
    text = buffer.strip()
    if not text:
        return None
    match = _SENTENCE_END.match(text)
    if match and len(match.group(1)) >= 12:
        return match.group(1).strip()
    return None


class VoiceSession:
    def __init__(self, websocket: WebSocket, tenant_id: str) -> None:
        self.ws = websocket
        self.tenant_id = tenant_id
        self.audio_buffer = bytearray()
        self.cancel_event = asyncio.Event()
        self.busy = False
        self._response_task: asyncio.Task | None = None
        # Multi-turn memory so guided procedures can verify step-by-step
        self.history: list[dict[str, str]] = []

    def _retrieval_query(self, transcript: str) -> str:
        """Keep procedure context when the user only confirms a step."""
        words = transcript.split()
        if len(words) <= 10 and self.history:
            prior = " ".join(m["content"] for m in self.history[-4:])
            return f"{prior}\nMessage client: {transcript}"
        return transcript

    def _claude_messages(self, transcript: str, context: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = list(self.history[-4:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Contexte KB:\n{context}\n\n"
                    f"Client: {transcript}\n\n"
                    "Réponds à l'oral, français, UNE phrase courte. "
                    "Procédure: une étape + confirmation. Prix: cite-le."
                ),
            }
        )
        return messages

    def _voice_context(self, chunks: list) -> str:
        """Compact context for faster Claude turns."""
        if not chunks:
            return "Aucune information pertinente."
        parts: list[str] = []
        for i, c in enumerate(chunks[:2], start=1):
            text = c.text.strip()
            if len(text) > 450:
                text = text[:450].rsplit(" ", 1)[0] + "…"
            parts.append(f"[{i}|p.{c.page}] {text}")
        return "\n".join(parts)

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.ws.client_state == WebSocketState.CONNECTED:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_audio(self, data: bytes) -> None:
        """Send complete MP3: JSON markers + binary body (chunked if needed)."""
        if self.ws.client_state != WebSocketState.CONNECTED or not data:
            return
        logger.info("TTS ready (%s bytes) — streaming to client", len(data))
        await self.send_json(
            {"type": "tts_start", "mime": "audio/mpeg", "bytes": len(data)}
        )
        # Prefer one frame when possible (lower WS overhead)
        chunk_size = 48_000
        if len(data) <= chunk_size:
            await self.ws.send_bytes(data)
        else:
            for i in range(0, len(data), chunk_size):
                if self.ws.client_state != WebSocketState.CONNECTED:
                    return
                await self.ws.send_bytes(data[i : i + chunk_size])
        await self.send_json({"type": "tts_end"})

    async def interrupt(self) -> None:
        self.cancel_event.set()
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        self.cancel_event.clear()
        await self.send_json({"type": "status", "state": "listening"})

    async def handle_user_text(self, transcript: str) -> None:
        transcript = transcript.strip()
        if not transcript:
            return
        if self.busy:
            await self.interrupt()
        self.busy = True
        self.cancel_event.clear()
        self._response_task = asyncio.create_task(self._respond_text(transcript, t0=time.perf_counter()))
        try:
            await self._response_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Voice response failed")
            message = str(exc)
            if "ANTHROPIC_API_KEY" in message:
                message = (
                    "Clé Anthropic manquante ou non chargée. "
                    "Vérifiez backend/.env puis redémarrez l'API."
                )
            await self.send_json({"type": "error", "message": message})
            await self.send_json({"type": "status", "state": "listening"})
        finally:
            self.busy = False

    async def handle_utterance(self, audio: bytes) -> None:
        if self.busy:
            await self.interrupt()
        self.busy = True
        self.cancel_event.clear()
        self._response_task = asyncio.create_task(self._respond_audio(audio))
        try:
            await self._response_task
        except asyncio.CancelledError:
            pass
        finally:
            self.busy = False

    async def _respond_audio(self, audio: bytes) -> None:
        t0 = time.perf_counter()
        await self.send_json({"type": "status", "state": "thinking"})
        stt = get_stt_provider()
        transcript = await stt.transcribe(audio)
        if not transcript:
            await self.send_json(
                {
                    "type": "error",
                    "message": "Transcription indisponible. Utilisez la reconnaissance navigateur ou configurez Deepgram.",
                }
            )
            await self.send_json({"type": "status", "state": "listening"})
            return
        await self._respond_text(transcript, t0=t0)

    async def _respond_text(self, transcript: str, t0: float) -> None:
        await self.send_json({"type": "status", "state": "thinking"})
        await self.send_json(
            {"type": "transcript", "role": "user", "text": transcript, "final": True}
        )

        speech_end_to_retrieval = (time.perf_counter() - t0) * 1000
        retrieve_q = self._retrieval_query(transcript)
        chunks, retrieval_ms = await rag_service.retrieve(self.tenant_id, retrieve_q, top_k=2)
        sources = rag_service.to_sources(chunks)
        await self.send_json({"type": "sources", "items": [s.model_dump() for s in sources]})

        context = self._voice_context(chunks)
        claude_messages = self._claude_messages(transcript, context)

        await self.send_json({"type": "status", "state": "thinking"})
        full_answer: list[str] = []
        first_token_ms: float | None = None
        first_audio_ms: float | None = None
        speak_text = ""
        tts_task: asyncio.Task[tuple[bytes, str]] | None = None

        settings = get_settings()
        voice_model = settings.anthropic_voice_model or settings.anthropic_model

        async with claude_service.client.messages.stream(
            model=voice_model,
            max_tokens=80,
            system=SYSTEM_PROMPT,
            messages=claude_messages,
        ) as stream:
            async for text in stream.text_stream:
                if self.cancel_event.is_set():
                    break
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                full_answer.append(text)
                # Start Vivienne as soon as the first oral sentence is ready,
                # overlapping the rest of Claude's stream (big latency win).
                if tts_task is None:
                    candidate = _first_sentence("".join(full_answer))
                    if candidate:
                        speak_text = candidate
                        from app.services.voice.tts import synthesize_with_fallback

                        tts_task = asyncio.create_task(synthesize_with_fallback(speak_text))

        answer_text = "".join(full_answer).strip()
        # Prefer the first complete sentence for speech (matches oral style);
        # fall back to the full answer if punctuation never arrived.
        early = _first_sentence(answer_text) if answer_text else None
        speak_text = early or answer_text

        if answer_text:
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": answer_text})
            if len(self.history) > 12:
                self.history = self.history[-12:]

            await self.send_json(
                {
                    "type": "transcript",
                    "role": "assistant",
                    "text": answer_text,
                    "final": True,
                }
            )
            if not self.cancel_event.is_set() and speak_text:
                await self.send_json({"type": "status", "state": "speaking"})
                try:
                    from app.services.voice.tts import synthesize_with_fallback

                    audio: bytes
                    if tts_task is not None and early and early == speak_text:
                        audio, _provider = await tts_task
                    else:
                        if tts_task is not None and not tts_task.done():
                            tts_task.cancel()
                        audio, _provider = await synthesize_with_fallback(speak_text)
                    first_audio_ms = (time.perf_counter() - t0) * 1000
                    await self.send_audio(audio)
                except Exception:
                    logger.exception("Server TTS failed — client HTTP fallback")
                    if tts_task is not None and not tts_task.done():
                        tts_task.cancel()
                    await self.send_json({"type": "speak", "text": speak_text})
                    first_audio_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t0) * 1000
        sample = LatencySample(
            tenant_id=self.tenant_id,
            path="voice",
            speech_end_to_retrieval_ms=speech_end_to_retrieval,
            retrieval_ms=retrieval_ms,
            llm_first_token_ms=first_token_ms,
            tts_first_audio_ms=first_audio_ms,
            total_ms=total_ms,
            retrieval_scores=[c.score for c in chunks],
            chunks_retrieved=len(chunks),
            source_pages=[{"document_name": s.document_name, "page": s.page} for s in sources],
        )
        latency_store.add(sample)

        async def _persist() -> None:
            try:
                from app.db.repository import db_save_turn

                await db_save_turn(
                    tenant_id=self.tenant_id,
                    channel="voice",
                    user_text=transcript,
                    assistant_text=answer_text,
                    sources=[s.model_dump() for s in sources],
                    latency_ms=total_ms,
                )
            except Exception:
                logger.exception("Failed to persist voice turn")

        asyncio.create_task(_persist())
        await self.send_json(
            {
                "type": "latency",
                "speech_end_to_retrieval_ms": sample.speech_end_to_retrieval_ms,
                "retrieval_ms": sample.retrieval_ms,
                "llm_first_token_ms": sample.llm_first_token_ms,
                "tts_first_audio_ms": sample.tts_first_audio_ms,
                "total_ms": sample.total_ms,
                "chunks_retrieved": sample.chunks_retrieved,
                "source_pages": sample.source_pages,
            }
        )
        await self.send_json({"type": "status", "state": "listening"})


async def voice_websocket_loop(websocket: WebSocket, tenant_id: str) -> None:
    session = VoiceSession(websocket, tenant_id)
    stt = get_stt_provider()
    await session.send_json(
        {
            "type": "ready",
            "stt": stt.name,
            "tts": get_tts_provider().name,
            "state": "listening",
            "prefer_client_stt": stt.name == "client-speech",
        }
    )

    # Warm edge-tts / network path so the first reply is faster
    async def _warm_tts() -> None:
        try:
            from app.services.voice.tts import synthesize_with_fallback

            await synthesize_with_fallback("Bonjour.")
        except Exception:
            logger.debug("TTS warm-up skipped", exc_info=True)

    asyncio.create_task(_warm_tts())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                session.audio_buffer.extend(message["bytes"])
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                msg_type = payload.get("type")
                if msg_type == "user_transcript":
                    await session.handle_user_text(str(payload.get("text", "")))
                elif msg_type == "audio_end":
                    audio = bytes(session.audio_buffer)
                    session.audio_buffer.clear()
                    if audio:
                        await session.handle_utterance(audio)
                elif msg_type == "interrupt":
                    await session.interrupt()
                elif msg_type == "ping":
                    await session.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("Voice WS disconnected for tenant %s", tenant_id)
    except Exception as exc:
        logger.exception("Voice WS error for tenant %s", tenant_id)
        try:
            message = str(exc)
            if "ANTHROPIC_API_KEY" in message:
                message = (
                    "Clé Anthropic manquante ou non chargée. "
                    "Vérifiez backend/.env puis redémarrez l'API."
                )
            await session.send_json({"type": "error", "message": message})
        except Exception:
            pass
