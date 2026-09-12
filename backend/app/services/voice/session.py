from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
import math
from contextlib import asynccontextmanager
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.config import get_settings
from app.core.logging_metrics import LatencySample, latency_store
from app.services.llm.claude import claude_service
from app.services.rag.service import SYSTEM_PROMPT, rag_service
from app.services.voice.stt import get_audio_stt_provider, get_stt_provider
from app.services.voice.tts import get_tts_provider

logger = logging.getLogger(__name__)


class PartialSpeechError(RuntimeError):
    """Some audio was already sent; replaying the whole answer would repeat it."""

_SENTENCE_END = re.compile(r"^(.+?[.!?…])(?:\s+|$)")


def _speech_prefix(buffer: str) -> str | None:
    """Flush the first speakable clause as soon as it is stable."""
    for match in re.finditer(r"[.!?…;:,](?=\s)", buffer):
        prefix = buffer[:match.end()].strip()
        if len(prefix) >= 12 and len(prefix.split()) >= 3:
            return prefix
    words = buffer.split()
    if len(words) >= 6 and (buffer.endswith(" ") or len(words) > 6):
        return " ".join(words[:6])
    return None


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
        self.call_id = str(uuid.uuid4())
        self.turn_id = ""
        self.trace_id = ""
        self.reported_turns: set[str] = set()
        self.prefetch_text = ""
        self.prefetch_task: asyncio.Task | None = None

        self.audio_buffer = bytearray()
        self.cancel_event = asyncio.Event()
        self.busy = False
        self._response_task: asyncio.Task | None = None
        # Multi-turn memory so guided procedures can verify step-by-step
        self.history: list[dict[str, str]] = []

    async def debug_timing(self, stage: str, started: float, t0: float, failed: bool = False) -> None:
        await self.send_json({
            "type": "debug_timing", "stage": stage,
            "offset_ms": max(0, (started - t0) * 1000),
            "duration_ms": max(0, (time.perf_counter() - started) * 1000),
            "failed": failed,
        })

    @asynccontextmanager
    async def debug_stage(self, stage: str, t0: float):
        started = time.perf_counter()
        failed = True
        try:
            yield
            failed = False
        finally:
            await self.debug_timing(stage, started, t0, failed)

    def prefetch(self, text: str) -> None:
        text = text.strip()[:2000]
        if len(text) < 8 or text == self.prefetch_text or self.busy:
            return
        if self.prefetch_task:
            self.prefetch_task.cancel()
        self.prefetch_text = text

        async def retrieve():
            async with asyncio.timeout(3):
                return await rag_service.retrieve(self.tenant_id, self._retrieval_query(text), top_k=4)

        self.prefetch_task = asyncio.create_task(retrieve())
        self.prefetch_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)

    def _retrieval_query(self, transcript: str) -> str:
        """Keep procedure context only for short confirmations, not topic changes."""
        from app.services.rag.query import is_followup, normalize_query

        text = normalize_query(transcript)
        if is_followup(text) and self.history:
            last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            if last_user:
                return f"{normalize_query(last_user)}\n{text}"
        return text

    def _claude_messages(self, transcript: str, context: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = list(self.history[-4:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Contexte KB:\n{context}\n\n"
                    f"Client: {transcript}\n\n"
                    "Réponds à l'oral, français. Commence directement par une phrase "
                    "utile de 6 à 12 mots, sans introduction. Au maximum deux phrases courtes. "
                    "Procédure: une étape + confirmation. Prix: cite le montant en DT. "
                    "N'escalade pas si un forfait ou un tarif est dans le contexte KB."
                ),
            }
        )
        return messages

    def _voice_context(self, chunks: list) -> str:
        """Compact context for faster Claude turns."""
        if not chunks:
            return "Aucune information pertinente."
        parts: list[str] = []
        for i, c in enumerate(chunks[:4], start=1):
            text = c.text.strip()
            if len(text) > 700:
                text = text[:700].rsplit(" ", 1)[0] + "…"
            parts.append(f"[{i}|p.{c.page}] {text}")
        return "\n".join(parts)

    async def send_json(self, payload: dict[str, Any]) -> None:
        payload = {"call_id": self.call_id, "turn_id": self.turn_id, "trace_id": self.trace_id, **payload}
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

    async def _synthesize_and_send(self, text: str, t0: float) -> float:
        phrases: asyncio.Queue[str | None] = asyncio.Queue()
        phrases.put_nowait(text)
        phrases.put_nowait(None)
        return await self._stream_phrases(phrases, t0)

    async def _stream_phrases(self, phrases: asyncio.Queue[str | None], t0: float) -> float:
        first_audio_ms = None
        tts_started = time.perf_counter()
        try:
            async with asyncio.timeout(15):
                while (text := await phrases.get()) is not None:
                    received = False
                    async for audio in get_tts_provider().synthesize_stream(text):
                        if self.cancel_event.is_set():
                            raise asyncio.CancelledError()
                        if not audio:
                            continue
                        received = True
                        if first_audio_ms is None:
                            first_audio_ms = (time.perf_counter() - t0) * 1000
                            await self.debug_timing("tts", tts_started, t0)
                            await self.send_json({"type": "status", "state": "speaking"})
                            await self.send_json({"type": "tts_start", "mime": "audio/mpeg", "server_first_audio_ms": first_audio_ms})
                        await self.ws.send_bytes(audio)
                    if not received:
                        raise RuntimeError("No audio for speech segment")
            if first_audio_ms is None:
                raise RuntimeError("No speech audio received")
            await self.send_json({"type": "tts_end"})
            return first_audio_ms
        except BaseException as exc:
            if first_audio_ms is None:
                await self.debug_timing("tts", tts_started, t0, failed=True)
            if first_audio_ms is not None:
                await self.send_json({"type": "tts_abort"})
                if isinstance(exc, Exception):
                    raise PartialSpeechError("La lecture a été interrompue. La réponse complète reste affichée.") from exc
            raise

    async def _prime_tts(self) -> None:
        prime = getattr(get_tts_provider(), "prime", None)
        if not callable(prime):
            return
        try:
            await prime()
        except Exception:
            logger.debug("TTS prime skipped", exc_info=True)

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
            async with asyncio.timeout(25):
                await self._response_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Voice response failed")
            message = str(exc) or "La réponse prend trop de temps. Veuillez réessayer."
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
            async with asyncio.timeout(25):
                await self._response_task
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            logger.exception("Voice audio processing timed out")
            await self.send_json(
                {"type": "error", "message": "La réponse prend trop de temps. Veuillez réessayer."}
            )
            await self.send_json({"type": "status", "state": "listening"})
        except ValueError as exc:
            logger.warning("Voice turn rejected: %s", exc)
            await self.send_json({"type": "error", "message": str(exc)})
            await self.send_json({"type": "status", "state": "listening"})
        except Exception:
            logger.exception("Voice audio processing failed")
            await self.send_json({"type": "error", "message": "Transcription audio impossible. Vérifiez le service vocal puis réessayez."})
            await self.send_json({"type": "status", "state": "listening"})
        finally:
            self.busy = False

    async def _respond_audio(self, audio: bytes) -> None:
        t0 = time.perf_counter()
        await self.send_json({"type": "status", "state": "thinking"})
        stt = get_audio_stt_provider()
        async with self.debug_stage("stt", t0), asyncio.timeout(12):
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
        async with self.debug_stage("retrieval", t0), asyncio.timeout(10):
            prefetched = self.prefetch_task
            self.prefetch_task = None
            matches = self.prefetch_text == transcript
            self.prefetch_text = ""
            if prefetched and matches and not prefetched.cancelled():
                try:
                    chunks, retrieval_ms = await prefetched
                except Exception:
                    chunks, retrieval_ms = await rag_service.retrieve(self.tenant_id, retrieve_q, top_k=4)
            else:
                if prefetched:
                    prefetched.cancel()
                chunks, retrieval_ms = await rag_service.retrieve(self.tenant_id, retrieve_q, top_k=4)
        sources = rag_service.to_sources(chunks)
        await self.send_json({"type": "sources", "items": [s.model_dump() for s in sources]})

        context = self._voice_context(chunks)
        claude_messages = self._claude_messages(transcript, context)

        await self.send_json({"type": "status", "state": "thinking"})
        full_answer: list[str] = []
        first_token_ms: float | None = None
        first_audio_ms: float | None = None
        pending_speech = ""
        phrases: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task: asyncio.Task[float] | None = None

        settings = get_settings()
        voice_model = settings.anthropic_voice_model or settings.anthropic_model
        llm_started = time.perf_counter()
        phrase_started = llm_started

        tts_prime = asyncio.create_task(self._prime_tts())
        try:
            async with claude_service.client.with_options(timeout=8.0, max_retries=0).messages.stream(
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
                        await self.debug_timing("llm_wait", llm_started, t0)
                        phrase_started = time.perf_counter()
                    full_answer.append(text)
                    pending_speech += text
                    # Split the opening once, then synthesize the remainder as
                    # one segment to avoid a provider request for every comma.
                    if tts_task is None and (candidate := _speech_prefix(pending_speech)) is not None:
                        phrases.put_nowait(candidate)
                        pending_speech = pending_speech.lstrip()[len(candidate):].lstrip()
                        await self.debug_timing("llm_phrase", phrase_started, t0)
                        tts_task = asyncio.create_task(self._stream_phrases(phrases, t0))
        except BaseException:
            tts_prime.cancel()
            await self.debug_timing("llm_wait" if first_token_ms is None else "llm_phrase", llm_started if first_token_ms is None else phrase_started, t0, failed=True)
            if tts_task is not None:
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            raise

        answer_text = "".join(full_answer).strip()
        if pending_speech.strip():
            phrases.put_nowait(pending_speech.strip())
        phrases.put_nowait(None)

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
            if not self.cancel_event.is_set():
                try:
                    if tts_task is not None:
                        first_audio_ms = await tts_task
                    else:
                        await self.debug_timing("llm_phrase", phrase_started, t0)
                        first_audio_ms = await self._stream_phrases(phrases, t0)
                except PartialSpeechError as exc:
                    await self.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    logger.exception("Server TTS failed — browser speech fallback")
                    if tts_task is not None and not tts_task.done():
                        tts_task.cancel()
                    await self.send_json({"type": "speak", "text": answer_text, "engine": "browser"})
        total_ms = (time.perf_counter() - t0) * 1000
        sample = LatencySample(
            tenant_id=self.tenant_id,
            path="voice",
            call_id=self.call_id,
            turn_id=self.turn_id,
            trace_id=self.trace_id,
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
            await get_tts_provider().prime()
        except Exception:
            logger.debug("TTS warm-up skipped", exc_info=True)

    warm_tts_task = asyncio.create_task(_warm_tts())
    async def _preload_context() -> None:
        try:
            await rag_service.store.preload(tenant_id)
        except Exception:
            logger.debug("Context preload skipped", exc_info=True)

    preload_task = asyncio.create_task(_preload_context())
    turn_task: asyncio.Task | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                if len(session.audio_buffer) + len(message["bytes"]) > 2_000_000:
                    await websocket.close(code=1009)
                    break
                session.audio_buffer.extend(message["bytes"])
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                msg_type = payload.get("type")
                if msg_type == "user_partial":
                    session.prefetch(str(payload.get("text", "")))
                elif msg_type in ("user_transcript", "audio_end"):
                    if turn_task and not turn_task.done():
                        await session.interrupt()
                        await turn_task
                    session.turn_id = str(payload.get("turn_id") or uuid.uuid4())[:80]
                    session.trace_id = str(uuid.uuid4())
                    if msg_type == "user_transcript":
                        turn_task = asyncio.create_task(session.handle_user_text(str(payload.get("text", ""))))
                    else:
                        audio = bytes(session.audio_buffer)
                        session.audio_buffer.clear()
                        if audio:
                            turn_task = asyncio.create_task(session.handle_utterance(audio))
                elif msg_type == "playback_started":
                    turn_id = str(payload.get("turn_id", ""))
                    # Accept one bounded timing report for the active turn only.
                    if turn_id != session.turn_id or turn_id in session.reported_turns:
                        continue
                    try:
                        ttfa = float(payload["ttfa_ms"])
                        endpointing = float(payload["endpointing_ms"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not all(math.isfinite(v) and 0 <= v <= 120000 for v in (ttfa, endpointing)):
                        continue
                    session.reported_turns = {turn_id}
                    latency_store.add(LatencySample(
                        tenant_id=tenant_id, path="voice_playback", call_id=session.call_id,
                        turn_id=turn_id, trace_id=session.trace_id,
                        client_ttfa_ms=ttfa, endpointing_ms=endpointing,
                    ))
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
    finally:
        if session.prefetch_task:
            session.prefetch_task.cancel()
            await asyncio.gather(session.prefetch_task, return_exceptions=True)
        warm_tts_task.cancel()
        preload_task.cancel()
        await asyncio.gather(warm_tts_task, preload_task, return_exceptions=True)
        if turn_task and not turn_task.done():
            turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)
