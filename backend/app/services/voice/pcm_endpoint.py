"""Bounded 16 kHz PCM buffer and VAD pause → Smart Turn state machine."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from app.services.voice.turn_detect import SAMPLE_RATE, TurnScore, VadModel, VAD_FRAME

PRE_ROLL_SAMPLES = int(0.2 * SAMPLE_RATE)
MAX_SAMPLES = 12 * SAMPLE_RATE
SPEECH_PROB = 0.5
MIN_SPEECH_FRAMES = 2


@dataclass(frozen=True)
class EndpointEvent:
    kind: str
    generation: int
    pause_ms: float


@dataclass(frozen=True)
class TurnCommit:
    pcm: np.ndarray
    endpointing_ms: float
    pause_ms: float
    probability: float | None
    smart_turn_ms: float
    extra_wait_ms: float
    generation: int
    reason: str


def s16le_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    leftover = len(data) % 2
    if leftover:
        data = data[: len(data) - leftover]
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_wav16k(pcm: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    s16 = (samples * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(s16.tobytes())
    return buf.getvalue()


class PcmEndpoint:
    def __init__(
        self,
        vad: VadModel,
        *,
        pause_ms: int = 280,
        extend_ms: int = 720,
        fallback_ms: int = 1400,
        max_samples: int = MAX_SAMPLES,
    ) -> None:
        self.vad = vad
        self.pause_ms = pause_ms
        self.extend_ms = extend_ms
        self.fallback_ms = fallback_ms
        self.max_samples = max_samples
        self.generation = 0
        self._chunks: list[np.ndarray] = []
        self._n = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_started = False
        self._in_speech = False
        self._speech_frames = 0
        self._pause_ms = 0.0
        self._speech_start = 0
        self._last_speech = 0
        self._emitted_pause = False
        self._emitted_extend = False
        self._emitted_fallback = False
        self._committed = False

    def reset(self) -> None:
        self.generation += 1
        self._chunks.clear()
        self._n = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_started = False
        self._in_speech = False
        self._speech_frames = 0
        self._pause_ms = 0.0
        self._speech_start = 0
        self._last_speech = 0
        self._emitted_pause = False
        self._emitted_extend = False
        self._emitted_fallback = False
        self._committed = False
        self.vad.reset()

    def turn_pcm(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks)
        start = max(0, self._speech_start - PRE_ROLL_SAMPLES) if self._speech_started else 0
        return audio[start:]

    def ingest(self, pcm: np.ndarray) -> list[EndpointEvent]:
        if self._committed:
            return []
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []
        if self._pending.size:
            x = np.concatenate([self._pending, x])
        events: list[EndpointEvent] = []
        offset = 0
        while offset + VAD_FRAME <= x.size:
            frame = x[offset : offset + VAD_FRAME]
            events.extend(self._ingest_frame(frame))
            offset += VAD_FRAME
        self._pending = x[offset:]
        return events

    def _trim(self) -> None:
        if self._n <= self.max_samples:
            return
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        keep_from = max(0, self._n - self.max_samples)
        audio = audio[keep_from:]
        self._chunks = [audio] if audio.size else []
        self._n = audio.size
        self._speech_start = max(0, self._speech_start - keep_from)
        self._last_speech = max(0, self._last_speech - keep_from)

    def _ingest_frame(self, frame: np.ndarray) -> list[EndpointEvent]:
        events: list[EndpointEvent] = []
        self._chunks.append(frame.copy())
        self._n += frame.size
        self._trim()
        speaking = self.vad.prob(frame) >= SPEECH_PROB
        frame_ms = 1000.0 * frame.size / SAMPLE_RATE
        if speaking:
            self._speech_frames += 1
            if not self._speech_started:
                if self._speech_frames >= MIN_SPEECH_FRAMES:
                    self._speech_started = True
                    self._speech_start = max(0, self._n - MIN_SPEECH_FRAMES * VAD_FRAME)
                    self._in_speech = True
            elif not self._in_speech:
                if self._emitted_pause or self._emitted_extend or self._emitted_fallback:
                    self.generation += 1
                    events.append(EndpointEvent("speech_resume", self.generation, self._pause_ms))
                self._in_speech = True
                self._emitted_pause = False
                self._emitted_extend = False
                self._emitted_fallback = False
            self._pause_ms = 0.0
            self._last_speech = self._n
        else:
            self._speech_frames = 0
            if self._speech_started:
                self._in_speech = False
                self._pause_ms += frame_ms
                if self._n >= self.max_samples:
                    events.append(EndpointEvent("max_duration", self.generation, self._pause_ms))
                    return events
                if not self._emitted_pause and self._pause_ms >= self.pause_ms:
                    self._emitted_pause = True
                    events.append(EndpointEvent("pause_eval", self.generation, self._pause_ms))
                elif self._emitted_pause and not self._emitted_extend and self._pause_ms >= self.extend_ms:
                    self._emitted_extend = True
                    events.append(EndpointEvent("reeval", self.generation, self._pause_ms))
                elif self._emitted_pause and not self._emitted_fallback and self._pause_ms >= self.fallback_ms:
                    self._emitted_fallback = True
                    events.append(EndpointEvent("fallback", self.generation, self._pause_ms))
        if self._speech_started and self._n >= self.max_samples and self._in_speech:
            events.append(EndpointEvent("max_duration", self.generation, self._pause_ms))
        return events


class PcmTurnController:
    def __init__(
        self,
        vad: VadModel,
        predict: Callable[[np.ndarray], TurnScore],
        on_commit: Callable[[TurnCommit], Awaitable[None]],
        *,
        pause_ms: int = 280,
        extend_ms: int = 720,
        fallback_ms: int = 1400,
        executor=None,
        max_samples: int = MAX_SAMPLES,
    ) -> None:
        self.endpoint = PcmEndpoint(
            vad, pause_ms=pause_ms, extend_ms=extend_ms, fallback_ms=fallback_ms, max_samples=max_samples
        )
        self._predict = predict
        self._on_commit = on_commit
        self._executor = executor
        self._inflight: dict[int, asyncio.Task] = {}
        self._commit_lock = asyncio.Lock()
        self._closed = False
        self.last_overhead_ms = 0.0

    def reset(self) -> None:
        self.endpoint.reset()
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()

    def close(self) -> None:
        self._closed = True
        self.reset()

    def feed(self, data: bytes) -> None:
        if self._closed or self.endpoint._committed:
            return
        events = self.endpoint.ingest(s16le_to_float32(data))
        for event in events:
            self._handle(event)

    def _handle(self, event: EndpointEvent) -> None:
        if event.kind == "speech_resume":
            for gen, task in list(self._inflight.items()):
                if gen != event.generation:
                    task.cancel()
                    self._inflight.pop(gen, None)
            return
        if event.kind in ("pause_eval", "reeval"):
            self._start_eval(event)
            return
        if event.kind in ("fallback", "max_duration"):
            asyncio.create_task(self._commit_now(event, probability=None, smart_turn_ms=0.0, reason=event.kind))

    def _start_eval(self, event: EndpointEvent) -> None:
        gen = event.generation
        if gen in self._inflight:
            return
        pcm = self.endpoint.turn_pcm()
        pause_ms = event.pause_ms
        task = asyncio.create_task(self._run_eval(gen, pcm, pause_ms))
        self._inflight[gen] = task
        task.add_done_callback(lambda done: self._inflight.pop(gen, None) if self._inflight.get(gen) is done else None)

    async def _run_eval(self, generation: int, pcm: np.ndarray, pause_ms: float) -> None:
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        try:
            if self._executor is not None:
                score = await loop.run_in_executor(self._executor, self._predict, pcm)
            else:
                score = self._predict(pcm)
        except asyncio.CancelledError:
            return
        except Exception:
            score = TurnScore(probability=0.0, elapsed_ms=(time.perf_counter() - started) * 1000, complete=False)
        if generation != self.endpoint.generation or self.endpoint._committed or self._closed:
            return
        self.last_overhead_ms = score.elapsed_ms
        extra = max(0.0, pause_ms - self.endpoint.pause_ms)
        if score.complete:
            await self._commit_now(
                EndpointEvent("pause_eval", generation, pause_ms),
                probability=score.probability,
                smart_turn_ms=score.elapsed_ms,
                reason="complete",
                extra_wait_ms=extra,
            )

    async def _commit_now(
        self,
        event: EndpointEvent,
        *,
        probability: float | None,
        smart_turn_ms: float,
        reason: str,
        extra_wait_ms: float | None = None,
    ) -> None:
        async with self._commit_lock:
            if self._closed or self.endpoint._committed:
                return
            if event.generation != self.endpoint.generation:
                return
            pcm = self.endpoint.turn_pcm()
            if pcm.size < SAMPLE_RATE * 0.2:
                return
            endpointing_ms = 1000.0 * max(0, self.endpoint._n - self.endpoint._last_speech) / SAMPLE_RATE
            extra = extra_wait_ms if extra_wait_ms is not None else max(0.0, event.pause_ms - self.endpoint.pause_ms)
            commit = TurnCommit(
                pcm=pcm,
                endpointing_ms=endpointing_ms,
                pause_ms=event.pause_ms,
                probability=probability,
                smart_turn_ms=smart_turn_ms,
                extra_wait_ms=extra,
                generation=event.generation,
                reason=reason,
            )
            self.endpoint._committed = True
            await self._on_commit(commit)
            self.endpoint.reset()
