from __future__ import annotations

import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.services.voice.pcm_endpoint import (
    PcmEndpoint,
    PcmTurnController,
    TurnCommit,
    float_to_wav16k,
    s16le_to_float32,
)
from app.services.voice.turn_detect import EnergyVad, SAMPLE_RATE, TurnScore, VAD_FRAME
from app.services.voice.whisper_features import N_SAMPLES, compute_whisper_log_mel_features


def _tone(samples: int, amp: float = 0.3, freq: float = 180.0) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t / SAMPLE_RATE)).astype(np.float32)


def _s16(pcm: np.ndarray) -> bytes:
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _frames(ms: float, amp: float = 0.3) -> bytes:
    n = int(round(ms * SAMPLE_RATE / 1000.0))
    n = max(VAD_FRAME, n - (n % VAD_FRAME))
    return _s16(_tone(n, amp=amp) if amp else np.zeros(n, dtype=np.float32))


class ScriptedPredictor:
    def __init__(self, probs: list[float], delay_s: float = 0.0) -> None:
        self.probs = list(probs)
        self.delay_s = delay_s
        self.calls = 0
        self.sizes: list[int] = []

    def __call__(self, pcm: np.ndarray) -> TurnScore:
        self.calls += 1
        self.sizes.append(int(np.asarray(pcm).size))
        if self.delay_s:
            time.sleep(self.delay_s)
        p = self.probs.pop(0) if self.probs else 0.9
        return TurnScore(probability=p, elapsed_ms=self.delay_s * 1000, complete=p > 0.5)


class PcmEndpointTests(unittest.TestCase):
    def test_log_mel_shape_is_whisper_8s(self):
        feats = compute_whisper_log_mel_features(np.zeros(1000, dtype=np.float32))
        self.assertEqual(feats.shape, (80, 800))
        self.assertEqual(N_SAMPLES, 128000)

    def test_pause_eval_after_about_280ms(self):
        ep = PcmEndpoint(EnergyVad(), pause_ms=280, extend_ms=720, fallback_ms=1400)
        speech = _tone(15 * VAD_FRAME)
        silence = np.zeros(9 * VAD_FRAME, dtype=np.float32)
        events = ep.ingest(speech)
        self.assertFalse(any(e.kind == "pause_eval" for e in events))
        events = ep.ingest(silence)
        kinds = [e.kind for e in events]
        self.assertIn("pause_eval", kinds)
        pause = next(e.pause_ms for e in events if e.kind == "pause_eval")
        self.assertGreaterEqual(pause, 256)
        self.assertLess(pause, 360)

    def test_speech_resume_bumps_generation_and_keeps_audio(self):
        ep = PcmEndpoint(EnergyVad(), pause_ms=280)
        ep.ingest(_tone(15 * VAD_FRAME))
        ep.ingest(np.zeros(9 * VAD_FRAME, dtype=np.float32))
        gen = ep.generation
        before = ep.turn_pcm().size
        events = ep.ingest(_tone(15 * VAD_FRAME))
        self.assertTrue(any(e.kind == "speech_resume" for e in events))
        self.assertGreater(ep.generation, gen)
        self.assertGreater(ep.turn_pcm().size, before)

    def test_wav_roundtrip(self):
        pcm = _tone(1600)
        wav = float_to_wav16k(pcm)
        back = s16le_to_float32(_s16(pcm))
        self.assertEqual(back.size, 1600)
        self.assertGreater(len(wav), 44)

    def test_silero_onnx_separates_voiced_from_silence(self):
        from pathlib import Path
        from app.services.voice.turn_detect import SileroVadOnnx, _silero_onnx_path

        path = _silero_onnx_path()
        if path is None or not Path(path).exists():
            self.skipTest("silero-vad ONNX not installed")
        vad = SileroVadOnnx(path)
        t = np.arange(8 * VAD_FRAME, dtype=np.float32)
        voiced = (
            0.4 * np.sin(2 * np.pi * 120 * t / SAMPLE_RATE)
            + 0.2 * np.sin(2 * np.pi * 240 * t / SAMPLE_RATE)
        ).astype(np.float32)
        speech_p = [vad.prob(voiced[i : i + VAD_FRAME]) for i in range(0, voiced.size, VAD_FRAME)]
        vad.reset()
        silence_p = [vad.prob(np.zeros(VAD_FRAME, dtype=np.float32)) for _ in range(8)]
        self.assertGreater(float(np.mean(speech_p)), 0.5)
        self.assertLess(float(np.mean(silence_p)), 0.2)


class PcmTurnControllerTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, predictor, **kwargs) -> tuple[PcmTurnController, list[TurnCommit]]:
        commits: list[TurnCommit] = []

        async def on_commit(commit: TurnCommit) -> None:
            commits.append(commit)

        ctrl = PcmTurnController(EnergyVad(), predictor, on_commit, **kwargs)
        return ctrl, commits

    async def test_complete_question_commits_after_short_pause(self):
        predictor = ScriptedPredictor([0.91])
        ctrl, commits = await self._collect(predictor)
        ctrl.feed(_frames(500))
        ctrl.feed(_frames(320, amp=0.0))
        await asyncio.sleep(0.05)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].reason, "complete")
        self.assertGreaterEqual(commits[0].pause_ms, 256)
        self.assertLess(commits[0].endpointing_ms, 400)
        ctrl.close()

    async def test_incomplete_waits_and_keeps_turn_on_resume(self):
        predictor = ScriptedPredictor([0.2, 0.93])
        ctrl, commits = await self._collect(predictor)
        ctrl.feed(_frames(500))
        ctrl.feed(_frames(320, amp=0.0))
        await asyncio.sleep(0.05)
        self.assertEqual(commits, [])
        self.assertEqual(predictor.calls, 1)
        gen = ctrl.endpoint.generation
        ctrl.feed(_frames(400))
        self.assertGreater(ctrl.endpoint.generation, gen)
        await asyncio.sleep(0.02)
        self.assertEqual(commits, [])
        ctrl.feed(_frames(320, amp=0.0))
        await asyncio.sleep(0.05)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].reason, "complete")
        self.assertGreater(commits[0].pcm.size, int(0.7 * SAMPLE_RATE))
        ctrl.close()

    async def test_stale_complete_is_not_committed(self):
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def predict(pcm: np.ndarray) -> TurnScore:
            loop.call_soon_threadsafe(started.set)
            time.sleep(0.12)
            return TurnScore(probability=0.99, elapsed_ms=120, complete=True)

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stale-turn")
        ctrl, commits = await self._collect(predict, executor=pool)
        try:
            ctrl.feed(_frames(500))
            ctrl.feed(_frames(320, amp=0.0))
            await asyncio.wait_for(started.wait(), 1)
            ctrl.feed(_frames(400))
            await asyncio.sleep(0.2)
            self.assertEqual(commits, [])
            started.clear()
            ctrl.feed(_frames(320, amp=0.0))
            await asyncio.sleep(0.25)
            self.assertEqual(len(commits), 1)
        finally:
            ctrl.close()
            pool.shutdown(wait=False, cancel_futures=True)

    async def test_fallback_commits_when_confidence_stays_low(self):
        predictor = ScriptedPredictor([0.1, 0.1, 0.1])
        ctrl, commits = await self._collect(predictor, pause_ms=280, extend_ms=400, fallback_ms=640)
        ctrl.feed(_frames(400))
        ctrl.feed(_frames(700, amp=0.0))
        await asyncio.sleep(0.08)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].reason, "fallback")
        ctrl.close()


class VoiceSessionPcmTests(unittest.IsolatedAsyncioTestCase):
    async def test_pcm_commit_invokes_utterance_and_ignores_audio_end(self):
        from unittest.mock import AsyncMock
        import json

        from starlette.websockets import WebSocketState
        from app.services.voice.session import VoiceSession

        sent: list[dict] = []

        class Ws:
            client_state = WebSocketState.CONNECTED

            async def send_text(self, text: str) -> None:
                sent.append(json.loads(text))

        session = VoiceSession(Ws(), "tenant")
        predictor = ScriptedPredictor([0.94])
        commits: list[TurnCommit] = []

        async def on_commit(commit: TurnCommit) -> None:
            commits.append(commit)
            await session._on_pcm_commit(commit)

        session.smart_turn_enabled = True
        session.pcm_controller = PcmTurnController(EnergyVad(), predictor, on_commit)
        handle = AsyncMock()
        session.handle_utterance = handle
        session.feed_pcm(_frames(500))
        session.feed_pcm(_frames(320, amp=0.0))
        await asyncio.sleep(0.08)
        self.assertEqual(len(commits), 1)
        handle.assert_awaited()
        self.assertTrue(any(m.get("type") == "turn_commit" for m in sent))
        session.pcm_controller.close()

    async def test_ready_advertises_pcm_when_detector_loads(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        import json

        from app.services.voice.session import VoiceSession, voice_websocket_loop
        from app.services.voice.turn_detect import TurnDetector, EnergyVad
        from starlette.websockets import WebSocketState

        messages: list[dict] = []

        class Socket:
            client_state = WebSocketState.CONNECTED

            async def send_text(self, text):
                messages.append(json.loads(text))

            async def receive(self):
                return {"type": "websocket.disconnect"}

        detector = TurnDetector(vad=EnergyVad(), predict_sync=lambda pcm: TurnScore(0.1, 1.0, False))
        settings = SimpleNamespace(
            voice_smart_turn=True,
            voice_smart_turn_pause_ms=280,
            voice_smart_turn_extend_ms=720,
            voice_smart_turn_fallback_ms=1400,
            llm_provider="gemini",
            active_llm_model="gemini-3.5-flash-lite",
            whisper_keepalive=True,
        )
        with (
            patch("app.services.voice.session.wait_voice_runtime", AsyncMock()),
            patch("app.services.voice.session.start_stt_keepalive"),
            patch("app.services.voice.session.stop_stt_keepalive"),
            patch("app.services.voice.session.claude_service.warm_connection", AsyncMock()),
            patch("app.services.voice.session.describe_server_stt", return_value={"stt": "faster-whisper", "stt_device": "cuda", "stt_model": "base"}),
            patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(name="edge", prime=AsyncMock())),
            patch("app.services.voice.session.rag_service.store.preload", AsyncMock()),
            patch("app.services.voice.session.get_settings", return_value=settings),
            patch("app.services.voice.session.get_turn_detector", return_value=detector),
        ):
            await asyncio.wait_for(voice_websocket_loop(Socket(), "test"), 2)
        ready = next(m for m in messages if m["type"] == "ready")
        self.assertTrue(ready["smart_turn"])
        self.assertEqual(ready["pcm_rate"], 16000)
        self.assertEqual(ready["pcm_format"], "s16le")


if __name__ == "__main__":
    unittest.main()
