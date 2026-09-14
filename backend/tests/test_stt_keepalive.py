import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from app.services.voice import stt
from app.services.voice.session import VoiceSession
from app.services.voice.stt import FasterWhisperStt, LazyWhisperStt, submit_gpu_pulse


class SttKeepaliveTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        stt._user_stt.clear()
        stt._user_stt_waiters = 0
        stt._pulse_queued = False
        stt._keep_alive = False

    def _provider(self) -> FasterWhisperStt:
        provider = FasterWhisperStt.__new__(FasterWhisperStt)
        provider._slot = asyncio.Semaphore(1)
        provider.last_timings = {}
        return provider

    async def test_overlapping_transcribe_waits_instead_of_failing(self):
        provider = self._provider()
        started = threading.Event()
        release = threading.Event()

        def fake_sync(audio_bytes: bytes) -> str:
            started.set()
            self.assertTrue(release.wait(2), "in-flight STT was not released")
            return audio_bytes.decode()

        provider._transcribe_sync = fake_sync  # type: ignore[method-assign]
        first = asyncio.create_task(provider.transcribe(b"one"))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        second = asyncio.create_task(provider.transcribe(b"two"))
        await asyncio.sleep(0.6)
        self.assertFalse(second.done(), "second STT failed instead of waiting for the slot")
        release.set()
        self.assertEqual(await first, "one")
        self.assertEqual(await second, "two")
        self.assertFalse(provider._slot.locked())
        self.assertFalse(stt._user_stt.is_set())

    async def test_transcribe_completes_on_executor_future(self):
        provider = self._provider()
        provider._transcribe_sync = lambda audio_bytes: "bonjour"  # type: ignore[method-assign]
        self.assertEqual(await provider.transcribe(b"clip"), "bonjour")
        self.assertEqual(provider.last_timings["stt_queue_ms"], 0)
        self.assertFalse(provider._slot.locked())

    async def test_cancel_waits_for_worker_before_returning(self):
        provider = self._provider()
        started = threading.Event()
        release = threading.Event()

        def fake_sync(audio_bytes: bytes) -> str:
            started.set()
            self.assertTrue(release.wait(2), "cancelled STT worker was abandoned")
            return "ok"

        provider._transcribe_sync = fake_sync  # type: ignore[method-assign]
        task = asyncio.create_task(provider.transcribe(b"one"))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        task.cancel()
        await asyncio.sleep(0.05)
        self.assertFalse(task.done(), "cancelled STT returned while Whisper still held the slot")
        self.assertTrue(provider._slot.locked())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(provider._slot.locked())
        self.assertFalse(stt._user_stt.is_set())

    def test_pulse_is_skipped_while_user_stt_is_active(self):
        stt._keep_alive = True
        stt._begin_user_stt()
        try:
            with patch.object(stt, "_stt_executor") as executor:
                self.assertFalse(submit_gpu_pulse(SimpleNamespace(device="cuda", _last_infer_at=0)))
                executor.submit.assert_not_called()
        finally:
            stt._end_user_stt()

    def test_pulse_submit_does_not_block_on_result(self):
        stt._keep_alive = True
        submitted = MagicMock()
        with patch.object(stt, "_stt_executor") as executor:
            executor.submit.return_value = submitted
            self.assertTrue(submit_gpu_pulse(SimpleNamespace(device="cuda", _last_infer_at=0.0)))
            executor.submit.assert_called_once()
            submitted.result.assert_not_called()

    def test_second_pulse_is_not_queued(self):
        stt._keep_alive = True
        stt._pulse_queued = True
        with patch.object(stt, "_stt_executor") as executor:
            self.assertFalse(submit_gpu_pulse(SimpleNamespace(device="cuda", _last_infer_at=0.0)))
            executor.submit.assert_not_called()

    async def test_lazy_whisper_skips_executor_when_model_is_loaded(self):
        inner = SimpleNamespace(last_timings={"stt_infer_ms": 21}, transcribe=AsyncMock(return_value="bonjour"))
        with patch.object(stt, "_load_whisper") as loaded, patch.object(stt, "_stt_executor") as executor:
            loaded.cache_info.return_value = SimpleNamespace(currsize=1)
            loaded.return_value = inner
            lazy = LazyWhisperStt()
            self.assertEqual(await lazy.transcribe(b"audio"), "bonjour")
            inner.transcribe.assert_awaited_once()
            executor.submit.assert_not_called()
        self.assertEqual(lazy.last_timings["stt_infer_ms"], 21)

    def test_transcribe_sync_keeps_partial_timings_when_infer_fails(self):
        provider = FasterWhisperStt.__new__(FasterWhisperStt)
        provider.device = "cpu"
        provider.startup_warmed = True
        provider._infer_calls = 0
        provider._last_infer_at = 0.0
        provider._keep_awake = False
        provider.model_name = "base"
        pcm = np.zeros(1600, dtype=np.float32)
        with patch("app.services.voice.stt._decode_pcm16k_timed", return_value=(pcm, {"stt_decode_codec_ms": 4})), patch.object(
            provider, "_infer_pcm", side_effect=RuntimeError("infer exploded")
        ):
            with self.assertRaisesRegex(RuntimeError, "infer exploded"):
                provider._transcribe_sync(b"clip")
        self.assertEqual(provider.last_timings["stt_decode_codec_ms"], 4)
        self.assertEqual(provider.last_timings["stt_samples"], 1600)
        self.assertEqual(provider.last_timings["stt_infer_n"], 0)

    async def test_stt_exception_is_copied_into_debug_timing(self):
        session = VoiceSession(None, "test")
        session.send_json = AsyncMock()
        provider = SimpleNamespace(
            transcribe=AsyncMock(side_effect=RuntimeError("decode failed: invalid webm")),
            last_timings={"stt_decode_ms": 12},
        )
        with (
            patch("app.services.voice.session.get_audio_stt_provider", return_value=provider),
            patch("app.services.voice.session.describe_server_stt", return_value={"stt": "faster-whisper"}),
            patch.object(VoiceSession, "_prime_tts", AsyncMock()),
            patch.object(VoiceSession, "_warm_ai", AsyncMock()),
        ):
            await session.handle_utterance(b"audio")
        payloads = [call.args[0] for call in session.send_json.call_args_list]
        debug = next(item for item in payloads if item.get("type") == "debug_timing" and item.get("stage") == "stt")
        self.assertTrue(debug["failed"])
        self.assertEqual(debug["stt_error"], "RuntimeError: decode failed: invalid webm")
        self.assertEqual(debug["stt_error_type"], "RuntimeError")
        self.assertEqual(debug["stt_decode_ms"], 12)
        error = next(item for item in payloads if item.get("type") == "error")
        self.assertEqual(error["message"], "RuntimeError: decode failed: invalid webm")

    async def test_slot_released_exactly_once_on_success(self):
        provider = self._provider()
        provider._transcribe_sync = lambda audio_bytes: "ok"  # type: ignore[method-assign]
        original = provider._slot.release
        count = {"n": 0}

        def release() -> None:
            count["n"] += 1
            original()

        provider._slot.release = release  # type: ignore[method-assign]
        self.assertEqual(await provider.transcribe(b"x"), "ok")
        self.assertEqual(count["n"], 1)
        self.assertFalse(provider._slot.locked())

    async def test_slot_released_exactly_once_when_executor_submit_fails(self):
        provider = self._provider()
        loop = asyncio.get_running_loop()
        original = provider._slot.release
        count = {"n": 0}

        def release() -> None:
            count["n"] += 1
            original()

        provider._slot.release = release  # type: ignore[method-assign]
        with patch.object(loop, "run_in_executor", side_effect=RuntimeError("executor closed")):
            with self.assertRaisesRegex(RuntimeError, "executor closed"):
                await provider.transcribe(b"x")
        self.assertEqual(count["n"], 1)
        self.assertFalse(provider._slot.locked())
        self.assertFalse(stt._user_stt.is_set())

    async def test_cancel_releases_slot_once_after_worker_finishes(self):
        provider = self._provider()
        started = threading.Event()
        release = threading.Event()
        original = provider._slot.release
        count = {"n": 0}

        def counted_release() -> None:
            count["n"] += 1
            original()

        def fake_sync(audio_bytes: bytes) -> str:
            started.set()
            self.assertTrue(release.wait(2), "cancelled STT worker was abandoned")
            return "ok"

        provider._slot.release = counted_release  # type: ignore[method-assign]
        provider._transcribe_sync = fake_sync  # type: ignore[method-assign]
        task = asyncio.create_task(provider.transcribe(b"one"))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        task.cancel()
        await asyncio.sleep(0.05)
        self.assertEqual(count["n"], 0)
        self.assertTrue(provider._slot.locked())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(count["n"], 1)
        self.assertFalse(provider._slot.locked())

    def test_transcribe_has_no_half_second_slot_timeout(self):
        import inspect

        source = inspect.getsource(FasterWhisperStt.transcribe)
        self.assertNotIn("timeout=0.5", source)
        self.assertNotIn("wait_for", source)


if __name__ == "__main__":
    unittest.main()
