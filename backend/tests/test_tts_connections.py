import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.voice.tts import EdgeTtsProvider
from app.services.voice.session import VoiceSession


class FakeProvider(EdgeTtsProvider):
    def __init__(self):
        super().__init__()
        self.opened = []
        self.reads = 0
        self.empty_first = False
        self.fail_after_audio = False

    async def _open_ws(self):
        ws = SimpleNamespace(closed=False)
        self.opened.append(ws)
        return ws

    async def _close_ws(self, ws):
        ws.closed = True

    async def _send_ssml(self, ws, text):
        pass

    async def _iter_audio(self, ws):
        self.reads += 1
        if self.empty_first and self.reads == 1:
            return
        yield b"first"
        if self.fail_after_audio:
            raise RuntimeError("disconnected")
        yield b"last"


class TtsConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_prime_refreshes_idle_socket_but_reuses_fresh_socket(self):
        p = FakeProvider()
        await p.prime()
        await p.prime()
        self.assertEqual(len(p.opened), 1)
        p.max_idle_seconds = 5
        p._ws_opened_at = time.monotonic() - 6
        await p.prime()
        self.assertEqual(len(p.opened), 2)
        self.assertTrue(p.opened[0].closed)
        await p.close()
        self.assertTrue(p.opened[1].closed)

    async def test_cached_phrase_avoids_provider_request(self):
        p = FakeProvider()
        self.assertEqual(await p.synthesize("Bonjour."), b"firstlast")
        if p._prime_task is not None:
            await p._prime_task
        reads = p.reads
        self.assertEqual(await p.synthesize("Bonjour."), b"firstlast")
        self.assertEqual(p.reads, reads)
        await p.close()
        self.assertFalse(p._audio_cache)

    async def test_empty_idle_socket_is_retried_once(self):
        p = FakeProvider()
        p.empty_first = True
        self.assertEqual(await p.synthesize("Bonjour."), b"firstlast")
        self.assertEqual(p.reads, 2)
        await p.close()

    async def test_partial_failure_is_never_retried_or_cached(self):
        p = FakeProvider()
        p.fail_after_audio = True
        with self.assertRaises(RuntimeError):
            await p.synthesize("Bonjour.")
        self.assertEqual(p.reads, 1)
        self.assertFalse(p._audio_cache)
        await p.close()

    async def test_cache_is_bounded(self):
        p = FakeProvider()
        for index in range(40):
            await p.synthesize(f"Phrase numéro {index}.")
        self.assertEqual(len(p._audio_cache), 32)
        self.assertEqual(p._cache_bytes, sum(len(data) for _, data in p._audio_cache.values()))
        await p.close()

    async def test_used_socket_is_closed_and_a_spare_is_primed(self):
        p = FakeProvider()
        await p.prime()
        first = p._ws
        await p.synthesize("Bonjour.")
        self.assertTrue(first.closed)
        if p._prime_task is not None:
            await p._prime_task
        spare = p._live_socket()
        self.assertIsNotNone(spare)
        self.assertIsNot(spare, first)
        await p.close()

    async def test_take_waits_for_in_flight_prime_instead_of_opening_twice(self):
        p = FakeProvider()
        gate = asyncio.Event()
        inflight = 0
        max_inflight = 0

        async def delayed_open():
            nonlocal inflight, max_inflight
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            await gate.wait()
            inflight -= 1
            return await FakeProvider._open_ws(p)

        p._open_ws = delayed_open
        prime_task = asyncio.create_task(p.prime())
        await asyncio.sleep(0.02)
        take_task = asyncio.create_task(p._take_socket())
        await asyncio.sleep(0.02)
        self.assertEqual(max_inflight, 1)
        gate.set()
        await take_task
        await prime_task
        self.assertEqual(max_inflight, 1)
        await p.close()

    async def test_connection_preparation_overlaps_audio_processing(self):
        started = asyncio.Event()
        async def prime():
            started.set()
            await asyncio.Event().wait()
        async def process(audio):
            await asyncio.wait_for(started.wait(), 1)
        session = VoiceSession(None, "test")
        session._respond_audio = AsyncMock(side_effect=process)
        with (
            patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(prime=prime)),
            patch("app.services.voice.session.claude_service.warm_connection", AsyncMock()),
        ):
            await session.handle_utterance(b"audio")
        session._respond_audio.assert_awaited_once()
