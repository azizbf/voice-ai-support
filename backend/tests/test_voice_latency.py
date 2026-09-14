import asyncio
import time
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.voice.session import (
    VoiceSession,
    voice_websocket_loop,
    _compact_chunk,
    _consume_speech_prefix,
    _speech_prefix,
    PartialSpeechError,
)
from starlette.websockets import WebSocketState


class VoiceLatencyTests(unittest.IsolatedAsyncioTestCase):
    def test_phrase_split_preserves_decimal_prices_and_waits_for_boundary(self):
        self.assertIsNone(_speech_prefix("Votre offre coûte 12,50"))
        self.assertEqual(_speech_prefix("Je vais vous aider tout de suite "), "Je vais vous aider")
        self.assertIsNone(_speech_prefix("Je vais vous"))
        self.assertEqual(_speech_prefix("Votre offre coûte 12,50 DT, sans engagement."), "Votre offre coûte 12,50 DT,")
        self.assertEqual(_speech_prefix("Votre offre coûte 12.50 DT."), "Votre offre coûte 12.50 DT.")
        self.assertEqual(_speech_prefix("Votre forfait coûte 49 DT."), "Votre forfait coûte 49 DT.")
        self.assertEqual(_speech_prefix("Votre offre coûte 12.50 DT. Ensuite"), "Votre offre coûte 12.50 DT.")
        self.assertIsNone(_speech_prefix("Souhaitez-vous continuer", opening=False))
        self.assertEqual(
            _speech_prefix("Souhaitez-vous continuer ?", opening=False),
            "Souhaitez-vous continuer ?",
        )
        self.assertIsNone(_speech_prefix("Le forfait Fibre 100 est "))
        self.assertEqual(
            _speech_prefix("Le forfait Fibre 100 est à 59 DT par mois, avec 50 DT."),
            "Le forfait Fibre 100 est à 59 DT par mois,",
        )

    def test_speech_prefixes_cover_the_full_answer_without_repeats(self):
        text = "Le forfait Fibre 100 est à 59 DT par mois, avec 50 DT de mise en service."
        pending = text
        spoken = []
        opening = True
        while True:
            candidate = _speech_prefix(pending, opening=opening)
            if candidate is None:
                break
            spoken.append(candidate)
            pending = _consume_speech_prefix(pending, candidate)
            opening = False
        if pending.strip():
            spoken.append(pending.strip())
        self.assertEqual(spoken[0], "Le forfait Fibre 100 est à 59 DT par mois,")
        self.assertEqual(" ".join(spoken), text)
        rest = text
        for phrase in spoken:
            self.assertTrue(rest.lstrip().startswith(phrase), phrase)
            rest = _consume_speech_prefix(rest, phrase)
        self.assertEqual(rest, "")

    def test_voice_context_keeps_prices_and_first_step_only(self):
        from app.services.rag.demo_knowledge import DEMO_PAGES

        fibre = _compact_chunk(next(page for page in DEMO_PAGES if "Forfaits Fibre" in page))
        self.assertIn("39 DT", fibre)
        self.assertIn("59 DT", fibre)
        self.assertIn("89 DT", fibre)
        self.assertIn("Fibre 50", fibre)
        self.assertIn("Fibre 300", fibre)
        reset = _compact_chunk(next(page for page in DEMO_PAGES if page.startswith("Réinitialisation")))
        self.assertIn("bouton Reset", reset)
        self.assertNotIn("Étape 2", reset)
        self.assertNotIn("IMPORTANT pour l'agent", reset)
        move = _compact_chunk(next(page for page in DEMO_PAGES if page.startswith("Déménagement")))
        self.assertIn("40 DT", move)
        self.assertIn("7 à 10 jours", move)

    async def test_failure_after_first_phrase_does_not_repeat_answer(self):
        async def chunks(text):
            if text == "first":
                yield b"audio"
            else:
                raise RuntimeError("provider failed")
        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock()), "test")
        session.send_json = AsyncMock()
        queue = asyncio.Queue()
        for phrase in ("first", "second", None):
            queue.put_nowait(phrase)
        with patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(synthesize_stream=chunks)):
            with self.assertRaises(PartialSpeechError):
                await session._stream_phrases(queue, time.perf_counter())
        session.ws.send_bytes.assert_awaited_once_with(b"audio")
        self.assertEqual(session.send_json.call_args.args[0]["type"], "tts_abort")

    async def test_debug_stage_reports_duration_and_offset_separately(self):
        session = VoiceSession(None, "test")
        session.send_json = AsyncMock()
        with patch("app.services.voice.session.time.perf_counter", side_effect=[100.0, 100.4]):
            async with session.debug_stage("retrieval", 90.0):
                pass
        report = session.send_json.call_args.args[0]
        self.assertEqual(report["stage"], "retrieval")
        self.assertEqual(report["offset_ms"], 10000)
        self.assertAlmostEqual(report["duration_ms"], 400)
        self.assertFalse(report["failed"])

    async def test_debug_stage_preserves_failed_measurements(self):
        session = VoiceSession(None, "test")
        session.send_json = AsyncMock()
        with self.assertRaises(TimeoutError):
            async with session.debug_stage("stt", time.perf_counter()):
                raise TimeoutError()
        self.assertTrue(session.send_json.call_args.args[0]["failed"])

    async def test_audio_stt_debug_includes_server_device(self):
        session = VoiceSession(None, "test")
        session.send_json = AsyncMock()
        extra = {"stt": "faster-whisper", "stt_device": "cuda", "stt_model": "base"}
        async with session.debug_stage("stt", time.perf_counter(), extra=extra):
            pass
        report = session.send_json.call_args.args[0]
        self.assertEqual(report["stt"], "faster-whisper")
        self.assertEqual(report["stt_device"], "cuda")

    async def test_socket_handles_ping_during_generation_and_cancels_on_disconnect(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        messages = []
        class Socket:
            client_state = WebSocketState.CONNECTED
            step = 0

            async def send_text(self, text):
                messages.append(json.loads(text))

            async def receive(self):
                self.step += 1
                if self.step == 1:
                    return {"text": json.dumps({"type": "user_transcript", "text": "Bonjour", "turn_id": "turn-test"})}
                if self.step == 2:
                    await asyncio.wait_for(started.wait(), 1)
                    return {"text": '{"type":"ping"}'}
                return {"type": "websocket.disconnect"}

        async def respond(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            patch.object(VoiceSession, "_respond_text", respond),
            patch("app.services.voice.session.wait_voice_runtime", AsyncMock()),
            patch("app.services.voice.session.start_stt_keepalive"),
            patch("app.services.voice.session.stop_stt_keepalive"),
            patch("app.services.voice.session.claude_service.warm_connection", AsyncMock()),
            patch("app.services.voice.session.describe_server_stt", return_value={"stt": "client-speech", "stt_device": None, "stt_model": None}),
            patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(name="test")),
            patch("app.services.voice.tts.synthesize_with_fallback", AsyncMock(return_value=(b"audio", "test"))),
            patch("app.services.voice.session.rag_service.store.preload", AsyncMock()),
        ):
            await asyncio.wait_for(voice_websocket_loop(Socket(), "test"), 2)
        ready = next(message for message in messages if message["type"] == "ready")
        self.assertIn("llm_provider", ready)
        self.assertIn("llm_model", ready)
        self.assertEqual(ready["stt"], "client-speech")
        self.assertIsNone(ready["stt_device"])
        self.assertTrue(ready["llm_model"])
        self.assertFalse(ready.get("smart_turn"))
        pong = next(message for message in messages if message["type"] == "pong")
        self.assertEqual(pong["turn_id"], "turn-test")
        self.assertTrue(pong["call_id"] and pong["trace_id"])
        self.assertTrue(cancelled.is_set())

    async def test_first_audio_precedes_provider_completion(self):
        sent = asyncio.Event()
        async def chunks(text):
            yield b"first"
            await asyncio.wait_for(sent.wait(), 1)
            yield b"last"

        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock(side_effect=lambda data: sent.set())), "test")
        session.send_json = AsyncMock()
        with patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(synthesize_stream=chunks)):
            elapsed = await session._synthesize_and_send("Bonjour", time.perf_counter())
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual([c.args[0] for c in session.ws.send_bytes.call_args_list], [b"first", b"last"])
        self.assertEqual(session.send_json.call_args_list[-1].args[0]["type"], "tts_end")

    async def test_cancelled_stream_is_aborted(self):
        sent = asyncio.Event()
        async def chunks(text):
            yield b"first"
            await asyncio.Event().wait()

        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock(side_effect=lambda data: sent.set())), "test")
        session.send_json = AsyncMock()
        with patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(synthesize_stream=chunks)):
            task = asyncio.create_task(session._synthesize_and_send("Bonjour", time.perf_counter()))
            await asyncio.wait_for(sent.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(session.send_json.call_args_list[-1].args[0]["type"], "tts_abort")

    async def test_partial_prefetch_cancels_when_user_changes_question(self):
        blocked = asyncio.Event()
        retrieve = AsyncMock(side_effect=lambda *args, **kwargs: None)
        async def lookup(*args, **kwargs):
            await blocked.wait()
            return [], 0
        retrieve.side_effect = lookup
        session = VoiceSession(None, "test")
        with patch("app.services.voice.session.rag_service.retrieve", retrieve):
            session.prefetch("Quel est le prix ?")
            old = session.prefetch_task
            await asyncio.sleep(0)
            session.prefetch("Comment résilier ?")
            await asyncio.sleep(0)
            self.assertTrue(old.cancelled())
            blocked.set()
            await session.prefetch_task
        self.assertEqual(session.prefetch_text, "Comment résilier ?")

    async def test_audio_is_sent_before_text_stream_finishes(self):
        audio_sent = asyncio.Event()
        stream_finished = False

        async def tokens():
            nonlocal stream_finished
            yield "Votre abonnement coûte vingt euros. "
            await asyncio.wait_for(audio_sent.wait(), 1)
            yield "Souhaitez-vous continuer ?"
            await asyncio.sleep(0.05)
            self.assertEqual(spoken[0], "Votre abonnement coûte vingt euros.")
            self.assertTrue(any("Souhaitez-vous continuer" in item for item in spoken))
            stream_finished = True

        class Stream:
            text_stream = tokens()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock(side_effect=lambda data: audio_sent.set())), "test")
        session.send_json = AsyncMock()

        async def send_audio(data):
            if not audio_sent.is_set():
                self.assertFalse(stream_finished)
            self.assertEqual(data, b"mp3")
            audio_sent.set()

        session.ws.send_bytes = AsyncMock(side_effect=send_audio)
        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: Stream()))
        client.with_options = lambda **kwargs: client
        claude = SimpleNamespace(client=client)
        rag = SimpleNamespace(retrieve=AsyncMock(return_value=([], 0)), to_sources=lambda chunks: [])
        spoken = []
        async def audio_chunks(text):
            spoken.append(text)
            yield b"mp3"

        with (
            patch("app.services.voice.session.claude_service", claude),
            patch("app.services.voice.session.rag_service", rag),
            patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(synthesize_stream=audio_chunks)),
            patch("app.db.repository.db_save_turn", AsyncMock()),
        ):
            await session._respond_text("Quel prix ?", time.perf_counter())
            await asyncio.sleep(0)
        self.assertEqual(session.ws.send_bytes.await_count, 2)
        self.assertEqual(spoken, ["Votre abonnement coûte vingt euros.", "Souhaitez-vous continuer ?"])
        events = [call.args[0]["type"] for call in session.send_json.call_args_list]
        self.assertEqual(events.count("tts_start"), 1)
        self.assertEqual(events.count("tts_end"), 1)
        self.assertTrue(stream_finished)

    async def test_tts_starts_on_first_clause_while_llm_continues(self):
        audio_sent = asyncio.Event()
        stream_finished = False
        last_token_sent = False
        pieces = [
            "Le ",
            "forfait ",
            "Fibre ",
            "100 ",
            "est ",
            "à ",
            "59 DT ",
            "par mois, ",
            "avec 50 DT de mise en service.",
        ]

        async def tokens():
            nonlocal stream_finished, last_token_sent
            for index, piece in enumerate(pieces):
                if index == len(pieces) - 1:
                    await asyncio.wait_for(audio_sent.wait(), 1)
                    self.assertFalse(stream_finished)
                    last_token_sent = True
                yield piece
            stream_finished = True

        class Stream:
            text_stream = tokens()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock()), "test")
        session.send_json = AsyncMock()
        spoken = []

        async def send_audio(data):
            if not audio_sent.is_set():
                self.assertFalse(last_token_sent)
            audio_sent.set()

        session.ws.send_bytes = AsyncMock(side_effect=send_audio)

        async def audio_chunks(text):
            spoken.append(text)
            yield b"mp3"

        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: Stream()))
        client.with_options = lambda **kwargs: client
        claude = SimpleNamespace(client=client)
        rag = SimpleNamespace(retrieve=AsyncMock(return_value=([], 0)), to_sources=lambda chunks: [])

        with (
            patch("app.services.voice.session.claude_service", claude),
            patch("app.services.voice.session.rag_service", rag),
            patch("app.services.voice.session.get_tts_provider", return_value=SimpleNamespace(synthesize_stream=audio_chunks)),
            patch("app.db.repository.db_save_turn", AsyncMock()),
        ):
            await session._respond_text("Combien coûte le forfait Fibre 100 ?", time.perf_counter())

        self.assertEqual(
            spoken,
            ["Le forfait Fibre 100 est à 59 DT par mois,", "avec 50 DT de mise en service."],
        )
        self.assertEqual(" ".join(spoken), "".join(pieces).strip())
        self.assertTrue(audio_sent.is_set())
        latency = next(call.args[0] for call in session.send_json.call_args_list if call.args[0].get("type") == "latency")
        self.assertTrue(latency["tts_overlapped_llm"])
        self.assertLess(latency["tts_first_audio_ms"], latency["llm_stream_ms"])
        self.assertTrue(stream_finished)
        phrase = next(
            call.args[0]
            for call in session.send_json.call_args_list
            if call.args[0].get("type") == "debug_timing" and call.args[0].get("stage") == "llm_phrase"
        )
        self.assertEqual(phrase.get("clause"), "Le forfait Fibre 100 est à 59 DT par mois,")


if __name__ == "__main__":
    unittest.main()
