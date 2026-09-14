import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.rag.demo_knowledge import DEMO_PAGES
from app.services.rag.query import expand_retrieval_query, lexical_needles, transcript_needs_clarification
from app.services.voice.session import VoiceSession, _compact_chunk


OFFERS_QUESTION = "Quelles sont les offres que vous avez ?"


def _offers_page() -> str:
    return next(page for page in DEMO_PAGES if "Forfaits Fibre" in page)


class OffersRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def test_offers_page_contains_all_plans(self):
        page = _offers_page()
        self.assertIn("Fibre 50 Mbps: 39 DT", page)
        self.assertIn("Fibre 100 Mbps", page)
        self.assertIn("59 DT", page)
        self.assertIn("Fibre 300 Mbps: 89 DT", page)
        self.assertIn("30 DT", page)
        self.assertIn("50 DT", page)

    def test_offers_query_maps_to_forfaits_fibre(self):
        self.assertIn("Forfaits Fibre", lexical_needles(OFFERS_QUESTION))
        self.assertIn("forfaits fibre tarifs", expand_retrieval_query(OFFERS_QUESTION).lower())
        compact = _compact_chunk(_offers_page())
        self.assertIn("Fibre 50", compact)
        self.assertIn("Fibre 100", compact)
        self.assertIn("Fibre 300", compact)
        self.assertIn("39 DT", compact)
        self.assertIn("59 DT", compact)
        self.assertIn("89 DT", compact)

    async def test_retrieve_and_voice_context_use_offer_document(self):
        from app.services.rag.service import rag_service
        from app.services.session.service import session_service

        meta = await session_service.create_tenant()
        info = await rag_service.ingest_demo_knowledge(meta.tenant_id)
        self.assertGreaterEqual(info.chunk_count, 1)
        status = await session_service.load_meta(meta.tenant_id)
        self.assertEqual(status.status, "ready")
        self.assertTrue(any("Forfaits Fibre" in page for page in DEMO_PAGES))

        chunks, _latency = await rag_service.retrieve(meta.tenant_id, OFFERS_QUESTION)
        self.assertIn("Forfaits Fibre", chunks[0].text)
        joined = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("Forfaits Fibre", joined)
        self.assertIn("Fibre 50", joined)
        self.assertIn("Fibre 100", joined)
        self.assertIn("Fibre 300", joined)
        self.assertIn("39 DT", joined)
        self.assertIn("59 DT", joined)
        self.assertIn("89 DT", joined)

        session = VoiceSession(SimpleNamespace(client_state=None), meta.tenant_id)
        context = session._voice_context(chunks)
        self.assertIn("Fibre 50", context)
        self.assertIn("39 DT", context)
        self.assertIn("59 DT", context)
        self.assertIn("89 DT", context)
        self.assertNotIn("Aucune information pertinente", context)

    async def test_text_answer_lists_grounded_offers(self):
        from app.core.config import get_settings
        from app.services.llm.claude import claude_service
        from app.services.rag.service import rag_service
        from app.services.session.service import session_service

        if not get_settings().gemini_api_key:
            self.skipTest("GEMINI_API_KEY missing")
        meta = await session_service.create_tenant()
        await rag_service.ingest_demo_knowledge(meta.tenant_id)
        response = await claude_service.answer(meta.tenant_id, OFFERS_QUESTION)
        answer = response.answer.lower()
        self.assertNotRegex(answer, r"absent|n'ai pas|pas dans")
        self.assertIn("39", response.answer)
        self.assertIn("59", response.answer)
        self.assertIn("89", response.answer)
        self.assertRegex(response.answer, r"\bDT\b")

    async def test_unclear_speech_asks_clarification_without_rag(self):
        session = VoiceSession(SimpleNamespace(send_bytes=AsyncMock()), "test")
        session.send_json = AsyncMock()
        session._synthesize_and_send = AsyncMock(return_value=0.0)
        retrieve = AsyncMock()
        with (
            patch("app.services.voice.session.rag_service.retrieve", retrieve),
            patch.object(session, "_prime_tts", AsyncMock()),
            patch.object(session, "_warm_ai", AsyncMock()),
        ):
            await session.handle_user_text("euh")
        retrieve.assert_not_called()
        texts = [call.args[0] for call in session.send_json.call_args_list]
        assistant = next(item["text"] for item in texts if item.get("role") == "assistant")
        self.assertIn("pas bien saisi", assistant.lower())
        self.assertIn("offres", assistant.lower())
        session._synthesize_and_send.assert_awaited()

    def test_filler_and_whisper_flags_need_clarification(self):
        self.assertTrue(transcript_needs_clarification("euh"))
        self.assertTrue(transcript_needs_clarification("Fibre 100", extra={"stt_unclear": 1}))
        self.assertFalse(transcript_needs_clarification(OFFERS_QUESTION))
        self.assertFalse(transcript_needs_clarification("oui"))


if __name__ == "__main__":
    unittest.main()
