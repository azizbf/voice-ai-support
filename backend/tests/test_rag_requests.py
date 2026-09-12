import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.security import issue_tenant_token
from app.main import app


class RagRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_json_bodies_reach_authentication(self):
        for path, body in (
            ("tts", {"tenant_id": "test", "text": "Bonjour"}),
            ("chat", {"tenant_id": "test", "message": "Bonjour"}),
            ("rag/retrieve", {"tenant_id": "test", "query": "Bonjour"}),
        ):
            with self.subTest(path=path):
                response = self.client.post(f"/api/v1/{path}", json=body)
                self.assertEqual(response.status_code, 401, response.text)

    def test_tts_returns_audio_for_authenticated_json(self):
        token = issue_tenant_token("test", time.time() + 60)
        synthesize = AsyncMock(return_value=(b"test-mp3", "test-provider"))
        with patch("app.services.voice.tts.synthesize_with_fallback", synthesize):
            response = self.client.post(
                "/api/v1/tts",
                json={"tenant_id": "test", "text": " Bonjour "},
                headers={"X-Tenant-Token": token},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"test-mp3")
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        synthesize.assert_awaited_once_with("Bonjour")

    def test_tts_still_requires_text(self):
        response = self.client.post("/api/v1/tts", json={"tenant_id": "test"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "text"])


if __name__ == "__main__":
    unittest.main()
