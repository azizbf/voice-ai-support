import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services.llm.claude import ClaudeService


class AiConnectionTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        service = ClaudeService()
        client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock()), close=AsyncMock())
        client.with_options = MagicMock(return_value=client)
        service._client = client
        return service, client

    async def test_warming_uses_read_only_request_not_generation(self):
        service, client = self.service()
        await service.warm_connection()
        client.models.list.assert_awaited_once_with(limit=1)
        client.with_options.assert_called_once_with(timeout=3.0, max_retries=0)
        await service.warm_connection()
        client.models.list.assert_awaited_once()

    async def test_active_connection_does_not_get_extra_requests(self):
        service, client = self.service()
        service._last_used = time.monotonic()
        await service.warm_connection()
        client.models.list.assert_not_awaited()

    async def test_warm_failure_does_not_break_the_call(self):
        service, client = self.service()
        client.models.list.side_effect = RuntimeError("unavailable")
        await service.warm_connection()
        await service.close()
        client.close.assert_awaited_once()
        self.assertIsNone(service._client)
