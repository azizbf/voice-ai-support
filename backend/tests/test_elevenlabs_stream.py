import asyncio
import base64
import json
import unittest
from types import SimpleNamespace

import aiohttp

from app.services.voice.elevenlabs import ElevenLabsTtsProvider


class Socket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        self.closed = True

    async def send_json(self, data):
        self.sent.append(data)
        if data['text'].strip():
            await self.incoming.put({'audio': base64.b64encode(b'audio').decode()})
        elif data['text'] == '':
            await self.incoming.put({'is_final': True})

    def __aiter__(self):
        return self

    async def __anext__(self):
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(await self.incoming.get()))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_arrives_before_remaining_text_and_one_socket(self):
        ws = Socket()
        calls = []
        def connect(*args, **kwargs):
            calls.append(kwargs)
            return ws
        provider = ElevenLabsTtsProvider('secret', 'voice', 'eleven_flash_v2_5')
        provider._session = SimpleNamespace(closed=False, ws_connect=connect)
        remaining = asyncio.Event()
        async def texts():
            yield 'Bonjour.'
            await remaining.wait()
            yield 'La suite.'
        stream = provider.synthesize_text_stream(texts())
        self.assertEqual(await asyncio.wait_for(anext(stream), 1), b'audio')
        self.assertFalse(remaining.is_set())
        remaining.set()
        self.assertEqual([audio async for audio in stream], [b'audio'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(ws.sent[-1], {'text': ''})
        self.assertTrue(ws.closed)

    async def test_close_cancels_waiting_text_sender(self):
        ws = Socket()
        provider = ElevenLabsTtsProvider('secret', 'voice', 'eleven_flash_v2_5')
        provider._session = SimpleNamespace(closed=False, ws_connect=lambda *a, **k: ws)
        cancelled = asyncio.Event()
        async def texts():
            try:
                yield 'Bonjour.'
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        stream = provider.synthesize_text_stream(texts())
        await anext(stream)
        await stream.aclose()
        self.assertTrue(cancelled.is_set())
        self.assertTrue(ws.closed)

    def test_credentials_required(self):
        with self.assertRaises(ValueError):
            ElevenLabsTtsProvider('', '', 'eleven_flash_v2_5')
