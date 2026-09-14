"""Sentence input and MP3 output concurrently over one socket per response."""
import asyncio
import base64
import json
from collections.abc import AsyncIterator
from urllib.parse import quote

import aiohttp

from app.services.voice.tts import TtsProvider, _SSL_CTX


class ElevenLabsTtsProvider(TtsProvider):
    name = "elevenlabs"
    supports_text_stream = True

    def __init__(self, api_key: str, voice_id: str, model: str):
        if not api_key.strip() or not voice_id.strip():
            raise ValueError("ElevenLabs requires ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID")
        self.api_key, self.voice_id, self.model = api_key, voice_id, model
        self._session = None

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def synthesize(self, text: str) -> bytes:
        return b"".join([part async for part in self.synthesize_stream(text)])

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        async def texts():
            yield text
        async for audio in self.synthesize_text_stream(texts()):
            yield audio

    async def synthesize_text_stream(self, texts: AsyncIterator[str]) -> AsyncIterator[bytes]:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=_SSL_CTX, keepalive_timeout=120),
                timeout=aiohttp.ClientTimeout(total=20, sock_connect=5),
            )
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{quote(self.voice_id, safe='')}/stream-input"
        async with self._session.ws_connect(
            url, headers={"xi-api-key": self.api_key},
            params={"model_id": self.model, "output_format": "mp3_22050_32", "auto_mode": "true"},
            receive_timeout=15,
        ) as ws:
            await ws.send_json({"text": " "})

            async def send_text():
                try:
                    async for text in texts:
                        if text.strip():
                            await ws.send_json({"text": text.strip() + " "})
                    await ws.send_json({"text": ""})
                except BaseException:
                    await ws.close()
                    raise

            sender = asyncio.create_task(send_text())
            finalized = False
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError("ElevenLabs audio connection failed")
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = json.loads(message.data)
                    if data.get("error") or data.get("message"):
                        raise RuntimeError("ElevenLabs rejected the speech request")
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"], validate=True)
                    if data.get("is_final") or data.get("isFinal"):
                        finalized = True
                        break
                await sender
                if not finalized:
                    raise RuntimeError("ElevenLabs stream closed before completion")
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
