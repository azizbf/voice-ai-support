from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any
from xml.sax.saxutils import escape

import aiohttp
import certifi

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


class TtsProvider:
    name: str

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        audio = await self.synthesize(text)
        yield audio

    async def prime(self) -> None:
        return None


class EdgeTtsProvider(TtsProvider):
    """Microsoft Edge neural TTS with a kept-alive HTTP session.

    Handshake is started with ``prime()`` while the LLM is still generating so
    the first SSML request only waits for audio, not TLS + WebSocket setup.
    """

    name = "edge-tts"

    def __init__(self, voice: str = "fr-FR-VivienneMultilingualNeural") -> None:
        self.voice = voice
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        self._ws: Any | None = None

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.synthesize_stream(text)])

    async def prime(self) -> None:
        try:
            async with self._lock:
                if self._ws is not None and not self._ws.closed:
                    return
                self._ws = await self._open_ws()
        except Exception:
            logger.debug("TTS prime failed", exc_info=True)
            self._ws = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        ws = await self._take_socket()
        got_audio = False
        try:
            try:
                await self._send_ssml(ws, text)
                async for audio in self._iter_audio(ws):
                    got_audio = True
                    yield audio
            except Exception:
                if got_audio:
                    raise
                await self._close_ws(ws)
                ws = await self._open_ws()
                await self._send_ssml(ws, text)
                async for audio in self._iter_audio(ws):
                    yield audio
        finally:
            await self._close_ws(ws)
            asyncio.create_task(self.prime())

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=8,
                ttl_dns_cache=300,
                keepalive_timeout=45,
                enable_cleanup_closed=True,
                ssl=_SSL_CTX,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=10),
            )
        return self._session

    async def _open_ws(self) -> Any:
        from edge_tts.communicate import connect_id
        from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
        from edge_tts.drm import DRM

        session = await self._session_get()
        ws = await session.ws_connect(
            f"{WSS_URL}&ConnectionId={connect_id()}"
            f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
            compress=15,
            headers=DRM.headers_with_muid(WSS_HEADERS),
            ssl=_SSL_CTX,
        )
        from edge_tts.communicate import date_to_string

        await ws.send_str(
            f"X-Timestamp:{date_to_string()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            '"sentenceBoundaryEnabled":"true","wordBoundaryEnabled":"false"'
            "},"
            '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
            "}}}}\r\n"
        )
        return ws

    async def _take_socket(self) -> Any:
        async with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None and not ws.closed:
            return ws
        return await self._open_ws()

    async def _send_ssml(self, ws: Any, text: str) -> None:
        from edge_tts.communicate import (
            connect_id,
            date_to_string,
            mkssml,
            remove_incompatible_characters,
            ssml_headers_plus_data,
        )
        from edge_tts.data_classes import TTSConfig

        config = TTSConfig(self.voice, "+5%", "+0%", "+0Hz", "SentenceBoundary")
        escaped = escape(remove_incompatible_characters(text))
        await ws.send_str(
            ssml_headers_plus_data(connect_id(), date_to_string(), mkssml(config, escaped))
        )

    async def _iter_audio(self, ws: Any) -> AsyncIterator[bytes]:
        from edge_tts.communicate import get_headers_and_data

        async for received in ws:
            if received.type == aiohttp.WSMsgType.TEXT:
                encoded = received.data.encode("utf-8")
                parameters, _data = get_headers_and_data(encoded, encoded.find(b"\r\n\r\n"))
                if parameters.get(b"Path") == b"turn.end":
                    return
            elif received.type == aiohttp.WSMsgType.BINARY:
                if len(received.data) < 2:
                    continue
                header_length = int.from_bytes(received.data[:2], "big")
                parameters, data = get_headers_and_data(received.data, header_length)
                if parameters.get(b"Path") == b"audio" and data:
                    yield data
            elif received.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(received.data or "TTS websocket error")

    async def _close_ws(self, ws: Any) -> None:
        try:
            await ws.close()
        except Exception:
            return


@lru_cache
def get_tts_provider() -> TtsProvider:
    settings = get_settings()
    logger.info("TTS provider: edge-tts (%s)", settings.edge_tts_voice)
    return EdgeTtsProvider(settings.edge_tts_voice)


def reset_tts_provider() -> None:
    get_settings.cache_clear()
    get_tts_provider.cache_clear()


async def synthesize_with_fallback(text: str) -> tuple[bytes, str]:
    provider = get_tts_provider()
    audio = await provider.synthesize(text)
    return audio, provider.name
