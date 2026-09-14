from __future__ import annotations

import asyncio
import logging
import ssl
import time
from collections.abc import AsyncIterator
from collections import OrderedDict
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

    async def close(self) -> None:
        return None


class EdgeTtsProvider(TtsProvider):
    """Microsoft Edge neural TTS with a kept-alive HTTP session.

    Handshake is started with ``prime()`` while the LLM is still generating so
    the first SSML request only waits for audio, not TLS + WebSocket setup.
    """

    name = "edge-tts"

    def __init__(self, voice: str = "fr-FR-DeniseNeural") -> None:
        self.voice = voice
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()
        self._ws: Any | None = None
        self._ws_opened_at = 0.0
        self._prime_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._connecting: asyncio.Future[bool] | None = None
        self._closing = False
        self.max_idle_seconds = 45.0
        self._audio_cache: OrderedDict[tuple[str, str], tuple[float, bytes]] = OrderedDict()
        self._cache_bytes = 0

    async def synthesize(self, text: str) -> bytes:
        return b"".join([chunk async for chunk in self.synthesize_stream(text)])

    def _live_socket(self) -> Any | None:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            return None
        if time.monotonic() - self._ws_opened_at >= self.max_idle_seconds:
            return None
        return ws

    async def ensure_ready(self) -> None:
        if self._live_socket() is not None:
            return
        await self.prime()

    async def prime(self) -> None:
        starter = False
        try:
            async with self._lock:
                if self._closing:
                    return
                if self._live_socket() is not None:
                    self._ensure_heartbeat()
                    return
                if self._connecting is not None and not self._connecting.done():
                    wait = self._connecting
                else:
                    wait = None
                    self._connecting = asyncio.get_running_loop().create_future()
                    starter = True
            if wait is not None:
                await wait
                return
            ws = await self._open_ws()
            async with self._lock:
                if self._closing:
                    await self._close_ws(ws)
                    return
                if self._ws is not None:
                    await self._close_ws(self._ws)
                self._ws = ws
                self._ws_opened_at = time.monotonic()
                self._ensure_heartbeat()
                if self._connecting is not None and not self._connecting.done():
                    self._connecting.set_result(True)
        except Exception:
            logger.debug("TTS prime failed", exc_info=True)
            async with self._lock:
                if self._ws is not None and getattr(self._ws, "closed", True):
                    self._ws = None
                if self._connecting is not None and not self._connecting.done():
                    self._connecting.set_result(False)
        finally:
            if starter:
                async with self._lock:
                    self._connecting = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        text = (text or "").strip()
        if not text:
            return
        cache_key = (self.voice, text)
        cached = self._audio_cache.pop(cache_key, None)
        if cached:
            self._cache_bytes -= len(cached[1])
            if time.monotonic() - cached[0] < 300:
                self._audio_cache[cache_key] = cached
                self._cache_bytes += len(cached[1])
                yield cached[1]
                return
        ws = await self._take_socket()
        got_audio = False
        chunks: list[bytes] = []
        size = 0
        try:
            try:
                await self._send_ssml(ws, text)
                async for audio in self._iter_audio(ws):
                    got_audio = True
                    size += len(audio)
                    if size <= 128_000:
                        chunks.append(audio)
                    yield audio
                if not got_audio:
                    raise RuntimeError("TTS connection ended before audio arrived")
            except Exception:
                if got_audio:
                    raise
                await self._close_ws(ws)
                ws = await self._take_socket()
                await self._send_ssml(ws, text)
                async for audio in self._iter_audio(ws):
                    got_audio = True
                    size += len(audio)
                    if size <= 128_000:
                        chunks.append(audio)
                    yield audio
                if not got_audio:
                    raise RuntimeError("TTS connection returned no audio")
            if len(text) <= 256 and 0 < size <= 128_000:
                previous = self._audio_cache.pop(cache_key, None)
                if previous:
                    self._cache_bytes -= len(previous[1])
                data = b"".join(chunks)
                self._audio_cache[cache_key] = (time.monotonic(), data)
                self._cache_bytes += len(data)
                while len(self._audio_cache) > 32 or self._cache_bytes > 2_000_000:
                    _, (_, removed) = self._audio_cache.popitem(last=False)
                    self._cache_bytes -= len(removed)
        finally:
            await self._recycle_socket(ws)

    async def close(self) -> None:
        self._closing = True
        self._audio_cache.clear()
        self._cache_bytes = 0
        tasks = [task for task in (self._prime_task, self._heartbeat_task) if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            if self._ws is not None:
                await self._close_ws(self._ws)
                self._ws = None
            if self._session is not None:
                await self._session.close()

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
            heartbeat=20.0,
            headers=DRM.headers_with_muid(WSS_HEADERS),
            ssl=_SSL_CTX,
        )
        from edge_tts.communicate import date_to_string

        try:
            await ws.send_str(
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                '{"context":{"synthesis":{"audio":{"metadataoptions":{'
                '"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"false"'
                "},"
                '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
                "}}}}\r\n"
            )
            await self._wait_config_ack(ws)
        except BaseException:
            await self._close_ws(ws)
            raise
        return ws

    async def _wait_config_ack(self, ws: Any) -> None:
        # Don't sit on Microsoft's config reply; SSML can follow immediately.
        try:
            received = await asyncio.wait_for(ws.receive(), timeout=0.05)
        except TimeoutError:
            return
        if received.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
            raise RuntimeError("TTS socket closed during handshake")

    def _schedule_prime(self) -> None:
        if self._closing:
            return
        if self._prime_task is None or self._prime_task.done():
            self._prime_task = asyncio.create_task(self.prime())

    def _ensure_heartbeat(self) -> None:
        if self._closing:
            return
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        while not self._closing:
            await asyncio.sleep(6)
            try:
                async with self._lock:
                    ws = self._live_socket()
                if ws is None:
                    await self.prime()
                    continue
                ping = getattr(ws, "ping", None)
                if callable(ping):
                    await ping()
            except Exception:
                logger.debug("TTS keepalive failed", exc_info=True)
                async with self._lock:
                    if self._ws is not None:
                        await self._close_ws(self._ws)
                        self._ws = None
                await self.prime()

    async def _take_socket(self) -> Any:
        while not self._closing:
            async with self._lock:
                ws = self._ws
                self._ws = None
                age = time.monotonic() - self._ws_opened_at
                connecting = self._connecting
            if ws is not None and not getattr(ws, "closed", False) and age < self.max_idle_seconds:
                self._schedule_prime()
                return ws
            if ws is not None:
                await self._close_ws(ws)
            if connecting is not None and not connecting.done():
                try:
                    await connecting
                except Exception:
                    pass
                continue
            opened = await self._open_ws()
            self._schedule_prime()
            return opened
        raise RuntimeError("TTS provider is closing")

    async def _recycle_socket(self, ws: Any) -> None:
        # Edge sockets are usually spent after turn.end. Recycle caused a cold
        # handshake (~1.5–2s) on the next phrase; close and keep a primed spare.
        if ws is not None:
            await self._close_ws(ws)
        self._schedule_prime()

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
            elif received.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                return
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
    if settings.tts_provider == "elevenlabs":
        from app.services.voice.elevenlabs import ElevenLabsTtsProvider

        return ElevenLabsTtsProvider(settings.elevenlabs_api_key, settings.elevenlabs_voice_id, settings.elevenlabs_model)
    logger.info("TTS provider: edge-tts (%s)", settings.edge_tts_voice)
    return EdgeTtsProvider(settings.edge_tts_voice)


def reset_tts_provider() -> None:
    get_settings.cache_clear()
    get_tts_provider.cache_clear()


async def synthesize_with_fallback(text: str) -> tuple[bytes, str]:
    provider = get_tts_provider()
    audio = await provider.synthesize(text)
    return audio, provider.name
