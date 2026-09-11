from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class TtsProvider:
    name: str

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        audio = await self.synthesize(text)
        yield audio


class EdgeTtsProvider(TtsProvider):
    name = "edge-tts"

    def __init__(self, voice: str = "fr-FR-VivienneMultilingualNeural") -> None:
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        import edge_tts

        # Natural French pacing for call-center replies
        communicate = edge_tts.Communicate(text, self.voice, rate="+5%", pitch="+0Hz")
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)


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
