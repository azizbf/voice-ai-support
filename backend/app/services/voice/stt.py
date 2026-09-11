from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class SttProvider:
    name: str

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        raise NotImplementedError


class DeepgramStt(SttProvider):
    name = "deepgram"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import httpx

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/webm",
        }
        params = {
            "model": "nova-2",
            "language": "fr",
            "smart_format": "true",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                content=audio_bytes,
            )
            resp.raise_for_status()
            data = resp.json()
        try:
            return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError):
            return ""


class FasterWhisperStt(SttProvider):
    name = "faster-whisper"

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel("small", device="cpu", compute_type="int8")

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        suffix = ".webm"
        if audio_bytes[:4] == b"RIFF":
            suffix = ".wav"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            suffix = ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        try:
            segments, _info = self.model.transcribe(path, language="fr", vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            Path(path).unlink(missing_ok=True)


class UnavailableStt(SttProvider):
    name = "client-speech"

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        return ""


@lru_cache
def get_stt_provider() -> SttProvider:
    settings = get_settings()
    if settings.deepgram_api_key:
        return DeepgramStt(settings.deepgram_api_key)
    try:
        import faster_whisper  # noqa: F401

        return FasterWhisperStt()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Server STT unavailable (%s); expecting client transcripts", exc)
        return UnavailableStt()
