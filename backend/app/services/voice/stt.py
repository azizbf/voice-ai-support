from __future__ import annotations

import io
import logging
import importlib.util
import tempfile
import wave
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_model_lock = Lock()


def _pcm16k_from_wav(audio_bytes: bytes) -> np.ndarray | None:
    """Decode 16-bit WAV in memory. Returns None if ffmpeg/tempfile fallback is needed."""
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    except wave.Error:
        return None
    if sample_width != 2 or not frames:
        return None
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000 and pcm.size > 1:
        duration = pcm.size / float(sample_rate)
        target = max(1, int(duration * 16000))
        pcm = np.interp(
            np.linspace(0, pcm.size - 1, target, dtype=np.float64),
            np.arange(pcm.size, dtype=np.float64),
            pcm,
        ).astype(np.float32)
    return pcm


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
            "Content-Type": "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/webm",
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
        import asyncio
        from faster_whisper import WhisperModel

        settings = get_settings()
        model_name = (settings.whisper_model or "base").strip() or "base"
        threads = max(1, int(settings.whisper_cpu_threads or 4))
        logger.info("Loading Whisper %s (cpu, int8, threads=%s)", model_name, threads)
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
        )
        self._slot = asyncio.Semaphore(1)
        # Compile CTranslate2 kernels so the first real utterance is not cold.
        silence = np.zeros(16000, dtype=np.float32)
        list(
            self.model.transcribe(
                silence,
                language="fr",
                vad_filter=False,
                beam_size=1,
                temperature=0,
                without_timestamps=True,
                condition_on_previous_text=False,
            )[0]
        )
        logger.info("Whisper %s ready", model_name)

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        try:
            await asyncio.wait_for(self._slot.acquire(), timeout=0.5)
        except TimeoutError as exc:
            raise RuntimeError("Reconnaissance vocale occupée. Réessayez dans un instant.") from exc
        worker = asyncio.create_task(asyncio.to_thread(self._transcribe_sync, audio_bytes))

        def finished(task: asyncio.Task) -> None:
            self._slot.release()
            if not task.cancelled():
                task.exception()

        worker.add_done_callback(finished)
        # Cancelling a call cannot stop CPU inference. Keep its slot reserved
        # until inference actually ends, preventing an unbounded worker queue.
        return await asyncio.shield(worker)

    def _transcribe_kwargs(self) -> dict:
        return {
            "language": "fr",
            # Browser already endpointed the clip; Silero VAD was extra latency.
            "vad_filter": False,
            "beam_size": 1,
            "temperature": 0,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        pcm = _pcm16k_from_wav(audio_bytes)
        if pcm is not None:
            segments, _info = self.model.transcribe(pcm, **self._transcribe_kwargs())
            return " ".join(seg.text.strip() for seg in segments).strip()

        suffix = ".webm"
        if audio_bytes[:4] == b"RIFF":
            suffix = ".wav"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            suffix = ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        try:
            segments, _info = self.model.transcribe(path, **self._transcribe_kwargs())
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            Path(path).unlink(missing_ok=True)


class UnavailableStt(SttProvider):
    name = "client-speech"

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        return ""


class LazyWhisperStt(SttProvider):
    name = "faster-whisper"

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        provider = await asyncio.to_thread(_faster_whisper_provider)
        return await provider.transcribe(audio_bytes, sample_rate)


@lru_cache
def get_stt_provider() -> SttProvider:
    """Advertise local transcription without loading a model on the socket loop."""
    settings = get_settings()
    if settings.deepgram_api_key:
        return DeepgramStt(settings.deepgram_api_key)
    if importlib.util.find_spec("faster_whisper") is not None:
        return LazyWhisperStt()
    return UnavailableStt()


@lru_cache
def _load_whisper() -> FasterWhisperStt:
    return FasterWhisperStt()


def _faster_whisper_provider() -> FasterWhisperStt:
    # Warm-up and the first call can arrive together; load only one model.
    with _model_lock:
        return _load_whisper()


async def warm_stt() -> None:
    import asyncio

    if get_stt_provider().name == "faster-whisper":
        await asyncio.to_thread(_faster_whisper_provider)


def get_audio_stt_provider() -> SttProvider:
    """Used only when the browser sends recorded audio frames."""
    settings = get_settings()
    if settings.deepgram_api_key:
        return DeepgramStt(settings.deepgram_api_key)
    try:
        import faster_whisper  # noqa: F401

        return LazyWhisperStt()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Server STT unavailable (%s); expecting client transcripts", exc)
        return UnavailableStt()
