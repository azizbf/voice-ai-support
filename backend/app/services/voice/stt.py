from __future__ import annotations

import io
import logging
import importlib.util
import wave
import os
import subprocess
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
STT_IMPL = "shield-executor-v2"
_model_lock = Lock()
_cuda_dll_handles: list = []
_cudart = None
_SPEECH_RMS = 0.012
_stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-stt")
_gpu_status: dict[str, float | str] = {}
_keep_alive = False
_keep_stop = Event()
_keep_thread: threading.Thread | None = None
_user_stt = Event()
_user_stt_lock = Lock()
_user_stt_waiters = 0
_pulse_lock = Lock()
_pulse_queued = False
_NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=pstate,clocks.gr,power.draw",
    "--format=csv,noheader,nounits",
]


def _cuda_synchronize() -> bool:
    """Block until queued CUDA work finishes. Returns False if CUDA is unavailable."""
    global _cudart
    if os.name != "nt":
        return False
    try:
        if _cudart is None:
            root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cuda_runtime" / "bin"
            matches = sorted(root.glob("cudart64_*.dll"))
            if not matches:
                return False
            import ctypes

            _cudart = ctypes.WinDLL(str(matches[-1]))
            _cudart.cudaDeviceSynchronize.restype = ctypes.c_int
        return _cudart.cudaDeviceSynchronize() == 0
    except Exception:
        return False


def _query_gpu_status() -> dict[str, float | str]:
    """Read-only nvidia-smi query. Never sets clocks or power limits."""
    global _gpu_status
    try:
        out = subprocess.check_output(_NVIDIA_QUERY, text=True, timeout=1.5)
        pstate, clock, power = [part.strip() for part in out.split(",", 2)]
        _gpu_status = {
            "stt_gpu_pstate": pstate,
            "stt_gpu_clock_mhz": float(clock),
            "stt_gpu_power_w": float(power),
        }
    except Exception:
        pass
    return dict(_gpu_status)


def _begin_user_stt() -> None:
    global _user_stt_waiters
    with _user_stt_lock:
        _user_stt_waiters += 1
        _user_stt.set()


def _end_user_stt() -> None:
    global _user_stt_waiters
    with _user_stt_lock:
        _user_stt_waiters = max(0, _user_stt_waiters - 1)
        if _user_stt_waiters == 0:
            _user_stt.clear()


def _pcm_speech_stats(pcm: np.ndarray, sample_rate: int = 16000) -> dict[str, float]:
    """Energy-based duration stats. Does not trim audio."""
    samples = int(np.asarray(pcm).size)
    audio_ms = 1000.0 * samples / float(sample_rate) if sample_rate else 0.0
    empty = {
        "stt_audio_ms": round(audio_ms),
        "stt_lead_silence_ms": round(audio_ms),
        "stt_trail_silence_ms": round(audio_ms),
        "stt_speech_ms": 0,
    }
    if samples <= 0 or sample_rate <= 0:
        return empty
    frame = max(1, int(round(sample_rate * 0.02)))
    usable = samples - (samples % frame)
    if usable < frame:
        return empty
    shaped = np.asarray(pcm, dtype=np.float32).reshape(-1)[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(shaped), axis=1))
    spoken = np.flatnonzero(rms > _SPEECH_RMS)
    if spoken.size == 0:
        return empty
    lead = spoken[0] * frame / sample_rate * 1000.0
    trail = (samples - min(samples, (int(spoken[-1]) + 1) * frame)) / sample_rate * 1000.0
    speech_ms = (int(spoken[-1]) - int(spoken[0]) + 1) * frame / sample_rate * 1000.0
    return {
        "stt_audio_ms": round(audio_ms),
        "stt_lead_silence_ms": round(lead),
        "stt_trail_silence_ms": round(trail),
        "stt_speech_ms": round(speech_ms),
    }


class _TimedFeatureExtractor:
    def __init__(self, inner, sink: dict[str, float]) -> None:
        self._inner = inner
        self._sink = sink

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        features = self._inner(*args, **kwargs)
        self._sink["stt_feat_ms"] = (time.perf_counter() - started) * 1000
        return features

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _prepare_cuda_libraries() -> None:
    """Expose project-local NVIDIA wheels for this process only.

    CTranslate2 on Windows resolves CUDA with the process PATH, not
    add_dll_directory. Prepend venv NVIDIA bins without editing the
    machine PATH or installing a display driver.
    """
    if os.name != "nt" or _cuda_dll_handles:
        return
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    dirs: list[str] = []
    for component in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        for folder in ("bin", "lib"):
            path = root / component / folder
            if path.is_dir():
                dirs.append(str(path))
                _cuda_dll_handles.append(os.add_dll_directory(str(path)))
    if dirs:
        os.environ["PATH"] = os.pathsep.join((*dirs, os.environ.get("PATH", "")))


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


def _decode_container_pcm16k(audio_bytes: bytes) -> tuple[np.ndarray, dict[str, float]]:
    """Decode compressed audio to 16 kHz mono float32 without forcing a GC cycle.

    faster_whisper.audio.decode_audio creates an AudioResampler, then calls
    gc.collect() to work around SYSTRAN/faster-whisper#390. That heap walk is
    on the live STT path. Reusing the resampler after a flush raises EOFError,
    so each call still builds a new one. Natural GC stays enabled.
    """
    import av
    from faster_whisper.audio import _group_frames, _ignore_invalid_frames, _resample_frames

    codec_started = time.perf_counter()
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
    raw_buffer = io.BytesIO()
    dtype = None
    with av.open(io.BytesIO(audio_bytes), mode="r", metadata_errors="ignore") as container:
        frames = _resample_frames(
            _group_frames(_ignore_invalid_frames(container.decode(audio=0)), 500000),
            resampler,
        )
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
    codec_ms = (time.perf_counter() - codec_started) * 1000

    alloc_started = time.perf_counter()
    if dtype is None or raw_buffer.tell() == 0:
        pcm = np.zeros(0, dtype=np.float32)
    else:
        pcm = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)
    alloc_ms = (time.perf_counter() - alloc_started) * 1000
    return pcm, {
        "stt_decode_codec_ms": round(codec_ms),
        "stt_decode_alloc_ms": round(alloc_ms),
    }


def _decode_pcm16k_timed(audio_bytes: bytes) -> tuple[np.ndarray, dict[str, float]]:
    wav = _pcm16k_from_wav(audio_bytes)
    if wav is not None:
        return wav, {}
    return _decode_container_pcm16k(audio_bytes)


def _decode_pcm16k(audio_bytes: bytes) -> np.ndarray:
    """PCM 16 kHz mono. WAV in-process; other containers via PyAV."""
    pcm, _timings = _decode_pcm16k_timed(audio_bytes)
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


def _stt_segment_quality(segments) -> tuple[str, dict]:
    texts: list[str] = []
    logprobs: list[float] = []
    no_speech: list[float] = []
    ratios: list[float] = []
    for seg in segments:
        piece = (getattr(seg, "text", "") or "").strip()
        if piece:
            texts.append(piece)
        logprobs.append(float(getattr(seg, "avg_logprob", 0.0) or 0.0))
        no_speech.append(float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
        ratios.append(float(getattr(seg, "compression_ratio", 1.0) or 1.0))
    text = " ".join(texts).strip()
    avg_lp = min(logprobs) if logprobs else 0.0
    ns = max(no_speech) if no_speech else 0.0
    cr = max(ratios) if ratios else 1.0
    unclear = bool(text) and (avg_lp < -1.35 or ns > 0.65 or cr > 2.6)
    return text, {
        "stt_avg_logprob": round(avg_lp, 3),
        "stt_no_speech": round(ns, 3),
        "stt_compression": round(cr, 3),
        "stt_unclear": 1 if unclear else 0,
    }


class FasterWhisperStt(SttProvider):
    name = "faster-whisper"

    def __init__(self) -> None:
        import asyncio

        settings = get_settings()
        model_name = (settings.whisper_model or "base").strip() or "base"
        threads = max(1, int(settings.whisper_cpu_threads or 4))
        device = settings.whisper_device
        if device != "cpu":
            _prepare_cuda_libraries()
        from faster_whisper import WhisperModel
        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.device = device
        logger.info("Loading Whisper %s (%s, %s, threads=%s)", model_name, device, compute_type, threads)
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=threads,
        )
        self._slot = asyncio.Semaphore(1)
        self.last_timings: dict[str, float | str | int] = {}
        self._infer_calls = 0
        self.startup_warmed = False
        self.model_name = model_name
        self._last_infer_at = 0.0
        self._pulse_features = None
        self._keep_awake = False
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
        if device == "cuda":
            _cuda_synchronize()
            _query_gpu_status()
        self.startup_warmed = True
        logger.info("Whisper %s ready", model_name)

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        waited = time.perf_counter()
        _begin_user_stt()
        acquired = False
        fut: asyncio.Future | None = None
        queue_ms = 0.0
        try:
            if getattr(self, "_slot", None) is None:
                self._slot = asyncio.Semaphore(1)
            await self._slot.acquire()
            acquired = True
            queue_ms = (time.perf_counter() - waited) * 1000
            if queue_ms >= 50:
                logger.info("STT queued %.0f ms behind in-flight work", queue_ms)
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(_stt_executor, self._transcribe_sync, audio_bytes)
            try:
                text = await asyncio.shield(fut)
            except asyncio.CancelledError:
                if fut is not None and not fut.done():
                    try:
                        await asyncio.shield(fut)
                    except Exception:
                        pass
                raise
            self.last_timings = {**getattr(self, "last_timings", {}), "stt_queue_ms": round(queue_ms)}
            return text
        except BaseException:
            self.last_timings = {**getattr(self, "last_timings", {}), "stt_queue_ms": round(queue_ms)}
            raise
        finally:
            if fut is not None and not fut.done():
                try:
                    await asyncio.shield(fut)
                except Exception:
                    pass
            if acquired:
                self._slot.release()
            _end_user_stt()

    def _transcribe_kwargs(self) -> dict:
        return {
            "language": "fr",
            # Browser already endpointed the clip; Silero VAD was extra latency.
            "vad_filter": False,
            "beam_size": 1,
            "temperature": 0,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "best_of": 1,
        }

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        started = time.perf_counter()
        decode_parts: dict[str, float] = {}
        infer_parts: dict[str, float] = {}
        speech = _pcm_speech_stats(np.zeros(0, dtype=np.float32))
        decode_ms = 0.0
        infer_ms = 0.0
        samples = 0
        text = ""
        last_infer = getattr(self, "_last_infer_at", 0.0)
        idle_ms = 0.0 if not last_infer else (time.perf_counter() - last_infer) * 1000
        try:
            pcm, decode_parts = _decode_pcm16k_timed(audio_bytes)
            decode_ms = (time.perf_counter() - started) * 1000
            speech = _pcm_speech_stats(pcm)
            samples = int(pcm.size)
            infer_started = time.perf_counter()
            infer_parts, text = self._infer_pcm(pcm)
            infer_ms = (time.perf_counter() - infer_started) * 1000
            self._infer_calls = getattr(self, "_infer_calls", 0) + 1
            self._last_infer_at = time.perf_counter()
            return text
        finally:
            synced = _cuda_synchronize() if getattr(self, "device", None) == "cuda" else False
            gpu = dict(_gpu_status)
            self.last_timings = {
                "stt_decode_ms": round(decode_ms),
                "stt_infer_ms": round(infer_ms),
                "stt_samples": samples,
                "stt_gpu_sync": 1 if synced else 0,
                "stt_startup_warmed": 1 if getattr(self, "startup_warmed", False) else 0,
                "stt_infer_n": getattr(self, "_infer_calls", 0),
                "stt_idle_ms": round(idle_ms),
                "stt_thread": threading.get_ident(),
                "stt_keep_awake": 1 if getattr(self, "_keep_awake", False) else 0,
                **speech,
                **decode_parts,
                **infer_parts,
                **gpu,
            }
            model_name = getattr(self, "model_name", None)
            if model_name:
                self.last_timings["stt_model"] = f"{model_name}:{getattr(self, 'device', '?')}"

    def _infer_pcm(self, pcm: np.ndarray) -> tuple[dict[str, float], str]:
        parts: dict[str, float] = {}
        if not hasattr(self.model, "encode") or not hasattr(self.model, "feature_extractor"):
            segments, _info = self.model.transcribe(pcm, **self._transcribe_kwargs())
            text, quality = _stt_segment_quality(segments)
            parts.update(quality)
            return parts, text
        orig_extractor = self.model.feature_extractor
        orig_encode = self.model.encode
        orig_generate = self.model.generate_with_fallback
        cuda = getattr(self, "device", None) == "cuda"

        def encode(*args, **kwargs):
            started = time.perf_counter()
            output = orig_encode(*args, **kwargs)
            if cuda:
                _cuda_synchronize()
            parts["stt_encode_ms"] = (time.perf_counter() - started) * 1000
            return output

        def generate_with_fallback(*args, **kwargs):
            started = time.perf_counter()
            output = orig_generate(*args, **kwargs)
            if cuda:
                _cuda_synchronize()
            parts["stt_tokens_ms"] = (time.perf_counter() - started) * 1000
            return output

        self.model.feature_extractor = _TimedFeatureExtractor(orig_extractor, parts)
        self.model.encode = encode
        self.model.generate_with_fallback = generate_with_fallback
        try:
            segments, _info = self.model.transcribe(pcm, **self._transcribe_kwargs())
            text, quality = _stt_segment_quality(segments)
            parts.update(quality)
        finally:
            self.model.feature_extractor = orig_extractor
            self.model.encode = orig_encode
            self.model.generate_with_fallback = orig_generate
        return {key: round(value, 3) if isinstance(value, float) else value for key, value in parts.items()}, text

    def _pulse_gpu(self) -> None:
        """Keep the GPU out of idle P-state during an active call without changing driver clocks."""
        if getattr(self, "device", None) != "cuda" or not hasattr(self.model, "transcribe"):
            return
        list(self.model.transcribe(np.zeros(1600, dtype=np.float32), **self._transcribe_kwargs())[0])
        _cuda_synchronize()
        self._last_infer_at = time.perf_counter()
        _query_gpu_status()



class UnavailableStt(SttProvider):
    name = "client-speech"

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        return ""


class LazyWhisperStt(SttProvider):
    name = "faster-whisper"

    def __init__(self) -> None:
        self.last_timings: dict[str, float] = {}

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import asyncio

        if _load_whisper.cache_info().currsize:
            provider = _load_whisper()
        else:
            loop = asyncio.get_running_loop()
            provider = await loop.run_in_executor(_stt_executor, _faster_whisper_provider)
        try:
            text = await provider.transcribe(audio_bytes, sample_rate)
            return text
        finally:
            self.last_timings = dict(getattr(provider, "last_timings", {}) or {})


def describe_server_stt() -> dict[str, str | None]:
    """Server STT capability. Live calls may still use browser speech."""
    provider = get_stt_provider()
    info: dict[str, str | None] = {
        "stt": provider.name,
        "stt_device": None,
        "stt_model": None,
        "stt_ready": "1" if stt_ready() else "0",
        "stt_impl": STT_IMPL,
    }
    if provider.name != "faster-whisper":
        return info
    settings = get_settings()
    info["stt_model"] = (settings.whisper_model or "base").strip() or "base"
    if _load_whisper.cache_info().currsize:
        info["stt_device"] = getattr(_load_whisper(), "device", None)
        return info
    device = settings.whisper_device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
    info["stt_device"] = device
    return info


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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_stt_executor, _faster_whisper_provider)


def stt_ready() -> bool:
    provider = get_stt_provider()
    if provider.name != "faster-whisper":
        return True
    if not _load_whisper.cache_info().currsize:
        return False
    return bool(getattr(_load_whisper(), "startup_warmed", False))


def _run_gpu_pulse(provider: FasterWhisperStt) -> None:
    global _pulse_queued
    try:
        if not _keep_alive or _user_stt.is_set():
            return
        provider._pulse_gpu()
    except Exception:
        logger.debug("GPU keepalive pulse skipped", exc_info=True)
    finally:
        with _pulse_lock:
            _pulse_queued = False


def submit_gpu_pulse(provider: FasterWhisperStt | None = None) -> bool:
    """Queue at most one keepalive pulse. Never blocks the caller or user STT."""
    global _pulse_queued
    if not _keep_alive or _user_stt.is_set():
        return False
    if provider is None:
        if not _load_whisper.cache_info().currsize:
            return False
        provider = _load_whisper()
    if getattr(provider, "device", None) != "cuda":
        return False
    if time.perf_counter() - getattr(provider, "_last_infer_at", 0) < 0.8:
        return False
    with _pulse_lock:
        if _pulse_queued:
            return False
        _pulse_queued = True
    try:
        _stt_executor.submit(_run_gpu_pulse, provider)
    except Exception:
        with _pulse_lock:
            _pulse_queued = False
        logger.debug("GPU keepalive pulse skipped", exc_info=True)
        return False
    return True


def _keep_loop() -> None:
    while not _keep_stop.wait(1.0):
        submit_gpu_pulse()


def start_stt_keepalive() -> None:
    global _keep_alive, _keep_thread
    if get_stt_provider().name != "faster-whisper":
        return
    if not get_settings().whisper_keepalive:
        logger.info("GPU keepalive disabled")
        return
    _keep_alive = True
    if _load_whisper.cache_info().currsize:
        _load_whisper()._keep_awake = True
    if _keep_thread and _keep_thread.is_alive():
        return
    _keep_stop.clear()
    _keep_thread = threading.Thread(target=_keep_loop, name="whisper-gpu-keep", daemon=True)
    _keep_thread.start()


def stop_stt_keepalive() -> None:
    global _keep_alive
    _keep_alive = False
    if _load_whisper.cache_info().currsize:
        _load_whisper()._keep_awake = False


async def wait_voice_runtime(timeout: float = 90) -> None:
    """Block a voice session until embeddings and Whisper have actually finished loading."""
    import asyncio

    from app.services.rag.embeddings import warm_embeddings

    await asyncio.wait_for(asyncio.to_thread(warm_embeddings), timeout=timeout)
    await asyncio.wait_for(warm_stt(), timeout=timeout)


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
