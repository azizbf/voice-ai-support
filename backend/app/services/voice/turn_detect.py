"""Silero VAD + Pipecat Smart Turn v3, CPU-only (no Pipecat runtime)."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from app.core.config import get_settings
from app.services.voice.whisper_features import N_SAMPLES, compute_whisper_log_mel_features

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_FRAME = 512
COMPLETE_THRESHOLD = 0.5
_MODEL_FILES = ("smart-turn-v3.2-cpu.onnx", "smart-turn-v3.2.onnx", "model.onnx")
_HF_REPO = "pipecat-ai/smart-turn-v3"

_cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart-turn-cpu")
_load_lock = threading.Lock()
_detector: "TurnDetector | None" = None
_load_attempted = False


class VadModel(Protocol):
    def reset(self) -> None: ...
    def prob(self, frame: np.ndarray) -> float: ...


@dataclass(frozen=True)
class TurnScore:
    probability: float
    elapsed_ms: float
    complete: bool


class EnergyVad:
    """Deterministic fallback used in tests and if Silero ONNX is missing."""

    def __init__(self, threshold: float = 0.015) -> None:
        self.threshold = threshold

    def reset(self) -> None:
        return

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + 1e-12))
        if rms >= self.threshold * 1.4:
            return 0.92
        if rms >= self.threshold:
            return 0.62
        return 0.08


class SileroVadOnnx:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = [item.name for item in self.session.get_inputs()]
        self._outputs = [item.name for item in self.session.get_outputs()]
        self._state: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self._state = {}
        for item in self.session.get_inputs():
            if item.name in ("input", "sr"):
                continue
            # Silero v5 LSTM state: (2, batch=1, 128)
            if item.name == "state" or (len(item.shape) == 3 and item.shape[0] == 2):
                self._state[item.name] = np.zeros((2, 1, 128), dtype=np.float32)
            elif item.name in ("h", "c"):
                self._state[item.name] = np.zeros((2, 1, 64), dtype=np.float32)
            else:
                self._state[item.name] = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        feeds: dict[str, np.ndarray] = {"input": x}
        if "sr" in self._inputs:
            feeds["sr"] = np.array(SAMPLE_RATE, dtype=np.int64)
        feeds.update(self._state)
        outs = self.session.run(self._outputs, feeds)
        probability = float(np.asarray(outs[0]).reshape(-1)[0])
        for name, value in zip(self._outputs[1:], outs[1:]):
            array = np.asarray(value, dtype=np.float32)
            if name == "stateN" and "state" in self._state:
                self._state["state"] = array
            elif name in self._state:
                self._state[name] = array
            elif name.endswith("N") and name[:-1] in self._state:
                self._state[name[:-1]] = array
        return probability


def _silero_onnx_path() -> Path | None:
    import importlib.util

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.origin:
        return None
    root = Path(spec.origin).resolve().parent
    matches = sorted(p for p in root.rglob("*.onnx") if "half" not in p.name.lower())
    if matches:
        return matches[0]
    fallback = sorted(root.rglob("*.onnx"))
    return fallback[0] if fallback else None


def truncate_audio_to_last_n_seconds(audio: np.ndarray, n_seconds: int = 8, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    max_samples = n_seconds * sample_rate
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size > max_samples:
        return x[-max_samples:]
    if x.size < max_samples:
        return np.pad(x, (max_samples - x.size, 0), mode="constant")
    return x


class SmartTurnOnnx:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, pcm: np.ndarray) -> TurnScore:
        started = time.perf_counter()
        audio = truncate_audio_to_last_n_seconds(pcm)
        log_mel = compute_whisper_log_mel_features(audio, do_normalize=True)
        features = np.expand_dims(log_mel, axis=0)
        outputs = self.session.run(None, {"input_features": features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        elapsed_ms = (time.perf_counter() - started) * 1000
        return TurnScore(probability=probability, elapsed_ms=elapsed_ms, complete=probability > COMPLETE_THRESHOLD)


def _model_dir() -> Path:
    path = get_settings().data_path / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _existing_model() -> Path | None:
    root = _model_dir()
    for name in _MODEL_FILES:
        candidate = root / name
        if candidate.exists() and candidate.stat().st_size > 1000:
            return candidate
    return None


def _download_smart_turn() -> Path | None:
    existing = _existing_model()
    if existing is not None:
        return existing
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        logger.warning("huggingface_hub missing; cannot download Smart Turn")
        return None
    last_error: Exception | None = None
    for name in _MODEL_FILES:
        try:
            downloaded = hf_hub_download(repo_id=_HF_REPO, filename=name, local_dir=_model_dir())
            path = Path(downloaded)
            if path.exists() and path.stat().st_size > 1000:
                return path
        except Exception as exc:
            last_error = exc
            continue
    logger.warning("Smart Turn model download failed: %s", last_error)
    return None


@dataclass
class TurnDetector:
    vad: VadModel
    predict_sync: Callable[[np.ndarray], TurnScore]

    def predict(self, pcm: np.ndarray) -> TurnScore:
        return self.predict_sync(pcm)

    def submit(self, pcm: np.ndarray):
        return _cpu_pool.submit(self.predict_sync, np.asarray(pcm, dtype=np.float32).copy())


def _build_detector() -> TurnDetector | None:
    model_path = _download_smart_turn()
    if model_path is None:
        return None
    try:
        smart = SmartTurnOnnx(model_path)
    except Exception:
        logger.exception("Failed to load Smart Turn ONNX from %s", model_path)
        return None
    vad: VadModel
    silero_path = _silero_onnx_path()
    if silero_path is not None:
        try:
            vad = SileroVadOnnx(silero_path)
            logger.info("Silero VAD ONNX loaded from %s", silero_path)
        except Exception:
            logger.exception("Silero VAD ONNX failed; using energy VAD")
            vad = EnergyVad()
    else:
        logger.warning("silero-vad package/ONNX not found; using energy VAD")
        vad = EnergyVad()
    dummy = np.zeros(N_SAMPLES, dtype=np.float32)
    dummy[N_SAMPLES // 2 :] = 0.02 * np.sin(2 * np.pi * 220 * np.arange(N_SAMPLES // 2) / SAMPLE_RATE)
    score = smart.predict(dummy)
    logger.info("Smart Turn CPU ready (%.1f ms dummy infer, p=%.3f)", score.elapsed_ms, score.probability)
    return TurnDetector(vad=vad, predict_sync=smart.predict)


def get_turn_detector() -> TurnDetector | None:
    global _detector, _load_attempted
    with _load_lock:
        if _load_attempted:
            return _detector
        _load_attempted = True
        _detector = _build_detector()
        return _detector


def reset_turn_detector_for_tests() -> None:
    global _detector, _load_attempted
    with _load_lock:
        _detector = None
        _load_attempted = False


def cpu_executor() -> ThreadPoolExecutor:
    return _cpu_pool
