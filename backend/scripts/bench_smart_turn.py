"""Measure Smart Turn CPU overhead. Downloads the ONNX model on first run."""

from __future__ import annotations

import statistics
import time

import numpy as np

from app.services.voice.turn_detect import SAMPLE_RATE, get_turn_detector, truncate_audio_to_last_n_seconds
from app.services.voice.whisper_features import compute_whisper_log_mel_features


def _tone(seconds: float, amp: float = 0.2) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * 180 * t / SAMPLE_RATE)).astype(np.float32)


def main() -> None:
    detector = get_turn_detector()
    if detector is None:
        print("Smart Turn model unavailable (install onnxruntime + network for HuggingFace download).")
        return
    audio = truncate_audio_to_last_n_seconds(_tone(3.0))
    # Warm
    detector.predict(audio)
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        score = detector.predict(audio)
        times.append((time.perf_counter() - t0) * 1000)
    feat_times = []
    for _ in range(8):
        t0 = time.perf_counter()
        compute_whisper_log_mel_features(audio)
        feat_times.append((time.perf_counter() - t0) * 1000)
    print(f"predict_ms p50={statistics.median(times):.1f}  max={max(times):.1f}  last_p={score.probability:.3f}")
    print(f"logmel_ms  p50={statistics.median(feat_times):.1f}  max={max(feat_times):.1f}")
    print("device=CPU executor=smart-turn-cpu (separate from Whisper GPU)")


if __name__ == "__main__":
    main()
