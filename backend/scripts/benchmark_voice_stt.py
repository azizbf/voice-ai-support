"""Compare local CPU/CUDA STT on the same recording, excluding model load.

Run from backend: .venv/Scripts/python.exe -m scripts.benchmark_voice_stt recording.wav
Reports transcripts so speed can be evaluated alongside recognition quality.
"""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from app.core.config import get_settings
from app.services.voice.stt import FasterWhisperStt


def benchmark(paths, runs):
    clips = [(str(path), Path(path).read_bytes()) for path in paths]
    settings = get_settings()
    results = []
    for device in ('cpu', 'cuda'):
        with patch('app.services.voice.stt.get_settings', return_value=settings.model_copy(update={'whisper_device': device})):
            provider = FasterWhisperStt()
        for name, audio in clips:
            for run in range(runs):
                start = time.perf_counter()
                transcript = provider._transcribe_sync(audio)
                results.append({
                    'file': Path(name).name,
                    'device': device,
                    'run': run + 1,
                    'ms': round((time.perf_counter() - start) * 1000),
                    'text': transcript,
                })
        del provider
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+')
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error('--runs must be between 1 and 20')
    print(json.dumps(benchmark(args.audio, args.runs), ensure_ascii=True, indent=2))
