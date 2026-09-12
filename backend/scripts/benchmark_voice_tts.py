"""Measure first TTS bytes separately from full synthesis; optional regression gate.

Run from backend: .venv/Scripts/python.exe -m scripts.benchmark_voice_tts --runs 5
These are provider timings, not microphone-to-speaker timings.
"""
import argparse
import asyncio
import json
import math
import time

from app.services.voice.tts import get_tts_provider


async def benchmark(runs):
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        first = None
        size = 0
        async for audio in get_tts_provider().synthesize_stream(
            "Pour redémarrer votre routeur, débranchez son alimentation pendant trente secondes, puis rebranchez-la."
        ):
            if first is None:
                first = (time.perf_counter() - start) * 1000
            size += len(audio)
        if first is None:
            raise RuntimeError("Provider returned no audio")
        samples.append({"first_audio_ms": round(first, 1),
                        "complete_audio_ms": round((time.perf_counter() - start) * 1000, 1), "bytes": size})
    values = sorted(sample["first_audio_ms"] for sample in samples)
    return {"samples": samples, "first_audio_ms": {
        f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)] for p in (50, 95, 99)
    }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-p95-ms", type=float)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100:
        parser.error("--runs must be between 1 and 100")
    result = asyncio.run(benchmark(args.runs))
    print(json.dumps(result, indent=2))
    if args.max_p95_ms is not None and result["first_audio_ms"]["p95"] > args.max_p95_ms:
        raise SystemExit(1)
