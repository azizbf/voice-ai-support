from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from threading import Lock
from typing import Any, Iterator

logger = logging.getLogger("latency")


@dataclass
class LatencySample:
    tenant_id: str
    path: str
    speech_end_to_retrieval_ms: float | None = None
    retrieval_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_audio_ms: float | None = None
    total_ms: float | None = None
    retrieval_scores: list[float] = field(default_factory=list)
    chunks_retrieved: int = 0
    source_pages: list[dict[str, Any]] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class LatencyStore:
    def __init__(self, maxlen: int = 200) -> None:
        self._samples: deque[LatencySample] = deque(maxlen=maxlen)
        self._lock = Lock()

    def add(self, sample: LatencySample) -> None:
        with self._lock:
            self._samples.append(sample)
        logger.info("latency_sample %s", asdict(sample))

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._samples)[-n:]
        return [asdict(s) for s in items]


latency_store = LatencyStore()


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    box: dict[str, float] = {"start": time.perf_counter(), "elapsed_ms": 0.0}
    try:
        yield box
    finally:
        box["elapsed_ms"] = (time.perf_counter() - box["start"]) * 1000
