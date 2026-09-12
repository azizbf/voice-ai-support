import uuid
import shutil
from contextlib import contextmanager
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

from app.core.logging_metrics import LatencySample, LatencyStore
from app.models.schemas import ChunkRecord
from app.services.vectorstore.faiss_store import FaissVectorStore


@contextmanager
def index_fixture():
    root = Path(__file__).resolve().parent
    directory = root / f"index-fixture-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        assert directory.resolve().is_relative_to(root)
        shutil.rmtree(directory)


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_index_is_isolated_and_invalidated_after_replacement(self):
        with index_fixture() as directory:
            store = FaissVectorStore()
            store.settings = SimpleNamespace(data_path=Path(directory))
            def chunk(tenant, text):
                return ChunkRecord(chunk_id=text, tenant_id=tenant, doc_id="doc",
                                   document_name="faq", page=1, text=text, token_estimate=1)
            vector = np.array([[1, 0]], dtype=np.float32)
            await store.upsert("a", [chunk("a", "old")], vector.copy())
            await store.upsert("b", [chunk("b", "other")], vector.copy())
            with patch("faiss.read_index", wraps=faiss.read_index) as read:
                await store.preload("a")
                await store.search("a", vector[0], 1)
                self.assertEqual(read.call_count, 1)
                result = await store.search("b", vector[0], 1)
                self.assertEqual(result[0].chunk.tenant_id, "b")
                self.assertEqual(read.call_count, 2)
            await store.upsert("a", [chunk("a", "new content")], vector.copy())
            result = await store.search("a", vector[0], 1)
            self.assertEqual(result[0].chunk.text, "new content")
            await store.delete_tenant("a")
            with self.assertRaises(FileNotFoundError):
                await store.search("a", vector[0], 1)


class MetricsTests(unittest.TestCase):
    def test_percentiles_use_only_measured_values(self):
        store = LatencyStore()
        store.add(LatencySample(tenant_id="test", path="voice"))
        for value in range(1, 101):
            store.add(LatencySample(tenant_id="test", path="voice_playback", client_ttfa_ms=value))
        self.assertEqual(store.percentiles()["client_ttfa_ms"],
                         {"count": 100, "p50": 50, "p95": 95, "p99": 99})
