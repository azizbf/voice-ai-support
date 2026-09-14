import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from app.services.voice.stt import FasterWhisperStt


class DeviceTests(unittest.TestCase):
    def test_compressed_audio_is_decoded_before_inference(self):
        provider = FasterWhisperStt.__new__(FasterWhisperStt)
        pcm = np.zeros(1600, dtype=np.float32)
        seen = {}
        def infer(source, **kwargs):
            seen["source"] = source
            return iter([SimpleNamespace(text=" Bonjour. ")]), None
        provider.model = SimpleNamespace(transcribe=infer)
        with patch("app.services.voice.stt._decode_pcm16k_timed", return_value=(pcm, {})) as decode:
            self.assertEqual(provider._transcribe_sync(b"webm recording"), "Bonjour.")
        decode.assert_called_once_with(b"webm recording")
        self.assertIs(seen["source"], pcm)
        self.assertEqual(provider.last_timings["stt_samples"], 1600)
        self.assertIn("stt_decode_ms", provider.last_timings)
        self.assertIn("stt_infer_ms", provider.last_timings)

    def check_device(self, device, precision):
        settings = SimpleNamespace(whisper_model="base", whisper_cpu_threads=6, whisper_device=device)
        model = MagicMock()
        model.transcribe.return_value = (iter([]), None)
        with patch('app.services.voice.stt.get_settings', return_value=settings), \
             patch('app.services.voice.stt._prepare_cuda_libraries') as prepare, \
             patch('faster_whisper.WhisperModel', return_value=model) as create:
            FasterWhisperStt()
        create.assert_called_once_with('base', device=device, compute_type=precision, cpu_threads=6)
        self.assertEqual(prepare.call_count, int(device == 'cuda'))
        model.transcribe.assert_called_once()

    def test_cpu_keeps_int8(self):
        self.check_device('cpu', 'int8')

    def test_cuda_uses_float16_and_warms_model(self):
        self.check_device('cuda', 'float16')

    def test_describe_server_stt_reports_loaded_cuda_device(self):
        from app.services.voice import stt

        with patch.object(stt, "get_stt_provider", return_value=SimpleNamespace(name="faster-whisper")), \
             patch.object(stt, "_load_whisper") as loaded:
            loaded.cache_info.return_value = SimpleNamespace(currsize=1)
            loaded.return_value = SimpleNamespace(device="cuda")
            info = stt.describe_server_stt()
        self.assertEqual(info["stt"], "faster-whisper")
        self.assertEqual(info["stt_device"], "cuda")

    def test_describe_server_stt_omits_device_without_whisper(self):
        from app.services.voice import stt

        with patch.object(stt, "get_stt_provider", return_value=SimpleNamespace(name="client-speech")):
            info = stt.describe_server_stt()
        self.assertEqual(info["stt"], "client-speech")
        self.assertIsNone(info["stt_device"])

    def test_nvidia_smi_query_never_sets_clocks(self):
        from app.services.voice.stt import _NVIDIA_QUERY

        joined = " ".join(_NVIDIA_QUERY)
        self.assertIn("--query-gpu", joined)
        self.assertNotIn("lock-gpu-clocks", joined)
        self.assertNotIn("-lgc", joined)
        self.assertNotIn("--power-limit", joined)

    def test_prepare_cuda_libraries_prepends_process_path(self):
        import os
        import sysconfig
        from pathlib import Path

        from app.services.voice import stt

        if os.name != "nt":
            self.skipTest("Windows CUDA DLL search only")
        cublas_bin = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cublas" / "bin"
        if not cublas_bin.is_dir():
            self.skipTest("CUDA wheels not installed")
        previous_handles = list(stt._cuda_dll_handles)
        previous_path = os.environ.get("PATH", "")
        stt._cuda_dll_handles.clear()
        try:
            stt._prepare_cuda_libraries()
            self.assertTrue(stt._cuda_dll_handles)
            self.assertEqual(os.environ["PATH"].split(os.pathsep)[0], str(cublas_bin))
        finally:
            os.environ["PATH"] = previous_path
            stt._cuda_dll_handles[:] = previous_handles
