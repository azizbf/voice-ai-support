import io
import unittest
import wave

import numpy as np

from app.services.voice.stt import _pcm16k_from_wav


class WavDecodeTests(unittest.TestCase):
    def test_reads_16k_mono_pcm_in_memory(self):
        samples = (np.sin(np.linspace(0, 20, 16000)) * 20000).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())
        decoded = _pcm16k_from_wav(buf.getvalue())
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.shape[0], 16000)

    def test_rejects_non_wav(self):
        self.assertIsNone(_pcm16k_from_wav(b"not a wav file"))
