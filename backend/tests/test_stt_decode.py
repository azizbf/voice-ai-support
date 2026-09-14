import gc
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.voice.stt import _decode_container_pcm16k, _decode_pcm16k, _decode_pcm16k_timed, _pcm_speech_stats


def _webm_opus(pcm: np.ndarray, rate: int = 16000) -> bytes:
    import av

    s16 = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=rate)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(s16.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


class ContainerDecodeTests(unittest.TestCase):
    def test_does_not_call_gc_collect(self):
        audio = _webm_opus(np.sin(np.linspace(0, 40, 16000)).astype(np.float32) * 0.2)
        calls: list[int] = []
        real_collect = gc.collect

        def spy(*args, **kwargs):
            calls.append(1)
            return real_collect(*args, **kwargs)

        with patch("gc.collect", spy), patch("faster_whisper.audio.gc.collect", spy):
            pcm, parts = _decode_container_pcm16k(audio)
        self.assertGreater(pcm.size, 0)
        self.assertEqual(calls, [])
        self.assertTrue(gc.isenabled())
        self.assertIn("stt_decode_codec_ms", parts)
        self.assertIn("stt_decode_alloc_ms", parts)

    def test_waveform_matches_faster_whisper_decode_audio(self):
        from faster_whisper.audio import decode_audio

        audio = _webm_opus(np.sin(np.linspace(0, 80, 32000)).astype(np.float32) * 0.25)
        stock = np.asarray(decode_audio(io.BytesIO(audio), sampling_rate=16000), dtype=np.float32)
        ours = _decode_pcm16k(audio)
        self.assertEqual(ours.shape, stock.shape)
        self.assertEqual(float(np.max(np.abs(ours - stock))), 0.0)

    def test_wav_path_skips_container_decoder(self):
        import wave

        samples = (np.sin(np.linspace(0, 20, 1600)) * 20000).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())
        with patch("app.services.voice.stt._decode_container_pcm16k") as container:
            pcm, parts = _decode_pcm16k_timed(buf.getvalue())
        container.assert_not_called()
        self.assertEqual(parts, {})
        self.assertEqual(pcm.shape[0], 1600)

    def test_gpu_base_transcripts_match_on_sample_clips(self):
        from app.core.config import get_settings
        from app.services.voice.stt import FasterWhisperStt
        from faster_whisper.audio import decode_audio

        clips = sorted((Path(__file__).resolve().parents[1] / "samples").glob("stt-gpu-*.mp3"))
        if len(clips) < 3:
            self.skipTest("sample clips not present")
        settings = get_settings()
        model_name = (settings.whisper_model or "base").strip() or "base"
        if model_name != "base":
            self.skipTest("Whisper Base required")
        with patch(
            "app.services.voice.stt.get_settings",
            return_value=settings.model_copy(update={"whisper_device": "cuda"}),
        ):
            provider = FasterWhisperStt()
        if provider.device != "cuda":
            self.skipTest("Whisper Base GPU required")
        for path in clips:
            data = path.read_bytes()
            stock = np.asarray(decode_audio(io.BytesIO(data), sampling_rate=16000), dtype=np.float32)
            ours = _decode_pcm16k(data)
            self.assertEqual(float(np.max(np.abs(ours - stock))), 0.0)
            kwargs = provider._transcribe_kwargs()

            def text(pcm):
                segments, _info = provider.model.transcribe(pcm, **kwargs)
                return " ".join(seg.text.strip() for seg in segments).strip()

            self.assertEqual(text(ours), text(stock))
            self.assertEqual(provider._transcribe_sync(data), text(ours))
            self.assertGreater(provider.last_timings["stt_audio_ms"], 0)
            self.assertEqual(provider.last_timings["stt_audio_ms"], round(1000.0 * ours.size / 16000))
            self.assertEqual(provider.last_timings["stt_startup_warmed"], 1)
            self.assertEqual(provider.last_timings["stt_gpu_sync"], 1)
            self.assertIn("stt_feat_ms", provider.last_timings)
            self.assertIn("stt_encode_ms", provider.last_timings)
            self.assertIn("stt_tokens_ms", provider.last_timings)


class SpeechStatsTests(unittest.TestCase):
    def test_measures_leading_and_trailing_silence_without_trimming(self):
        sr = 16000
        lead = np.zeros(int(0.3 * sr), dtype=np.float32)
        speech = (np.sin(np.linspace(0, 80, int(0.8 * sr))) * 0.2).astype(np.float32)
        trail = np.zeros(int(0.5 * sr), dtype=np.float32)
        pcm = np.concatenate([lead, speech, trail])
        stats = _pcm_speech_stats(pcm, sr)
        self.assertEqual(stats["stt_audio_ms"], 1600)
        self.assertAlmostEqual(stats["stt_lead_silence_ms"], 300, delta=20)
        self.assertAlmostEqual(stats["stt_trail_silence_ms"], 500, delta=20)
        self.assertAlmostEqual(stats["stt_speech_ms"], 800, delta=40)
        self.assertEqual(pcm.size, int(1.6 * sr))

    def test_all_silence_is_reported_as_lead(self):
        stats = _pcm_speech_stats(np.zeros(16000, dtype=np.float32))
        self.assertEqual(stats["stt_audio_ms"], 1000)
        self.assertEqual(stats["stt_speech_ms"], 0)
        self.assertEqual(stats["stt_lead_silence_ms"], 1000)
