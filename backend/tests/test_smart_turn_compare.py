"""Compare Smart Turn state machine vs the current 300/450 ms silence timeout."""

from __future__ import annotations

import asyncio
import unittest

import numpy as np

from app.services.voice.pcm_endpoint import PcmTurnController, TurnCommit
from app.services.voice.turn_detect import EnergyVad, SAMPLE_RATE, TurnScore, VAD_FRAME


def silence_should_end(heard_ms: float, silence_ms: float, hard_silence_ms: float, cap: float = 450, fast: float = 300) -> bool:
    if heard_ms < 200:
        return False
    if silence_ms >= cap:
        return True
    return heard_ms >= 1200 and hard_silence_ms >= fast


def _tone(samples: int, amp: float = 0.25, freq: float = 180.0) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t / SAMPLE_RATE)).astype(np.float32)


def scenario_pcm(name: str) -> tuple[np.ndarray, int]:
    """Return 16 kHz PCM and the sample index where the true user turn ends."""
    sr = SAMPLE_RATE

    def cat(*parts: np.ndarray) -> np.ndarray:
        return np.concatenate(parts)

    if name == "complete_question":
        speech = _tone(int(0.9 * sr), amp=0.28)
        silence = np.zeros(int(0.8 * sr), dtype=np.float32)
        audio = cat(speech, silence)
        return audio, speech.size
    if name == "hesitation":
        a = _tone(int(1.3 * sr), amp=0.28)
        gap = np.zeros(int(0.48 * sr), dtype=np.float32)
        b = _tone(int(0.8 * sr), amp=0.28, freq=220)
        tail = np.zeros(int(0.8 * sr), dtype=np.float32)
        audio = cat(a, gap, b, tail)
        return audio, a.size + gap.size + b.size
    if name == "quiet_speech":
        speech = _tone(int(0.9 * sr), amp=0.02)
        silence = np.zeros(int(0.8 * sr), dtype=np.float32)
        return cat(speech, silence), speech.size
    if name == "numbers":
        parts: list[np.ndarray] = []
        for i in range(10):
            parts.append(_tone(int(0.16 * sr), amp=0.3, freq=160 + i * 15))
            parts.append(np.zeros(int(0.48 * sr), dtype=np.float32))
        parts.append(np.zeros(int(0.5 * sr), dtype=np.float32))
        audio = cat(*parts)
        last_digit_end = sum(p.size for p in parts[:-2])  # drop trailing 320 ms gap + extra silence
        return audio, last_digit_end
    if name == "background_noise":
        noise = (0.008 * np.random.default_rng(0).standard_normal(int(2.5 * sr))).astype(np.float32)
        speech = _tone(int(0.9 * sr), amp=0.25)
        noise[int(0.2 * sr) : int(0.2 * sr) + speech.size] += speech
        true_end = int(0.2 * sr) + speech.size
        return noise, true_end
    raise KeyError(name)


def simulate_fixed(pcm: np.ndarray, tick_ms: int = 80) -> dict:
    tick = int(tick_ms * SAMPLE_RATE / 1000)
    heard = silence = hard = 0.0
    speaking = False
    for i in range(0, pcm.size, tick):
        frame = pcm[i : i + tick]
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-12))
        if rms > 0.012:
            speaking = True
            heard += tick_ms
            silence = 0.0
            hard = 0.0
        elif speaking:
            silence += tick_ms
            hard += tick_ms if rms < 0.005 else 0
            if silence_should_end(heard, silence, hard):
                commit_at = min(pcm.size, i + tick)
                return {
                    "committed": True,
                    "endpointing_ms": silence,
                    "commit_sample": commit_at,
                }
    return {"committed": False, "endpointing_ms": None, "commit_sample": None}


async def simulate_smart(pcm: np.ndarray, complete_after: int) -> dict:
    commits: list[TurnCommit] = []

    def predict(audio: np.ndarray) -> TurnScore:
        # Incomplete while the buffered turn still ends before the remaining speech.
        span = int(np.asarray(audio).size)
        p = 0.88 if span >= complete_after - int(0.05 * SAMPLE_RATE) else 0.18
        return TurnScore(probability=p, elapsed_ms=4.0, complete=p > 0.5)

    async def on_commit(commit: TurnCommit) -> None:
        commits.append(commit)

    ctrl = PcmTurnController(EnergyVad(threshold=0.012), predict, on_commit)
    step = VAD_FRAME * 2
    for i in range(0, pcm.size, step):
        chunk = pcm[i : i + step]
        s16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        ctrl.feed(s16)
        await asyncio.sleep(0)
        if commits:
            break
    await asyncio.sleep(0.05)
    ctrl.close()
    if not commits:
        return {"committed": False, "endpointing_ms": None, "commit_sample": None, "reason": None}
    commit = commits[0]
    return {
        "committed": True,
        "endpointing_ms": commit.endpointing_ms,
        "commit_sample": commit.pcm.size,  # includes preroll; compared via premature flag
        "reason": commit.reason,
        "pause_ms": commit.pause_ms,
        "pcm_samples": commit.pcm.size,
    }


def premature(commit_sample: int | None, true_end: int, pcm_len: int) -> bool:
    if commit_sample is None:
        return True
    # Fixed-timeout reports absolute index; smart reports captured samples with preroll.
    return commit_sample < true_end - int(0.12 * SAMPLE_RATE)


class SmartTurnCompareTests(unittest.IsolatedAsyncioTestCase):
    async def test_matrix_against_fixed_timeout(self):
        rows = []
        for name in ("complete_question", "hesitation", "quiet_speech", "numbers", "background_noise"):
            pcm, true_end = scenario_pcm(name)
            fixed = simulate_fixed(pcm)
            smart = await simulate_smart(pcm, true_end)
            if name == "hesitation" or name == "numbers":
                fixed_cut = premature(fixed["commit_sample"], true_end, pcm.size)
            else:
                fixed_cut = premature(fixed["commit_sample"], true_end, pcm.size) if fixed["committed"] else False
            smart_cut = False
            if smart["committed"] and name in ("hesitation", "numbers"):
                smart_cut = smart["pcm_samples"] < true_end - int(0.12 * SAMPLE_RATE)
            rows.append((name, fixed, smart, fixed_cut, smart_cut, true_end))

        hesitation = next(r for r in rows if r[0] == "hesitation")
        self.assertTrue(hesitation[3], "fixed timeout should cut the mid-sentence pause")
        self.assertFalse(hesitation[4], "smart turn should keep the turn across hesitation")

        numbers = next(r for r in rows if r[0] == "numbers")
        self.assertTrue(numbers[3], "fixed timeout should cut between spoken numbers")
        self.assertFalse(numbers[4], "smart turn should wait through number gaps")

        complete = next(r for r in rows if r[0] == "complete_question")
        self.assertTrue(complete[1]["committed"] and complete[2]["committed"])
        self.assertFalse(complete[4])
        self.assertLessEqual(complete[2]["endpointing_ms"], complete[1]["endpointing_ms"])

        quiet = next(r for r in rows if r[0] == "quiet_speech")
        self.assertTrue(quiet[1]["committed"] or quiet[2]["committed"])
        self.assertTrue(quiet[2]["committed"])

        noise = next(r for r in rows if r[0] == "background_noise")
        self.assertTrue(noise[2]["committed"])

        print("\nendpointing comparison (synthetic 16 kHz PCM)")
        print(f"{'scenario':<20} {'fixed_cut':<10} {'smart_cut':<10} {'fixed_ms':<10} {'smart_ms':<10} {'reason'}")
        for name, fixed, smart, fcut, scut, _true in rows:
            print(
                f"{name:<20} {str(fcut):<10} {str(scut):<10} "
                f"{str(fixed['endpointing_ms']):<10} {str(None if not smart['committed'] else round(smart['endpointing_ms'])):<10} "
                f"{smart.get('reason')}"
            )


if __name__ == "__main__":
    unittest.main()
