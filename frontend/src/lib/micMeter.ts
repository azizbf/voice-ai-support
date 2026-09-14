export const MIC_SPEECH_RMS = 0.012;
export const MIC_HARD_SILENCE_RMS = 0.005;
export const MIC_TICK_MS = 80;

export type MicMeter = {
  stop: () => void;
  lastSpeechAt: () => number | undefined;
  heard: () => boolean;
};

export function rmsFromTimeDomain(samples: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = (samples[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / samples.length);
}

export function startMicMeter(
  audioCtx: AudioContext,
  stream: MediaStream,
  options?: { ignore?: () => boolean },
): MicMeter {
  const sourceNode = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  sourceNode.connect(analyser);
  const samples = new Uint8Array(analyser.fftSize);
  let alive = true;
  let heard = false;
  let lastSpeechAt: number | undefined;
  const ignore = options?.ignore;

  const tick = () => {
    if (!alive) return;
    analyser.getByteTimeDomainData(samples);
    if (!ignore?.() && rmsFromTimeDomain(samples) > MIC_SPEECH_RMS) {
      heard = true;
      lastSpeechAt = performance.now();
    }
    window.setTimeout(tick, MIC_TICK_MS);
  };
  tick();

  return {
    stop: () => {
      if (!alive) return;
      alive = false;
      try {
        sourceNode.disconnect();
        analyser.disconnect();
      } catch {
        /* already disconnected */
      }
    },
    lastSpeechAt: () => lastSpeechAt,
    heard: () => heard,
  };
}
