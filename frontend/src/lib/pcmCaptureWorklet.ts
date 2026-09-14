/** AudioWorklet: downsample capture to 16 kHz mono float and post to the main thread. */

export const PCM_CAPTURE_WORKLET = `
class Pcm16kCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / 16000;
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || !channel.length) return true;
    const ratio = this._ratio;
    const outLen = Math.max(1, Math.round(channel.length / ratio));
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const i0 = Math.min(Math.floor(pos), channel.length - 1);
      const i1 = Math.min(i0 + 1, channel.length - 1);
      const frac = pos - i0;
      out[i] = channel[i0] * (1 - frac) + channel[i1] * frac;
    }
    this.port.postMessage(out, [out.buffer]);
    return true;
  }
}
registerProcessor("pcm16k-capture", Pcm16kCaptureProcessor);
`;
