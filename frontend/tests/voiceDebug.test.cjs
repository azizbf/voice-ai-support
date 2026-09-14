const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const ts = require('typescript');
const context = { exports: {} };
vm.runInNewContext(ts.transpileModule(fs.readFileSync('src/lib/voiceDebug.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, context);
const { updateDebugTurn, transportRemainder, llmLabel, sttRouteLabel, elapsedFromSpeechEnd, medianMs, availableTimings, applySttSubstageTimings, silenceShouldEnd } = context.exports;

test('late server timings merge into the correct completed turn', () => {
  const turns = [
    { id: 'new', status: 'pending', stages: {} },
    { id: 'old', status: 'playing', total: 2500, stages: { buffering: { duration: 100 } } },
  ];
  const result = updateDebugTurn(turns, 'old', { stages: { tts: { duration: 350 } } });
  assert.equal(result[0], turns[0]);
  assert.equal(result[1].total, 2500);
  assert.equal(result[1].status, 'playing');
  assert.equal(result[1].stages.buffering.duration, 100);
  assert.equal(result[1].stages.tts.duration, 350);
  assert.equal(turns[1].stages.tts, undefined);
});

test('residual uses server first audio, not a sum of overlapping stages', () => {
  assert.equal(transportRemainder({ total: 3000, serverFirstAudio: 2200, stages: {
    endpointing: { duration: 550 }, preparation: { duration: 20 }, buffering: { duration: 150 },
    tts: { duration: 900 }, llm_phrase: { duration: 900 },
  } }), 80);
});

test('llm label names Gemini and Claude from the provider and model', () => {
  assert.equal(llmLabel('gemini', 'gemini-3.1-flash-lite'), 'Gemini · gemini-3.1-flash-lite');
  assert.equal(llmLabel('anthropic', 'claude-haiku-4-5'), 'Claude · claude-haiku-4-5');
  assert.equal(llmLabel(), undefined);
});

test('browser turns are never labeled as CUDA Whisper', () => {
  assert.equal(sttRouteLabel({ input: 'browser', sttProvider: 'faster-whisper', sttDevice: 'cuda' }), 'Navigateur · Web Speech');
  assert.equal(sttRouteLabel({ input: 'audio', sttProvider: 'faster-whisper', sttDevice: 'cuda' }), 'faster-whisper · cuda');
  const patched = updateDebugTurn(
    [{ id: 't1', input: 'browser', sttProvider: 'browser-speech', stages: {} }],
    't1',
    { sttProvider: 'faster-whisper', sttDevice: 'cuda' },
  );
  assert.equal(patched[0].sttProvider, 'browser-speech');
  assert.equal(patched[0].sttDevice, undefined);
  assert.equal(sttRouteLabel(patched[0]), 'Navigateur · Web Speech');
});

test('speech-end timings are omitted when the microphone origin is missing', () => {
  assert.equal(elapsedFromSpeechEnd(undefined, 1000), undefined);
  assert.equal(elapsedFromSpeechEnd(800, 500), undefined);
  assert.equal(elapsedFromSpeechEnd(800, 1100), 300);
  assert.equal(medianMs(availableTimings([
    { speechEndToAudio: 900 },
    { speechEndToAudio: undefined },
    { speechEndToAudio: 1100 },
    { speechEndToAudio: 1000 },
  ], 'speechEndToAudio')), 1000);
});

test('STT substages are sequential instead of sharing the STT start offset', () => {
  const stages = applySttSubstageTimings(12, {
    stt_queue_ms: 0,
    stt_decode_ms: 19,
    stt_decode_codec_ms: 16,
    stt_decode_alloc_ms: 0,
    stt_infer_ms: 249,
    stt_feat_ms: 2,
    stt_encode_ms: 220,
    stt_tokens_ms: 27,
  });
  assert.equal(stages.stt_queue.duration, 0);
  assert.equal(stages.stt_queue.offset, 12);
  assert.equal(stages.stt_decode.duration, 19);
  assert.equal(stages.stt_decode.offset, 12);
  assert.equal(stages.stt_decode_codec.duration, 16);
  assert.equal(stages.stt_decode_codec.offset, 12);
  assert.equal(stages.stt_decode_alloc.duration, 0);
  assert.equal(stages.stt_decode_alloc.offset, 28);
  assert.equal(stages.stt_infer.duration, 249);
  assert.equal(stages.stt_infer.offset, 31);
  assert.equal(stages.stt_feat.duration, 2);
  assert.equal(stages.stt_feat.offset, 31);
  assert.equal(stages.stt_encode.duration, 220);
  assert.equal(stages.stt_encode.offset, 33);
  assert.equal(stages.stt_tokens.duration, 27);
  assert.equal(stages.stt_tokens.offset, 253);
});

test('missing or incompatible timings are not presented as zero network delay', () => {
  assert.equal(transportRemainder({ stages: {} }), undefined);
  assert.equal(transportRemainder({ total: 100, serverFirstAudio: 500, stages: {
    endpointing: { duration: 10 }, preparation: { duration: 10 }, buffering: { duration: 10 },
  } }), undefined);
});

test('300ms endpointing is not used for short or hesitant pauses', () => {
  assert.equal(silenceShouldEnd({ heardMs: 600, silenceMs: 300, hardSilenceMs: 300 }), false);
  assert.equal(silenceShouldEnd({ heardMs: 2000, silenceMs: 300, hardSilenceMs: 0 }), false);
  assert.equal(silenceShouldEnd({ heardMs: 2000, silenceMs: 300, hardSilenceMs: 300 }), true);
  assert.equal(silenceShouldEnd({ heardMs: 400, silenceMs: 450, hardSilenceMs: 450 }), true);
});
