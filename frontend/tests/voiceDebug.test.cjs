const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const ts = require('typescript');
const context = { exports: {} };
vm.runInNewContext(ts.transpileModule(fs.readFileSync('src/lib/voiceDebug.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, context);
const { updateDebugTurn, transportRemainder } = context.exports;

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

test('missing or incompatible timings are not presented as zero network delay', () => {
  assert.equal(transportRemainder({ stages: {} }), undefined);
  assert.equal(transportRemainder({ total: 100, serverFirstAudio: 500, stages: {
    endpointing: { duration: 10 }, preparation: { duration: 10 }, buffering: { duration: 10 },
  } }), undefined);
});
