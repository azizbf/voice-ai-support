const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const ts = require('typescript');
const context = { exports: {} };
vm.runInNewContext(ts.transpileModule(fs.readFileSync('src/lib/micMeter.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, context);
const { rmsFromTimeDomain, MIC_SPEECH_RMS } = context.exports;

test('silence is below the speech threshold and a full-scale signal is above it', () => {
  const quiet = Uint8Array.from({ length: 8 }, () => 128);
  const loud = Uint8Array.from({ length: 8 }, () => 255);
  assert.equal(rmsFromTimeDomain(quiet), 0);
  assert.ok(rmsFromTimeDomain(loud) > MIC_SPEECH_RMS);
});
