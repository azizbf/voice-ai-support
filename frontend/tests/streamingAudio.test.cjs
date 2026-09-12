const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const ts = require('typescript');

const source = ts.transpileModule(fs.readFileSync('src/lib/streamingAudio.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

function setup(supported = true) {
  let audio, media, revoked = 0, played = 0;
  class Events {
    listeners = {};
    addEventListener(name, fn) { this.listeners[name] = fn; }
    emit(name) { this.listeners[name]?.(); }
  }
  class FakeSource extends Events {
    updating = false;
    appended = [];
    appendBuffer(data) { assert.equal(this.updating, false); this.appended.push(data); this.updating = true; }
    complete() { this.updating = false; this.emit('updateend'); }
  }
  class FakeMedia extends Events {
    static isTypeSupported() { return supported; }
    readyState = 'closed';
    source = new FakeSource();
    constructor() { super(); media = this; }
    addSourceBuffer() { return this.source; }
    endOfStream() { assert.equal(this.source.updating, false); this.readyState = 'ended'; }
    open() { this.readyState = 'open'; this.emit('sourceopen'); }
  }
  class FakeAudio {
    constructor() { audio = this; }
    play() { played++; return Promise.resolve(); }
    pause() {}
    load() {}
    removeAttribute() {}
  }
  const context = { exports: {}, Audio: FakeAudio, MediaSource: FakeMedia, Blob,
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => revoked++ }, setTimeout, clearTimeout };
  vm.runInNewContext(source, context);
  return { Player: context.exports.StreamingAudio, get audio() { return audio; },
    get media() { return media; }, get revoked() { return revoked; }, get played() { return played; } };
}

test('starts before end, serializes appends and drains before closing', async () => {
  const env = setup(); let started = 0;
  const p = new env.Player(() => started++);
  p.push(new ArrayBuffer(3)); p.push(new ArrayBuffer(4));
  env.media.open();
  assert.equal(env.media.source.appended.length, 1);
  env.audio.onplaying();
  assert.equal(started, 1);
  const done = p.end();
  env.media.source.complete();
  assert.equal(env.media.source.appended.length, 2);
  assert.equal(env.media.readyState, 'open');
  env.media.source.complete();
  assert.equal(env.media.readyState, 'ended');
  env.audio.onended(); await done;
  assert.equal(env.revoked, 1);
});

test('unsupported browsers buffer until end', async () => {
  const env = setup(false); const p = new env.Player(() => {});
  p.push(new ArrayBuffer(3)); assert.equal(env.played, 0);
  const done = p.end(); assert.equal(env.played, 1);
  env.audio.onended(); await done;
});

test('stop settles playback and ignores late data', async () => {
  const env = setup(); const p = new env.Player(() => {});
  p.stop(); p.push(new ArrayBuffer(4)); env.media.open(); await p.done;
  assert.equal(env.media.source.appended.length, 0);
  assert.equal(env.revoked, 1);
});

test('midstream errors do not replay already spoken words', async () => {
  const env = setup(); const p = new env.Player(() => {});
  p.push(new ArrayBuffer(4)); env.media.open(); env.audio.onplaying(); env.audio.onerror();
  await assert.rejects(p.done, /interrompue/); assert.equal(env.played, 1);
});

test('MSE failure before playback falls back to one complete MP3', async () => {
  const env = setup(); const p = new env.Player(() => {});
  p.push(new ArrayBuffer(4)); env.media.open(); env.media.source.emit('error');
  const done = p.end(); assert.equal(env.played, 2);
  env.audio.onended(); await done;
});
