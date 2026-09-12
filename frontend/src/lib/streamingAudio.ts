/** Incremental MP3 playback, with buffered playback for browsers without MSE. */
export class StreamingAudio {
  private audio = new Audio();
  private media: MediaSource | null = null;
  private source: SourceBuffer | null = null;
  private pending: ArrayBuffer[] = [];
  private chunks: ArrayBuffer[] = [];
  private ended = false;
  private settled = false;
  private started = false;
  private url = "";
  private timer: ReturnType<typeof setTimeout> | undefined;
  private resolve!: () => void;
  private reject!: (error: Error) => void;
  readonly done: Promise<void>;

  constructor(onPlaying: () => void) {
    this.done = new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
    // The caller awaits completion at tts_end; handle early playback failures too.
    void this.done.catch(() => undefined);
    this.audio.onplaying = () => {
      if (!this.started) { this.started = true; onPlaying(); }
      this.armTimeout();
    };
    this.audio.ontimeupdate = () => this.armTimeout();
    this.audio.onended = () => this.finish();
    this.audio.onerror = () => this.fallback();
    if (typeof MediaSource !== "undefined" && MediaSource.isTypeSupported("audio/mpeg")) {
      this.media = new MediaSource();
      this.media.addEventListener("sourceopen", () => {
        if (this.settled || !this.media) return;
        try {
          this.source = this.media.addSourceBuffer("audio/mpeg");
          // Consecutive TTS phrases share one timeline and one playback session.
          this.source.mode = "sequence";
          this.source.addEventListener("updateend", () => this.pump());
          this.source.addEventListener("error", () => this.fallback());
          this.pump();
        } catch { this.fallback(); }
      }, { once: true });
      this.url = URL.createObjectURL(this.media);
      this.audio.src = this.url;
      void this.audio.play().catch(() => this.fallback());
    }
    this.armTimeout();
  }

  push(data: ArrayBuffer) {
    if (this.settled || this.ended) return;
    this.chunks.push(data);
    if (this.media) this.pending.push(data);
    this.armTimeout();
    this.pump();
  }

  end() {
    if (this.settled) return this.done;
    this.ended = true;
    if (!this.chunks.length) this.finish(new Error("Aucun audio reçu."));
    else if (this.media) this.pump();
    else this.playBuffered();
    return this.done;
  }

  stop() { this.finish(); }

  private pump() {
    if (this.settled || !this.source || this.source.updating || this.media?.readyState !== "open") return;
    try {
      const next = this.pending.shift();
      if (next) this.source.appendBuffer(next);
      else if (this.ended) this.media.endOfStream();
    } catch { this.fallback(); }
  }

  private detach() {
    this.source = null;
    this.media = null;
    this.pending = [];
    this.audio.pause();
    this.audio.removeAttribute("src");
    this.audio.load();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = "";
  }

  private fallback() {
    if (this.settled) return;
    // Never replay words already heard after a mid-stream failure.
    if (this.started || !this.media) { this.finish(new Error("Lecture audio interrompue.")); return; }
    this.detach();
    if (this.ended) this.playBuffered();
  }

  private playBuffered() {
    this.url = URL.createObjectURL(new Blob(this.chunks, { type: "audio/mpeg" }));
    this.audio.src = this.url;
    void this.audio.play().catch(() => this.finish(new Error("Autorisez la lecture audio dans le navigateur.")));
  }

  private armTimeout() {
    if (this.settled) return;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.finish(new Error("Le flux audio a expiré.")), 20000);
  }

  private finish(error?: Error) {
    if (this.settled) return;
    this.settled = true;
    clearTimeout(this.timer);
    this.audio.onplaying = this.audio.ontimeupdate = this.audio.onended = this.audio.onerror = null;
    this.detach();
    this.chunks = [];
    if (error) this.reject(error); else this.resolve();
  }
}
