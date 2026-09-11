"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { synthesizeSpeech, voiceWsUrl, type Source } from "@/lib/api";

type VoiceState = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error";

type TranscriptItem = { id: string; role: "user" | "assistant"; text: string };

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((ev: unknown) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

function getSpeechRecognition(): (new () => SpeechRecognitionLike) | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  };
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

/** Convert recorded browser audio to 16kHz mono WAV for Whisper. */
async function blobToWav(blob: Blob): Promise<ArrayBuffer> {
  const audioCtx = new AudioContext();
  try {
    const decoded = await audioCtx.decodeAudioData(await blob.arrayBuffer());
    const targetRate = 16000;
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetRate), targetRate);
    const src = offline.createBufferSource();
    // Downmix to mono
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    const ch0 = mono.getChannelData(0);
    const channels = decoded.numberOfChannels;
    for (let i = 0; i < decoded.length; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += decoded.getChannelData(c)[i];
      ch0[i] = sum / channels;
    }
    src.buffer = mono;
    src.connect(offline.destination);
    src.start(0);
    const rendered = await offline.startRendering();
    const pcm = rendered.getChannelData(0);
    const samples = pcm.length;
    const buffer = new ArrayBuffer(44 + samples * 2);
    const view = new DataView(buffer);
    const writeStr = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + samples * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, samples * 2, true);
    let offset = 44;
    for (let i = 0; i < samples; i++) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
    return buffer;
  } finally {
    await audioCtx.close();
  }
}

export function useVoiceCall(tenantId: string | null, token: string | null) {
  const [state, setState] = useState<VoiceState>("idle");
  const [muted, setMuted] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [interim, setInterim] = useState("");
  const [holding, setHolding] = useState(false);
  const [serverStt, setServerStt] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const playingRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const nextPlayTimeRef = useRef(0);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const timerRef = useRef<number | null>(null);
  const startedAtRef = useRef<number | null>(null);
  const activeRef = useRef(false);
  const mutedRef = useRef(false);
  const mimeRef = useRef("audio/webm");
  const holdModeRef = useRef<"speech" | "media" | null>(null);
  const speechFinalRef = useRef("");
  const serverSttRef = useRef(false);
  const ttsChunksRef = useRef<Uint8Array[]>([]);
  const ttsReceivingRef = useRef(false);
  const browserSpeakRef = useRef<SpeechSynthesisUtterance | null>(null);
  const speakQueueRef = useRef<string[]>([]);
  const speakBusyRef = useRef(false);
  const usedSpeakEventsRef = useRef(false);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  const startContinuousListenRef = useRef<() => void>(() => undefined);
  const RESUME_MS = 60;
  const SILENCE_MS = 850;

  const ensureAudioCtx = useCallback(async () => {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new Ctx();
    }
    if (audioCtxRef.current.state === "suspended") {
      await audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  const stopBrowserSpeak = useCallback(() => {
    try {
      window.speechSynthesis?.cancel();
    } catch {
      /* ignore */
    }
    browserSpeakRef.current = null;
  }, []);

  const speakBrowser = useCallback(
    (text: string): Promise<boolean> => {
      const cleaned = text.trim();
      if (!cleaned || typeof window === "undefined" || !window.speechSynthesis) {
        return Promise.resolve(false);
      }
      stopBrowserSpeak();
      const utter = new SpeechSynthesisUtterance(cleaned);
      utter.lang = "fr-FR";
      utter.rate = 0.95;
      const voices = window.speechSynthesis.getVoices();
      const fr =
        voices.find((v) => v.lang.toLowerCase().startsWith("fr-fr")) ||
        voices.find((v) => v.lang.toLowerCase().startsWith("fr"));
      if (fr) utter.voice = fr;
      browserSpeakRef.current = utter;
      playingRef.current = true;
      setState("speaking");
      return new Promise((resolve) => {
        utter.onend = () => {
          if (browserSpeakRef.current === utter) {
            browserSpeakRef.current = null;
            playingRef.current = false;
            setState((prev) => (prev === "speaking" ? "listening" : prev));
          }
          resolve(true);
        };
        utter.onerror = () => {
          if (browserSpeakRef.current === utter) {
            browserSpeakRef.current = null;
            playingRef.current = false;
            setState((prev) => (prev === "speaking" ? "listening" : prev));
          }
          resolve(false);
        };
        window.speechSynthesis.speak(utter);
      });
    },
    [stopBrowserSpeak],
  );

  const stopPlayback = useCallback(() => {
    stopBrowserSpeak();
    audioQueueRef.current = [];
    playingRef.current = false;
    nextPlayTimeRef.current = 0;
    for (const src of activeSourcesRef.current) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    activeSourcesRef.current = [];
    if (audioElRef.current) {
      try {
        audioElRef.current.onended = null;
        audioElRef.current.onerror = null;
        audioElRef.current.pause();
        const prev = audioElRef.current.getAttribute("src");
        audioElRef.current.removeAttribute("src");
        audioElRef.current.load();
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      } catch {
        /* ignore */
      }
    }
  }, [stopBrowserSpeak]);

  const pauseRecognition = useCallback(() => {
    try {
      recognitionRef.current?.stop();
    } catch {
      /* ignore */
    }
  }, []);

  /** Play one complete MP3 and resolve only when playback finishes. */
  const playCompleteAudio = useCallback(
    async (data: ArrayBuffer) => {
      if (!data || data.byteLength < 64) {
        setError("Audio TTS trop court / vide.");
        return;
      }
      stopBrowserSpeak();
      pauseRecognition();
      playingRef.current = true;
      setState("speaking");

      if (audioElRef.current) {
        try {
          audioElRef.current.pause();
        } catch {
          /* ignore */
        }
      }
      const blob = new Blob([data], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioElRef.current = audio;
      audio.volume = 1;
      audio.preload = "auto";

      await new Promise<void>((resolve) => {
        let settled = false;
        const finish = () => {
          if (settled) return;
          settled = true;
          try {
            URL.revokeObjectURL(url);
          } catch {
            /* ignore */
          }
          if (audioElRef.current === audio) audioElRef.current = null;
          playingRef.current = false;
          setState((prev) => (prev === "speaking" ? "listening" : prev));
          resolve();
        };

        audio.onended = () => finish();
        audio.onerror = () => {
          setError("Lecture MP3 impossible — voix navigateur utilisée si disponible.");
          finish();
        };

        void (async () => {
          try {
            await ensureAudioCtx();
            await audio.play();
          } catch (err) {
            setError(
              err instanceof Error
                ? `Lecture audio bloquée: ${err.message}`
                : "Lecture audio bloquée par le navigateur.",
            );
            finish();
          }
        })();
      });
    },
    [ensureAudioCtx, pauseRecognition, stopBrowserSpeak],
  );

  const flushTtsChunks = useCallback(async () => {
    const chunks = ttsChunksRef.current;
    ttsChunksRef.current = [];
    ttsReceivingRef.current = false;
    if (!chunks.length) return;

    const total = chunks.reduce((n, c) => n + c.byteLength, 0);
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.byteLength;
    }

    usedSpeakEventsRef.current = true;
    speakBusyRef.current = true;
    holdModeRef.current = null;
    setHolding(false);
    try {
      await playCompleteAudio(merged.buffer.slice(merged.byteOffset, merged.byteOffset + merged.byteLength));
    } finally {
      speakBusyRef.current = false;
      if (activeRef.current && !mutedRef.current && !playingRef.current) {
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
      }
    }
  }, [playCompleteAudio]);

  const speakNeural = useCallback(
    async (text: string) => {
      const cleaned = text.trim();
      if (!cleaned || !tenantId || !token) return;
      stopBrowserSpeak();
      pauseRecognition();
      holdModeRef.current = null;
      setHolding(false);
      setState("speaking");
      playingRef.current = true;
      try {
        const buf = await synthesizeSpeech(tenantId, token, cleaned);
        await playCompleteAudio(buf);
      } catch (err) {
        const ok = await speakBrowser(cleaned);
        if (!ok) {
          playingRef.current = false;
          setState("listening");
          setError(
            err instanceof Error
              ? `Voix: ${err.message}`
              : "Impossible de lire la réponse à voix haute.",
          );
        }
      }
    },
    [pauseRecognition, playCompleteAudio, speakBrowser, stopBrowserSpeak, tenantId, token],
  );

  const drainSpeakQueue = useCallback(async () => {
    if (speakBusyRef.current) return;
    speakBusyRef.current = true;
    try {
      while (speakQueueRef.current.length && activeRef.current) {
        const next = speakQueueRef.current.shift();
        if (!next) continue;
        await speakNeural(next);
      }
    } finally {
      speakBusyRef.current = false;
      if (speakQueueRef.current.length && activeRef.current) {
        void drainSpeakQueue();
        return;
      }
      if (activeRef.current && !mutedRef.current && !playingRef.current) {
        setState((prev) => (prev === "speaking" ? "listening" : prev));
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
      }
    }
  }, [speakNeural]);

  const enqueueSpeak = useCallback(
    (text: string) => {
      const cleaned = text.trim();
      if (!cleaned) return;
      // Ignore duplicate speak for the same turn
      if (usedSpeakEventsRef.current && speakBusyRef.current) return;
      if (usedSpeakEventsRef.current && speakQueueRef.current[0] === cleaned) return;
      usedSpeakEventsRef.current = true;
      speakQueueRef.current = [cleaned];
      void drainSpeakQueue();
    },
    [drainSpeakQueue],
  );

  const sendUserText = useCallback((text: string) => {
    const cleaned = text.trim();
    if (!cleaned) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Appel non connecté. Cliquez Démarrer l'appel IA.");
      return;
    }
    if (playingRef.current) stopPlayback();
    setTranscript((prev) => [
      ...prev,
      { id: `${Date.now()}-u`, role: "user", text: cleaned },
    ]);
    setInterim("");
    ws.send(JSON.stringify({ type: "user_transcript", text: cleaned }));
  }, [stopPlayback]);

  const stopCapture = useCallback(() => {
    try {
      recognitionRef.current?.abort();
    } catch {
      /* ignore */
    }
    recognitionRef.current = null;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.onstop = null;
        recorder.stop();
      } catch {
        /* ignore */
      }
    }
    mediaRecorderRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
  }, []);

  const cleanupSession = useCallback(
    (nextState: VoiceState = "idle") => {
      activeRef.current = false;
      holdModeRef.current = null;
      speakQueueRef.current = [];
      speakBusyRef.current = false;
      usedSpeakEventsRef.current = false;
      setHolding(false);
      setInterim("");
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
      startedAtRef.current = null;
      setElapsed(0);
      stopPlayback();
      stopCapture();
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      }
      setState(nextState);
    },
    [stopCapture, stopPlayback],
  );

  const beginHoldSpeech = useCallback(() => {
    const Ctor = getSpeechRecognition();
    if (!Ctor || mutedRef.current || !activeRef.current || playingRef.current) return false;

    try {
      recognitionRef.current?.abort();
    } catch {
      /* ignore */
    }

    speechFinalRef.current = "";
    setInterim("● À l'écoute… parlez naturellement");
    const recognition = new Ctor();
    recognition.lang = "fr-FR";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognitionRef.current = recognition;

    let sendTimer: number | null = null;
    const armAutoSend = () => {
      if (sendTimer) window.clearTimeout(sendTimer);
      sendTimer = window.setTimeout(() => {
        if (!activeRef.current || playingRef.current || mutedRef.current) return;
        const text = speechFinalRef.current.trim();
        if (!text) return;
        speechFinalRef.current = "";
        setInterim("");
        holdModeRef.current = null;
        setHolding(false);
        try {
          recognition.stop();
        } catch {
          /* ignore */
        }
        recognitionRef.current = null;
        sendUserText(text);
      }, SILENCE_MS);
    };

    recognition.onresult = (ev: unknown) => {
      if (playingRef.current) return;
      const event = ev as {
        resultIndex: number;
        results: Array<{ isFinal: boolean; 0: { transcript: string } }>;
      };
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const piece = result[0].transcript;
        if (result.isFinal) {
          speechFinalRef.current = `${speechFinalRef.current} ${piece}`.trim();
          armAutoSend();
        } else {
          interimText += piece;
        }
      }
      setInterim(
        (speechFinalRef.current + (interimText ? ` ${interimText}` : "")).trim() ||
          "● À l'écoute…",
      );
    };

    recognition.onerror = (ev: unknown) => {
      const err = ev as { error?: string };
      if (!err.error || ["no-speech", "aborted"].includes(err.error)) return;
      if (err.error === "network") {
        serverSttRef.current = true;
        setServerStt(true);
        holdModeRef.current = null;
        setHolding(false);
        window.setTimeout(() => startContinuousListenRef.current(), 300);
        return;
      }
      setError(`Reconnaissance vocale: ${err.error}`);
    };

    recognition.onend = () => {
      if (sendTimer) window.clearTimeout(sendTimer);
      if (
        activeRef.current &&
        !mutedRef.current &&
        !playingRef.current &&
        holdModeRef.current === "speech"
      ) {
        window.setTimeout(() => {
          try {
            recognition.start();
          } catch {
            /* ignore */
          }
        }, 250);
      }
    };

    try {
      recognition.start();
      holdModeRef.current = "speech";
      setHolding(true);
      return true;
    } catch {
      return false;
    }
  }, [sendUserText]);

  const beginHoldMedia = useCallback(() => {
    const stream = mediaStreamRef.current;
    const ws = wsRef.current;
    if (!stream || !ws || ws.readyState !== WebSocket.OPEN || !activeRef.current) return false;
    if (mutedRef.current || playingRef.current) return false;

    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "";
    if (!mime) {
      setError("Enregistrement audio non supporté par ce navigateur.");
      return false;
    }
    mimeRef.current = mime;
    audioChunksRef.current = [];

    const recorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorderRef.current = recorder;

    const audioCtx = new AudioContext();
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    sourceNode.connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);

    let speaking = false;
    let silenceMs = 0;
    let heardMs = 0;
    let alive = true;
    const startedAt = performance.now();

    const teardownMeter = () => {
      alive = false;
      void audioCtx.close().catch(() => undefined);
    };

    const stopAndSend = () => {
      if (!alive) return;
      teardownMeter();
      if (recorder.state === "recording") {
        try {
          recorder.stop();
        } catch {
          /* ignore */
        }
      }
    };

    const tick = () => {
      if (!alive || !activeRef.current || mutedRef.current || playingRef.current) {
        stopAndSend();
        return;
      }
      analyser.getByteTimeDomainData(samples);
      let sum = 0;
      for (let i = 0; i < samples.length; i++) {
        const v = (samples[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / samples.length);
      const elapsed = performance.now() - startedAt;

      if (rms > 0.045) {
        speaking = true;
        heardMs += 80;
        silenceMs = 0;
        setInterim("● Je vous écoute…");
      } else if (speaking) {
        silenceMs += 80;
      }

      if ((speaking && silenceMs >= SILENCE_MS && heardMs >= 200) || elapsed > 10000) {
        stopAndSend();
        return;
      }
      window.setTimeout(tick, 80);
    };

    recorder.ondataavailable = (ev) => {
      if (ev.data.size > 0) audioChunksRef.current.push(ev.data);
    };

    recorder.onstop = () => {
      teardownMeter();
      const chunks = audioChunksRef.current;
      audioChunksRef.current = [];
      mediaRecorderRef.current = null;
      holdModeRef.current = null;
      setHolding(false);
      if (!chunks.length || !activeRef.current || ws.readyState !== WebSocket.OPEN) return;
      if (!speaking) {
        setInterim("● À l'écoute…");
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        return;
      }
      const blob = new Blob(chunks, { type: mimeRef.current });
      if (blob.size < 1500) {
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        return;
      }
      setState("thinking");
      setInterim("Transcription…");
      void blobToWav(blob)
        .then((buf) => {
          if (!activeRef.current || ws.readyState !== WebSocket.OPEN) return;
          ws.send(buf);
          ws.send(JSON.stringify({ type: "audio_end" }));
          setInterim("");
        })
        .catch(() => {
          void blob.arrayBuffer().then((buf) => {
            if (!activeRef.current || ws.readyState !== WebSocket.OPEN) return;
            ws.send(buf);
            ws.send(JSON.stringify({ type: "audio_end" }));
            setInterim("");
          });
        });
    };

    try {
      recorder.start(250);
      holdModeRef.current = "media";
      setHolding(true);
      setInterim("● À l'écoute… parlez naturellement");
      window.setTimeout(tick, 80);
      return true;
    } catch (err) {
      teardownMeter();
      setError(err instanceof Error ? err.message : "Impossible de démarrer le micro.");
      return false;
    }
  }, []);

  const startContinuousListen = useCallback(() => {
    if (!activeRef.current || mutedRef.current || playingRef.current) return;
    if (holdModeRef.current) return;
    setError(null);
    const ok = serverSttRef.current
      ? beginHoldMedia()
      : beginHoldSpeech() || beginHoldMedia();
    if (!ok) setError("Micro indisponible pour l'écoute continue.");
  }, [beginHoldMedia, beginHoldSpeech]);

  startContinuousListenRef.current = startContinuousListen;

  const endCall = useCallback(() => {
    cleanupSession("idle");
  }, [cleanupSession]);

  const startCall = useCallback(async () => {
    if (!tenantId || !token) {
      setError("Session demo manquante. Réimportez un PDF puis réessayez.");
      setState("error");
      return;
    }

    cleanupSession("connecting");
    setError(null);
    setTranscript([]);
    setSources([]);
    setInterim("");
    setState("connecting");
    activeRef.current = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      try {
        await ensureAudioCtx();
        // Warm browser voices (Chrome loads them async)
        window.speechSynthesis?.getVoices();
      } catch {
        /* autoplay unlock best-effort */
      }

      const wsUrl = voiceWsUrl(tenantId, token);
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      let opened = false;
      const isCurrentSocket = () => wsRef.current === ws && activeRef.current;

      ws.onmessage = (ev) => {
        if (!isCurrentSocket()) return;
        if (typeof ev.data !== "string") {
          if (ttsReceivingRef.current) {
            ttsChunksRef.current.push(new Uint8Array(ev.data as ArrayBuffer));
          }
          return;
        }
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === "ready") {
          const sttName = String(msg.stt || "client-speech");
          const hasServer = sttName !== "client-speech";
          serverSttRef.current = hasServer;
          setServerStt(hasServer);
          setState("listening");
          setError(null);
          window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        }
        if (msg.type === "tts_start") {
          ttsReceivingRef.current = true;
          ttsChunksRef.current = [];
          usedSpeakEventsRef.current = true;
          speakBusyRef.current = true;
          holdModeRef.current = null;
          setHolding(false);
          pauseRecognition();
          setState("speaking");
          return;
        }
        if (msg.type === "tts_end") {
          void flushTtsChunks();
          return;
        }
        if (msg.type === "status") {
          const s = msg.state as VoiceState | undefined;
          if (
            s === "listening" &&
            (playingRef.current ||
              ttsReceivingRef.current ||
              speakBusyRef.current ||
              speakQueueRef.current.length > 0)
          )
            return;
          if (s === "listening" || s === "thinking" || s === "speaking") {
            setState(s);
            if (s === "speaking" || s === "thinking") {
              holdModeRef.current = null;
              setHolding(false);
              pauseRecognition();
              const recorder = mediaRecorderRef.current;
              if (recorder && recorder.state === "recording") {
                try {
                  recorder.onstop = null;
                  recorder.stop();
                } catch {
                  /* ignore */
                }
                mediaRecorderRef.current = null;
              }
            }
            if (s === "listening") {
              window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
            }
          }
        }
        if (msg.type === "speak") {
          // HTTP fallback only when server could not stream WS audio
          const text = String(msg.text || "");
          if (text && !ttsReceivingRef.current && !usedSpeakEventsRef.current) {
            enqueueSpeak(text);
          }
        }
        if (msg.type === "transcript") {
          const role = msg.role as "user" | "assistant";
          const text = String(msg.text || "");
          if (role === "user") {
            usedSpeakEventsRef.current = false;
            speakQueueRef.current = [];
          }
          setTranscript((prev) => {
            if (role === "user") {
              const last = prev[prev.length - 1];
              if (last?.role === "user" && last.text === text) return prev;
            }
            if (role === "assistant") {
              const last = prev[prev.length - 1];
              if (last?.role === "assistant") {
                return [...prev.slice(0, -1), { ...last, text }];
              }
            }
            return [...prev, { id: `${Date.now()}-${prev.length}`, role, text }];
          });
          // TTS only via "speak" events — never also on transcript (avoids double reading)
        }
        if (msg.type === "sources") {
          setSources((msg.items as Source[]) || []);
        }
        if (msg.type === "error") {
          ttsReceivingRef.current = false;
          ttsChunksRef.current = [];
          setError(String(msg.message || "Erreur vocale"));
          setState("listening");
        }
      };

      ws.onopen = () => {
        if (!isCurrentSocket()) return;
        opened = true;
        startedAtRef.current = Date.now();
        if (timerRef.current) window.clearInterval(timerRef.current);
        timerRef.current = window.setInterval(() => {
          if (startedAtRef.current) {
            setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
          }
        }, 500);
        window.setTimeout(() => {
          if (!isCurrentSocket() || ws.readyState !== WebSocket.OPEN) return;
          setState((prev) => (prev === "connecting" ? "listening" : prev));
        }, 2000);
      };

      ws.onerror = () => {
        if (!isCurrentSocket()) return;
        if (!opened) {
          setError(
            "Connexion vocale impossible. Vérifiez que l'API tourne (port 8001).",
          );
          cleanupSession("error");
        }
      };

      ws.onclose = (ev) => {
        if (!isCurrentSocket()) return;
        const reason = ev.reason || `code ${ev.code}`;
        setError((prev) => prev || `Session vocale fermée (${reason}).`);
        cleanupSession("error");
      };
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Microphone inaccessible. Autorisez le micro dans le navigateur.";
      setError(message);
      cleanupSession("error");
    }
  }, [cleanupSession, ensureAudioCtx, enqueueSpeak, flushTtsChunks, pauseRecognition, tenantId, token]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      try {
        recognitionRef.current?.abort();
      } catch {
        /* ignore */
      }
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!activeRef.current) return;
    mediaStreamRef.current?.getAudioTracks().forEach((t) => {
      t.enabled = !muted;
    });
    if (muted) {
      holdModeRef.current = null;
      setHolding(false);
      try {
        recognitionRef.current?.abort();
      } catch {
        /* ignore */
      }
      recognitionRef.current = null;
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state === "recording") {
        try {
          recorder.stop();
        } catch {
          /* ignore */
        }
      }
      setInterim("Micro coupé");
    } else if (state === "listening") {
      window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
    }
  }, [muted, state]);

  return {
    state,
    muted,
    setMuted,
    transcript,
    sources,
    error,
    elapsed,
    interim,
    holding,
    serverStt,
    startCall,
    endCall,
    sendUserText,
    speakNeural,
    speakBrowser,
    setTranscript,
    setSources,
  };
}
