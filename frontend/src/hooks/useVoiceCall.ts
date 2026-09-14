"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { synthesizeSpeech, voiceWsUrl, type Source } from "@/lib/api";
import { StreamingAudio } from "@/lib/streamingAudio";
import { applyRetrievalSubstageTimings, applySttSubstageTimings, elapsedFromSpeechEnd, silenceShouldEnd, updateDebugTurn, type SpeechEndSource, type VoiceDebugTurn } from "@/lib/voiceDebug";
import { startMicMeter, rmsFromTimeDomain, MIC_HARD_SILENCE_RMS, MIC_SPEECH_RMS, type MicMeter } from "@/lib/micMeter";
import { downsampleTo16k, floatToPcm16 } from "@/lib/pcmWav";
import { PCM_CAPTURE_WORKLET } from "@/lib/pcmCaptureWorklet";

type VoiceState = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error";

type TranscriptItem = { id: string; role: "user" | "assistant"; text: string };

const BARGE_IN_GRACE_MS = 450;

function normalizeHeard(text: string): string {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function isLikelyEcho(heard: string, spoken: string): boolean {
  const a = normalizeHeard(heard);
  const b = normalizeHeard(`${spoken} un instant sil vous plait`);
  if (!a || a.split(" ").filter(Boolean).length < 2) return false;
  return b.includes(a) || (a.length >= 8 && b.includes(a.slice(0, Math.min(24, a.length))));
}

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
  const [preferBrowserStt, setPreferBrowserStt] = useState(true);
  const [browserSpeechAvailable, setBrowserSpeechAvailable] = useState(false);
  useEffect(() => { setBrowserSpeechAvailable(Boolean(getSpeechRecognition())); }, []);
  const [debugTurns, setDebugTurns] = useState<VoiceDebugTurn[]>([]);
  const [llmProvider, setLlmProvider] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const llmProviderRef = useRef("");
  const llmModelRef = useRef("");
  const updateDebug = useCallback((id: string, patch: Partial<VoiceDebugTurn>) => {
    setDebugTurns(turns => updateDebugTurn(turns, id, patch));
  }, []);

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
  const captureStartingRef = useRef(false);
  const mutedRef = useRef(false);
  const mimeRef = useRef("audio/webm");
  const holdModeRef = useRef<"speech" | "media" | "pcm" | null>(null);
  const speechFinalRef = useRef("");
  const serverSttRef = useRef(false);
  const smartTurnRef = useRef(false);
  const pcmSendingRef = useRef(false);
  const pcmWorkletReadyRef = useRef(false);
  const pcmSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pcmNodeRef = useRef<AudioWorkletNode | ScriptProcessorNode | null>(null);
  const pcmAccRef = useRef<Int16Array>(new Int16Array(0));
  const pcmBargeTimerRef = useRef<number | null>(null);
  const serverSttNameRef = useRef("client-speech");
  const serverSttDeviceRef = useRef<string | undefined>(undefined);
  const ttsChunksRef = useRef<Uint8Array[]>([]);
  const ttsReceivingRef = useRef(false);
  const browserSpeakRef = useRef<SpeechSynthesisUtterance | null>(null);
  const speakQueueRef = useRef<string[]>([]);
  const speakBusyRef = useRef(false);
  const usedSpeakEventsRef = useRef(false);
  const streamPlayerRef = useRef<StreamingAudio | null>(null);
  const responsePendingRef = useRef(false);
  const turnTimingRef = useRef<{ id: string; speechEnd?: number; endpointing?: number; begunAt: number; firstAudioAt?: number; input?: "audio" | "browser" } | null>(null);
  const micMeterRef = useRef<MicMeter | null>(null);
  const thinkingTimerRef = useRef<number | null>(null);
  const thinkingUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const lastAssistantRef = useRef("");
  const ttsStartedAtRef = useRef(0);

  const stopThinkingPhrase = useCallback(() => {
    if (thinkingTimerRef.current !== null) window.clearTimeout(thinkingTimerRef.current);
    thinkingTimerRef.current = null;
    if (thinkingUtteranceRef.current) window.speechSynthesis?.cancel();
    thinkingUtteranceRef.current = null;
  }, []);

  const beginTurn = useCallback((speechEnd: number | undefined, input: "audio" | "browser", turnId?: string) => {
    stopThinkingPhrase();
    const id = turnId || crypto.randomUUID();
    const begunAt = performance.now();
    const speechEndSource: SpeechEndSource = speechEnd === undefined ? "unavailable" : "microphone";
    const endpointing = elapsedFromSpeechEnd(speechEnd, begunAt);
    turnTimingRef.current = { id, speechEnd, endpointing, begunAt, input };
    const sttMeta = input === "browser"
      ? { sttProvider: "browser-speech" }
      : { sttProvider: serverSttNameRef.current, sttDevice: serverSttDeviceRef.current };
    const stages = endpointing === undefined ? {} : { endpointing: { duration: endpointing } };
    setDebugTurns(turns => [{ id, createdAt: Date.now(), input, ...sttMeta, speechEndSource, status: "pending", llmProvider: llmProviderRef.current, llmModel: llmModelRef.current, stages }, ...turns].slice(0, 10) as VoiceDebugTurn[]);
    responsePendingRef.current = true;
    setState("thinking");
    thinkingTimerRef.current = window.setTimeout(() => {
      thinkingTimerRef.current = null;
      if (!activeRef.current || !responsePendingRef.current || playingRef.current || !window.speechSynthesis) return;
      const utterance = new SpeechSynthesisUtterance("Un instant, s’il vous plaît.");
      utterance.lang = "fr-FR";
      thinkingUtteranceRef.current = utterance;
      const clear = () => {
        if (thinkingUtteranceRef.current === utterance) thinkingUtteranceRef.current = null;
      };
      utterance.onend = clear;
      utterance.onerror = clear;
      // An acknowledgement is not counted as answer audio in TTFA metrics.
      window.speechSynthesis.speak(utterance);
    }, 2200);
    return id;
  }, [stopThinkingPhrase]);

  const markSent = useCallback((id: string) => {
    const timing = turnTimingRef.current;
    if (timing?.id !== id) return;
    const sentAt = performance.now();
    updateDebug(id, {
      ...(timing.input === "browser" ? { speechEndToSent: elapsedFromSpeechEnd(timing.speechEnd, sentAt) } : {}),
      stages: { preparation: { duration: sentAt - timing.begunAt } },
    });
  }, [updateDebug]);

  const reportPlayback = useCallback((turnId?: string) => {
    const timing = turnTimingRef.current;
    const ws = wsRef.current;
    if (!timing || (turnId && timing.id !== turnId) || ws?.readyState !== WebSocket.OPEN) return;
    const now = performance.now();
    const speechEndToAudio = elapsedFromSpeechEnd(timing.speechEnd, now);
    updateDebug(timing.id, {
      status: "playing",
      speechEndToAudio,
      ...(speechEndToAudio === undefined ? {} : { total: speechEndToAudio }),
      ...(timing.firstAudioAt !== undefined ? { stages: { buffering: { duration: now - timing.firstAudioAt } } } : {}),
    });
    if (speechEndToAudio !== undefined && timing.endpointing !== undefined) {
      ws.send(JSON.stringify({
        type: "playback_started",
        turn_id: timing.id,
        ttfa_ms: speechEndToAudio,
        endpointing_ms: timing.endpointing,
      }));
    }
    turnTimingRef.current = null;
  }, [updateDebug]);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  const releaseMicStream = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
  }, []);

  const ensureMicStream = useCallback(async () => {
    const existing = mediaStreamRef.current;
    if (existing && existing.getAudioTracks().some((t) => t.readyState === "live")) {
      return existing;
    }
    releaseMicStream();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    mediaStreamRef.current = stream;
    return stream;
  }, [releaseMicStream]);
  const startContinuousListenRef = useRef<() => void>(() => undefined);
  const RESUME_MS = 60;
  // Shorter endpointing reduces the pause before STT; increase for hesitant speech.
  const configuredSilence = Number(process.env.NEXT_PUBLIC_VOICE_SILENCE_MS ?? 450);
  const SILENCE_MS = Number.isFinite(configuredSilence)
    ? Math.min(1200, Math.max(300, configuredSilence))
    : 450;

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
        utter.onstart = () => reportPlayback();
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
    [reportPlayback, stopBrowserSpeak],
  );

  const stopPlayback = useCallback(() => {
    stopThinkingPhrase();
    streamPlayerRef.current?.stop();
    streamPlayerRef.current = null;
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
  }, [stopBrowserSpeak, stopThinkingPhrase]);

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

  const sendUserText = useCallback((text: string, speechEnd?: number) => {
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
    const turnId = beginTurn(speechEnd, "browser");
    updateDebug(turnId, { heardText: cleaned });
    ws.send(JSON.stringify({ type: "user_transcript", text: cleaned, turn_id: turnId }));
    markSent(turnId);
  }, [beginTurn, markSent, stopPlayback, updateDebug]);

  const stopPcmCapture = useCallback(() => {
    pcmSendingRef.current = false;
    pcmAccRef.current = new Int16Array(0);
    if (pcmBargeTimerRef.current !== null) {
      window.clearTimeout(pcmBargeTimerRef.current);
      pcmBargeTimerRef.current = null;
    }
    const node = pcmNodeRef.current;
    pcmNodeRef.current = null;
    const source = pcmSourceRef.current;
    pcmSourceRef.current = null;
    try {
      source?.disconnect();
    } catch {
      /* ignore */
    }
    try {
      node?.disconnect();
    } catch {
      /* ignore */
    }
    if (node && "port" in node) {
      node.port.onmessage = null;
    }
    if (node && "onaudioprocess" in node) {
      node.onaudioprocess = null;
    }
  }, []);

  const stopCapture = useCallback(() => {
    stopPcmCapture();
    micMeterRef.current?.stop();
    micMeterRef.current = null;
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
    releaseMicStream();
  }, [releaseMicStream, stopPcmCapture]);

  const cleanupSession = useCallback(
    (nextState: VoiceState = "idle") => {
      activeRef.current = false;
      if (turnTimingRef.current) updateDebug(turnTimingRef.current.id, { status: "cancelled", error: "Appel terminé avant la lecture audio." });
      responsePendingRef.current = false;
      turnTimingRef.current = null;
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
    [stopCapture, stopPlayback, updateDebug],
  );

  const beginHoldSpeech = useCallback(() => {
    const Ctor = getSpeechRecognition();
    if (!Ctor || mutedRef.current || !activeRef.current) return false;

    try {
      recognitionRef.current?.abort();
    } catch {
      /* ignore */
    }

    speechFinalRef.current = "";
    let interimText = "";
    let flushing = false;
    let partialTimer: number | null = null;
    setInterim("● Micro actif — parlez maintenant");
    const recognition = new Ctor();
    recognition.lang = "fr-FR";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognitionRef.current = recognition;

    const heardText = () => `${speechFinalRef.current} ${interimText}`.trim();

    const flushSpeech = () => {
      if (flushing) return false;
      const text = heardText();
      if (!text) return false;
      flushing = true;
      if (partialTimer) window.clearTimeout(partialTimer);
      speechFinalRef.current = "";
      interimText = "";
      setInterim("");
      holdModeRef.current = null;
      setHolding(false);
      try {
        recognition.stop();
      } catch {
        /* ignore */
      }
      recognitionRef.current = null;
      const meter = micMeterRef.current;
      const speechEnd = meter?.heard() ? meter.lastSpeechAt() : undefined;
      meter?.stop();
      if (micMeterRef.current === meter) micMeterRef.current = null;
      sendUserText(text, speechEnd);
      window.setTimeout(() => {
        if (!activeRef.current || mutedRef.current) return;
        beginHoldSpeech();
      }, 80);
      return true;
    };

    let sendTimer: number | null = null;
    const armAutoSend = () => {
      if (sendTimer) window.clearTimeout(sendTimer);
      sendTimer = window.setTimeout(() => {
        if (!activeRef.current || mutedRef.current) return;
        flushSpeech();
      }, SILENCE_MS);
    };

    recognition.onresult = (ev: unknown) => {
      if (flushing) return;
      const event = ev as {
        resultIndex: number;
        results: Array<{ isFinal: boolean; 0: { transcript: string } }>;
      };
      interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const piece = result[0].transcript;
        if (result.isFinal) {
          speechFinalRef.current = `${speechFinalRef.current} ${piece}`.trim();
        } else {
          interimText += piece;
        }
      }
      const heard = heardText();
      const agentBusy = playingRef.current || ttsReceivingRef.current || responsePendingRef.current;
      if (agentBusy && heard) {
        const words = heard.split(/\s+/).filter(Boolean);
        const inGrace = playingRef.current && performance.now() - ttsStartedAtRef.current < BARGE_IN_GRACE_MS;
        if (inGrace || words.length < 2 || isLikelyEcho(heard, lastAssistantRef.current)) return;
        stopPlayback();
        ttsReceivingRef.current = false;
        playingRef.current = false;
        responsePendingRef.current = false;
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "interrupt" }));
        setState("listening");
      }
      setInterim(heard || "● Micro actif — parlez maintenant");
      if (heard) armAutoSend();
      if (partialTimer) window.clearTimeout(partialTimer);
      partialTimer = window.setTimeout(() => {
        const ws = wsRef.current;
        if (!flushing && activeRef.current && !mutedRef.current && !playingRef.current && ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "user_partial", text: heardText() }));
        }
      }, 80);
    };

    recognition.onerror = (ev: unknown) => {
      const err = ev as { error?: string };
      if (!err.error || ["no-speech", "aborted"].includes(err.error)) return;
      if (err.error === "not-allowed") {
        setError("Micro bloqué. Autorisez le microphone pour localhost:3000 dans Chrome.");
        return;
      }
      if (["network", "service-not-allowed"].includes(err.error)) {
        micMeterRef.current?.stop();
        micMeterRef.current = null;
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
      if (partialTimer) window.clearTimeout(partialTimer);
      if (sendTimer) window.clearTimeout(sendTimer);
      if (flushing) return;
      if (!activeRef.current || mutedRef.current) return;
      if (holdModeRef.current !== "speech") return;
      if (!playingRef.current && flushSpeech()) return;
      window.setTimeout(() => {
        if (!activeRef.current || mutedRef.current) return;
        if (holdModeRef.current !== "speech") return;
        try {
          recognition.start();
        } catch {
          beginHoldSpeech();
        }
      }, playingRef.current ? 80 : 250);
    };

    try {
      recognition.start();
      holdModeRef.current = "speech";
      setHolding(true);
      void (async () => {
        try {
          const stream = await ensureMicStream();
          const ctx = await ensureAudioCtx();
          if (recognitionRef.current !== recognition || !activeRef.current) return;
          micMeterRef.current?.stop();
          micMeterRef.current = startMicMeter(ctx, stream, {
            ignore: () => playingRef.current || ttsReceivingRef.current || mutedRef.current,
          });
        } catch {
          micMeterRef.current = null;
        }
      })();
      return true;
    } catch {
      return false;
    }
  }, [ensureAudioCtx, ensureMicStream, sendUserText, stopPlayback]);

  const beginHoldMedia = useCallback(async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !activeRef.current) return false;
    if (mutedRef.current) return false;

    let stream: MediaStream;
    try {
      stream = await ensureMicStream();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Impossible d'accéder au micro.");
      return false;
    }

    if (!activeRef.current || mutedRef.current || wsRef.current !== ws) return false;
    micMeterRef.current?.stop();
    micMeterRef.current = null;

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

    // Reuse the context unlocked by the call button; a new context can remain
    // suspended and make the silence detector read zeros forever.
    const audioCtx = await ensureAudioCtx();
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    sourceNode.connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);

    let speaking = false;
    let silenceMs = 0;
    let hardSilenceMs = 0;
    let heardMs = 0;
    let lastSpeechAt = performance.now();
    let alive = true;
    const startedAt = performance.now();

    const teardownMeter = () => {
      alive = false;
      sourceNode.disconnect();
      analyser.disconnect();
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
      if (!alive || recorder.state === "inactive" || !activeRef.current || mutedRef.current) {
        stopAndSend();
        return;
      }
      analyser.getByteTimeDomainData(samples);
      const rms = rmsFromTimeDomain(samples);
      const elapsed = performance.now() - startedAt;

      if (playingRef.current || ttsReceivingRef.current) {
        const barge =
          performance.now() - ttsStartedAtRef.current >= BARGE_IN_GRACE_MS && rms > 0.035;
        if (barge) {
          stopPlayback();
          ttsReceivingRef.current = false;
          playingRef.current = false;
          responsePendingRef.current = false;
          const socket = wsRef.current;
          if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "interrupt" }));
          audioChunksRef.current = [];
          teardownMeter();
          try {
            recorder.onstop = null;
            if (recorder.state === "recording") recorder.stop();
          } catch {
            /* ignore */
          }
          mediaRecorderRef.current = null;
          holdModeRef.current = null;
          setHolding(false);
          setState("listening");
          window.setTimeout(() => startContinuousListenRef.current(), 60);
          return;
        }
        window.setTimeout(tick, 80);
        return;
      }

      if (responsePendingRef.current) {
        window.setTimeout(tick, 80);
        return;
      }

      if (rms > MIC_SPEECH_RMS) {
        lastSpeechAt = performance.now();
        speaking = true;
        heardMs += 80;
        silenceMs = 0;
        hardSilenceMs = 0;
        setInterim("● Je vous écoute…");
      } else if (speaking) {
        silenceMs += 80;
        if (rms < MIC_HARD_SILENCE_RMS) hardSilenceMs += 80;
        else hardSilenceMs = 0;
      }

      if ((speaking && silenceShouldEnd({ heardMs, silenceMs, hardSilenceMs, maxSilenceMs: SILENCE_MS })) || elapsed > 12000) {
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
      if (!chunks.length || !activeRef.current || mutedRef.current || wsRef.current !== ws || ws.readyState !== WebSocket.OPEN) return;
      if (!speaking) {
        setInterim("● Micro actif — parlez maintenant");
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        return;
      }
      const blob = new Blob(chunks, { type: mimeRef.current });
      if (blob.size < 800) {
        window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        return;
      }
      setState("thinking");
      setInterim("Transcription…");
      const turnId = beginTurn(lastSpeechAt, "audio");
      // Both server providers decode WebM/Opus directly. Avoid decoding,
      // resampling and uploading a larger WAV before transcription can start.
      void blob.arrayBuffer()
        .then((buf) => {
          if (!activeRef.current || ws.readyState !== WebSocket.OPEN) return;
          ws.send(buf);
          ws.send(JSON.stringify({ type: "audio_end", turn_id: turnId }));
          markSent(turnId);
          setInterim("");
          window.setTimeout(() => {
            if (!activeRef.current || mutedRef.current || holdModeRef.current) return;
            void beginHoldMedia();
          }, 80);
        })
        .catch(() => {
          setError("Impossible de lire l'enregistrement audio. Réessayez.");
          responsePendingRef.current = false;
          setInterim("");
          setState("listening");
          window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        });
    };

    try {
      recorder.start(250);
      holdModeRef.current = "media";
      setHolding(true);
      setInterim("● Micro actif — parlez maintenant");
      window.setTimeout(tick, 80);
      return true;
    } catch (err) {
      teardownMeter();
      setError(err instanceof Error ? err.message : "Impossible de démarrer le micro.");
      return false;
    }
  }, [beginTurn, ensureAudioCtx, ensureMicStream, markSent, stopPlayback]);

  const beginHoldPcm = useCallback(async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !activeRef.current) return false;
    if (mutedRef.current) return false;
    if (holdModeRef.current === "pcm" && pcmNodeRef.current) {
      pcmSendingRef.current = true;
      setHolding(true);
      return true;
    }

    let stream: MediaStream;
    try {
      stream = await ensureMicStream();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Impossible d'accéder au micro.");
      return false;
    }
    if (!activeRef.current || mutedRef.current || wsRef.current !== ws) return false;

    const audioCtx = await ensureAudioCtx();
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    sourceNode.connect(analyser);
    const meter = new Uint8Array(analyser.fftSize);

    const sendPcm = (samples: Float32Array) => {
      if (!pcmSendingRef.current || !activeRef.current || mutedRef.current) return;
      if (playingRef.current || ttsReceivingRef.current || responsePendingRef.current) return;
      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      const i16 = floatToPcm16(samples);
      const prev = pcmAccRef.current;
      const acc = new Int16Array(prev.length + i16.length);
      acc.set(prev);
      acc.set(i16, prev.length);
      let offset = 0;
      while (offset + 512 <= acc.length) {
        const frame = acc.subarray(offset, offset + 512);
        socket.send(frame.slice().buffer);
        offset += 512;
      }
      pcmAccRef.current = acc.subarray(offset).slice();
    };

    const tickBarge = () => {
      if (!activeRef.current || holdModeRef.current !== "pcm") return;
      analyser.getByteTimeDomainData(meter);
      const rms = rmsFromTimeDomain(meter);
      if (playingRef.current || ttsReceivingRef.current) {
        if (performance.now() - ttsStartedAtRef.current >= BARGE_IN_GRACE_MS && rms > 0.035) {
          stopPlayback();
          ttsReceivingRef.current = false;
          playingRef.current = false;
          responsePendingRef.current = false;
          const socket = wsRef.current;
          if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "interrupt" }));
          pcmSendingRef.current = true;
          setState("listening");
        }
      } else if (rms > MIC_SPEECH_RMS && !responsePendingRef.current) {
        setInterim("● Je vous écoute…");
      }
      pcmBargeTimerRef.current = window.setTimeout(tickBarge, 80);
    };

    const mute = audioCtx.createGain();
    mute.gain.value = 0;
    try {
      if (!pcmWorkletReadyRef.current) {
        const blob = new Blob([PCM_CAPTURE_WORKLET], { type: "application/javascript" });
        const url = URL.createObjectURL(blob);
        try {
          await audioCtx.audioWorklet.addModule(url);
          pcmWorkletReadyRef.current = true;
        } finally {
          URL.revokeObjectURL(url);
        }
      }
      const node = new AudioWorkletNode(audioCtx, "pcm16k-capture");
      node.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        if (ev.data instanceof Float32Array) sendPcm(ev.data);
      };
      sourceNode.connect(node);
      node.connect(mute);
      mute.connect(audioCtx.destination);
      pcmNodeRef.current = node;
    } catch {
      const proc = audioCtx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        sendPcm(downsampleTo16k(new Float32Array(input), audioCtx.sampleRate));
      };
      sourceNode.connect(proc);
      proc.connect(mute);
      mute.connect(audioCtx.destination);
      pcmNodeRef.current = proc;
    }
    pcmSourceRef.current = sourceNode;
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "pcm_start" }));
    pcmSendingRef.current = true;
    holdModeRef.current = "pcm";
    setHolding(true);
    setInterim("● Micro actif — parlez maintenant");
    pcmBargeTimerRef.current = window.setTimeout(tickBarge, 80);
    return true;
  }, [ensureAudioCtx, ensureMicStream, stopPlayback]);

  const startContinuousListen = useCallback(() => {
    if (!activeRef.current || mutedRef.current || playingRef.current || responsePendingRef.current) return;
    if (holdModeRef.current || captureStartingRef.current) return;
    captureStartingRef.current = true;
    void (async () => {
      try {
        if (smartTurnRef.current) {
          const pcmOk = await beginHoldPcm();
          if (!pcmOk && activeRef.current) setError("Micro indisponible pour l'écoute continue.");
          return;
        }
        if (serverSttRef.current) {
          const mediaOk = await beginHoldMedia();
          if (!mediaOk && activeRef.current) setError("Micro indisponible pour l'écoute continue.");
          return;
        }
        if (beginHoldSpeech()) return;
        const mediaOk = await beginHoldMedia();
        if (!mediaOk && activeRef.current) setError("Micro indisponible pour l'écoute continue.");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Impossible de démarrer le micro.");
      } finally {
        captureStartingRef.current = false;
      }
    })();
  }, [beginHoldMedia, beginHoldPcm, beginHoldSpeech]);

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
    setDebugTurns([]);
    llmProviderRef.current = "";
    llmModelRef.current = "";
    setLlmProvider("");
    setLlmModel("");
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
            const timing = turnTimingRef.current;
            if (timing && timing.firstAudioAt === undefined) timing.firstAudioAt = performance.now();
            streamPlayerRef.current?.push(ev.data as ArrayBuffer);
          }
          return;
        }
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        const debugId = String(msg.turn_id || "");
        if (debugId) updateDebug(debugId, { callId: String(msg.call_id || ""), traceId: String(msg.trace_id || "") });
        if (msg.type === "debug_timing") {
          const duration = Number(msg.duration_ms);
          const offset = Number(msg.offset_ms);
          if (Number.isFinite(duration) && duration >= 0 && Number.isFinite(offset) && offset >= 0) {
            const patch: Partial<VoiceDebugTurn> = { stages: { [String(msg.stage)]: { duration, offset, failed: Boolean(msg.failed) } } };
            if (String(msg.stage) === "stt") {
              if (msg.stt) patch.sttProvider = String(msg.stt);
              if (msg.stt_device) patch.sttDevice = String(msg.stt_device);
              patch.stages = { ...patch.stages, ...applySttSubstageTimings(offset, msg) };
              const audioMs = Number(msg.stt_audio_ms);
              if (Number.isFinite(audioMs) && audioMs >= 0) patch.sttAudioMs = audioMs;
              const leadMs = Number(msg.stt_lead_silence_ms);
              if (Number.isFinite(leadMs) && leadMs >= 0) patch.sttLeadSilenceMs = leadMs;
              const trailMs = Number(msg.stt_trail_silence_ms);
              if (Number.isFinite(trailMs) && trailMs >= 0) patch.sttTrailSilenceMs = trailMs;
              const speechMs = Number(msg.stt_speech_ms);
              if (Number.isFinite(speechMs) && speechMs >= 0) patch.sttSpeechMs = speechMs;
              if (msg.stt_startup_warmed !== undefined) patch.sttStartupWarmed = Number(msg.stt_startup_warmed) === 1;
              const idleMs = Number(msg.stt_idle_ms);
              if (Number.isFinite(idleMs) && idleMs >= 0) patch.sttIdleMs = idleMs;
              if (msg.stt_gpu_pstate) patch.sttGpuPstate = String(msg.stt_gpu_pstate);
              const clock = Number(msg.stt_gpu_clock_mhz);
              if (Number.isFinite(clock) && clock >= 0) patch.sttGpuClockMhz = clock;
              if (msg.stt_keep_awake !== undefined) patch.sttKeepAwake = Number(msg.stt_keep_awake) === 1;
              if (msg.stt_error) patch.error = String(msg.stt_error);
            }
            if (String(msg.stage) === "retrieval") {
              patch.stages = { ...patch.stages, ...applyRetrievalSubstageTimings(offset, msg) };
            }
            updateDebug(debugId, patch);
          }
          return;
        }
        if (msg.type === "ready") {
          const sttName = String(msg.stt || "client-speech");
          serverSttNameRef.current = sttName;
          serverSttDeviceRef.current = msg.stt_device ? String(msg.stt_device) : undefined;
          const smartTurn = Boolean(msg.smart_turn);
          smartTurnRef.current = smartTurn;
          const hasServer = smartTurn || (sttName !== "client-speech" && !(preferBrowserStt && getSpeechRecognition()));
          serverSttRef.current = hasServer;
          setServerStt(hasServer);
          const provider = String(msg.llm_provider || "");
          const model = String(msg.llm_model || "");
          llmProviderRef.current = provider;
          llmModelRef.current = model;
          setLlmProvider(provider);
          setLlmModel(model);
          setState("listening");
          setError(null);
          window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
        }
        if (msg.type === "turn_commit") {
          const endpointing = Number(msg.endpointing_ms);
          const speechEnd = Number.isFinite(endpointing) ? performance.now() - endpointing : undefined;
          const id = beginTurn(speechEnd, "audio", String(msg.turn_id || "") || undefined);
          markSent(id);
          setInterim("Transcription…");
          return;
        }
        if (msg.type === "tts_start") {
          if (typeof msg.server_first_audio_ms === "number") updateDebug(debugId, { serverFirstAudio: msg.server_first_audio_ms });
          stopThinkingPhrase();
          streamPlayerRef.current?.stop();
          const player = new StreamingAudio(() => reportPlayback(String(msg.turn_id || "")));
          streamPlayerRef.current = player;
          playingRef.current = true;
          ttsReceivingRef.current = true;
          ttsChunksRef.current = [];
          usedSpeakEventsRef.current = true;
          speakBusyRef.current = true;
          ttsStartedAtRef.current = performance.now();
          if (holdModeRef.current !== "speech" && holdModeRef.current !== "media" && holdModeRef.current !== "pcm") {
            holdModeRef.current = null;
            setHolding(false);
            pauseRecognition();
          }
          setState("speaking");
          return;
        }
        if (msg.type === "tts_end") {
          const player = streamPlayerRef.current;
          ttsReceivingRef.current = false;
          if (player) void player.end().catch((err: Error) => {
            updateDebug(debugId, { status: "error", error: err.message });
            if (streamPlayerRef.current === player) setError(err.message);
          }).finally(() => {
            if (streamPlayerRef.current !== player) return;
            streamPlayerRef.current = null;
            playingRef.current = false;
            speakBusyRef.current = false;
            if (!activeRef.current) return;
            setState(responsePendingRef.current ? "thinking" : "listening");
            window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
          });
          return;
        }
        if (msg.type === "tts_abort") {
          streamPlayerRef.current?.stop();
          streamPlayerRef.current = null;
          playingRef.current = false;
          speakBusyRef.current = false;
          usedSpeakEventsRef.current = false;
          ttsReceivingRef.current = false;
          return;
        }
        if (msg.type === "status") {
          const s = msg.state as VoiceState | undefined;
          if (s === "listening") responsePendingRef.current = false;
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
            if ((s === "speaking" || s === "thinking") && holdModeRef.current !== "speech" && holdModeRef.current !== "media" && holdModeRef.current !== "pcm") {
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
          stopThinkingPhrase();
          if (msg.engine === "browser") {
            const socket = ws;
            speakBusyRef.current = true;
            void speakBrowser(String(msg.text || "")).finally(() => {
              if (wsRef.current !== socket) return;
              speakBusyRef.current = false;
              window.setTimeout(() => startContinuousListenRef.current(), RESUME_MS);
            });
            return;
          }
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
            if (debugId) {
              const timing = turnTimingRef.current;
              const patch: Partial<VoiceDebugTurn> = { heardText: text };
              if (timing?.id === debugId && timing.input === "audio") {
                patch.speechEndToSent = elapsedFromSpeechEnd(timing.speechEnd, performance.now());
              }
              updateDebug(debugId, patch);
            }
          }
          setTranscript((prev) => {
            if (role === "user") {
              const last = prev[prev.length - 1];
              if (last?.role === "user" && last.text === text) return prev;
            }
            if (role === "assistant") {
              lastAssistantRef.current = text;
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
          updateDebug(debugId, { status: "error", error: String(msg.message || "Erreur vocale") });
          stopThinkingPhrase();
          responsePendingRef.current = false;
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
        // Capture starts only after "ready" selects the transcription provider.
      };

      ws.onerror = () => {
        if (!isCurrentSocket()) return;
        if (!opened) {
          setError(
            "Connexion vocale impossible. Vérifiez que l'API tourne (port 8000).",
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
  }, [cleanupSession, ensureAudioCtx, enqueueSpeak, pauseRecognition, preferBrowserStt, releaseMicStream, reportPlayback, speakBrowser, stopThinkingPhrase, tenantId, token, updateDebug]);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      streamPlayerRef.current?.stop();
      stopThinkingPhrase();
      try {
        recognitionRef.current?.abort();
      } catch {
        /* ignore */
      }
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      wsRef.current?.close();
    };
  }, [stopThinkingPhrase]);

  useEffect(() => {
    if (!activeRef.current) return;
    mediaStreamRef.current?.getAudioTracks().forEach((t) => {
      t.enabled = !muted;
    });
    if (muted) {
      holdModeRef.current = null;
      pcmSendingRef.current = false;
      stopPcmCapture();
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
  }, [muted, state, stopPcmCapture]);

  return {
    debugTurns,
    llmProvider,
    llmModel,
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
    preferBrowserStt,
    setPreferBrowserStt,
    browserSpeechAvailable,
    startCall,
    endCall,
    sendUserText,
    speakNeural,
    speakBrowser,
    setTranscript,
    setSources,
  };
}
