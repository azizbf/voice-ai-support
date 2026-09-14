export type DebugStage = { duration: number; offset?: number; failed?: boolean };
export type SpeechEndSource = "microphone" | "unavailable";
export type VoiceDebugTurn = {
  id: string;
  createdAt: number;
  input: "audio" | "browser";
  sttProvider?: string;
  sttDevice?: string;
  speechEndSource?: SpeechEndSource;
  speechEndToSent?: number;
  speechEndToAudio?: number;
  heardText?: string;
  status: "pending" | "playing" | "error" | "cancelled";
  stages: Record<string, DebugStage>;
  total?: number;
  serverFirstAudio?: number;
  callId?: string;
  traceId?: string;
  llmProvider?: string;
  llmModel?: string;
  sttAudioMs?: number;
  sttLeadSilenceMs?: number;
  sttTrailSilenceMs?: number;
  sttSpeechMs?: number;
  sttStartupWarmed?: boolean;
  sttIdleMs?: number;
  sttGpuPstate?: string;
  sttGpuClockMhz?: number;
  sttKeepAwake?: boolean;
  error?: string;
};

export const STT_COMPARE_PHRASES = [
  "Je voudrais consulter ma facture.",
  "Combien coûte un abonnement à 30 dinars par mois ?",
  "Je souhaite changer mon abonnement et conserver mon numéro de téléphone.",
  "Mon internet ne fonctionne plus depuis ce matin.",
  "Pouvez-vous me confirmer le solde de mon compte ?",
];

export function llmLabel(provider?: string, model?: string): string | undefined {
  if (!provider && !model) return undefined;
  const brands: Record<string, string> = { gemini: "Gemini", anthropic: "Claude" };
  const brand = (provider && brands[provider]) || provider || "IA";
  return model ? `${brand} · ${model}` : brand;
}

export function sttRouteLabel(turn?: Pick<VoiceDebugTurn, "input" | "sttProvider" | "sttDevice">): string | undefined {
  if (!turn) return undefined;
  if (turn.input === "browser" || turn.sttProvider === "browser-speech" || turn.sttProvider === "client-speech") {
    return "Navigateur · Web Speech";
  }
  const provider = turn.sttProvider || "serveur";
  if (provider === "faster-whisper" && turn.sttDevice) {
    return `faster-whisper · ${turn.sttDevice}`;
  }
  return provider;
}

export function elapsedFromSpeechEnd(speechEnd: number | undefined, at: number): number | undefined {
  if (speechEnd === undefined || !Number.isFinite(speechEnd) || !Number.isFinite(at) || at < speechEnd) return undefined;
  return at - speechEnd;
}

export function availableTimings(turns: VoiceDebugTurn[], key: "speechEndToSent" | "speechEndToAudio" | "total"): number[] {
  return turns
    .map(turn => turn[key])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0);
}

export function medianMs(values: number[]): number | undefined {
  if (!values.length) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

export function updateDebugTurn(turns: VoiceDebugTurn[], id: string, patch: Partial<VoiceDebugTurn>): VoiceDebugTurn[] {
  return turns.map(turn => {
    if (turn.id !== id) return turn;
    const next = { ...turn, ...patch, stages: { ...turn.stages, ...patch.stages } };
    if (next.input === "browser") {
      next.sttProvider = "browser-speech";
      delete next.sttDevice;
    }
    return next;
  });
}

export function transportRemainder(turn: VoiceDebugTurn): number | undefined {
  const { endpointing, preparation, buffering } = turn.stages;
  if (turn.total === undefined || turn.serverFirstAudio === undefined || !endpointing || !preparation || !buffering) return undefined;
  const remainder = turn.total - turn.serverFirstAudio - endpointing.duration - preparation.duration - buffering.duration;
  // Small rounding differences are harmless; large negatives indicate incompatible clocks/data.
  return remainder >= -2 ? Math.max(0, remainder) : undefined;
}

function finiteMs(value: unknown): number | undefined {
  const ms = Number(value);
  return Number.isFinite(ms) && ms >= 0 ? ms : undefined;
}

/** Queue, decode, then inference are sequential; nested decode parts share the decode window. */
export function applySttSubstageTimings(sttOffset: number, timings: Record<string, unknown>): Record<string, DebugStage> {
  const stages: Record<string, DebugStage> = {};
  let cursor = sttOffset;
  const queue = finiteMs(timings.stt_queue_ms);
  if (queue !== undefined) {
    stages.stt_queue = { duration: queue, offset: cursor };
    cursor += queue;
  }
  const decode = finiteMs(timings.stt_decode_ms);
  const decodeOffset = cursor;
  if (decode !== undefined) {
    stages.stt_decode = { duration: decode, offset: decodeOffset };
    cursor += decode;
  }
  let insideDecode = decodeOffset;
  const codec = finiteMs(timings.stt_decode_codec_ms);
  if (codec !== undefined) {
    stages.stt_decode_codec = { duration: codec, offset: insideDecode };
    insideDecode += codec;
  }
  const alloc = finiteMs(timings.stt_decode_alloc_ms);
  if (alloc !== undefined) {
    stages.stt_decode_alloc = { duration: alloc, offset: insideDecode };
  }
  const infer = finiteMs(timings.stt_infer_ms);
  const inferOffset = cursor;
  if (infer !== undefined) {
    stages.stt_infer = { duration: infer, offset: inferOffset };
  }
  let insideInfer = inferOffset;
  const feat = finiteMs(timings.stt_feat_ms);
  if (feat !== undefined) {
    stages.stt_feat = { duration: feat, offset: insideInfer };
    insideInfer += feat;
  }
  const encode = finiteMs(timings.stt_encode_ms);
  if (encode !== undefined) {
    stages.stt_encode = { duration: encode, offset: insideInfer };
    insideInfer += encode;
  }
  const tokens = finiteMs(timings.stt_tokens_ms);
  if (tokens !== undefined) {
    stages.stt_tokens = { duration: tokens, offset: insideInfer };
  }
  return stages;
}

/** 300 ms only after enough speech and true floor silence; otherwise 450 ms. */
export function silenceShouldEnd(input: {
  heardMs: number;
  silenceMs: number;
  hardSilenceMs: number;
  maxSilenceMs?: number;
  fastSilenceMs?: number;
}): boolean {
  const cap = input.maxSilenceMs ?? 450;
  const fast = input.fastSilenceMs ?? 300;
  if (input.heardMs < 200) return false;
  if (input.silenceMs >= cap) return true;
  return input.heardMs >= 1200 && input.hardSilenceMs >= fast;
}

export function applyRetrievalSubstageTimings(retrievalOffset: number, timings: Record<string, unknown>): Record<string, DebugStage> {
  const stages: Record<string, DebugStage> = {};
  let cursor = retrievalOffset;
  for (const [stage, key] of [
    ["rag_embed_load", "rag_embed_load_ms"],
    ["rag_embed", "rag_embed_ms"],
    ["rag_search", "rag_search_ms"],
    ["rag_backfill", "rag_backfill_ms"],
  ] as const) {
    const value = finiteMs(timings[key]);
    if (value === undefined) continue;
    stages[stage] = { duration: value, offset: cursor };
    cursor += value;
  }
  return stages;
}

export const DEBUG_STAGES = [
  ["endpointing", "Détection de fin de parole"],
  ["preparation", "Préparation et envoi de l’audio"],
  ["stt", "Transcription (STT)"],
  ["stt_queue", "STT : file d’attente GPU"],
  ["stt_decode", "STT : décodage audio"],
  ["stt_decode_codec", "STT : codec / rééchantillonnage"],
  ["stt_decode_alloc", "STT : conversion PCM"],
  ["stt_infer", "STT : inférence Whisper"],
  ["stt_feat", "STT : spectrogramme"],
  ["stt_encode", "STT : encodeur GPU"],
  ["stt_tokens", "STT : décodage de tokens"],
  ["retrieval", "Recherche documentaire (RAG)"],
  ["rag_embed_load", "RAG : chargement du modèle"],
  ["rag_embed", "RAG : embedding requête"],
  ["rag_search", "RAG : recherche vectorielle"],
  ["rag_backfill", "RAG : rattrapage d’index"],
    ["llm_wait", "IA : attente du premier token"],
    ["llm_phrase", "IA : génération de la clause parlable"],
    ["llm_stream", "IA : fin de génération"],
    ["tts", "Synthèse : premier audio (TTS)"],
  ["buffering", "Navigateur : début de lecture"],
] as const;
