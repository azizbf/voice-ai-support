export type DebugStage = { duration: number; offset?: number; failed?: boolean };
export type VoiceDebugTurn = {
  id: string;
  createdAt: number;
  input: "audio" | "browser";
  status: "pending" | "playing" | "error" | "cancelled";
  stages: Record<string, DebugStage>;
  total?: number;
  serverFirstAudio?: number;
  callId?: string;
  traceId?: string;
  error?: string;
};

export function updateDebugTurn(turns: VoiceDebugTurn[], id: string, patch: Partial<VoiceDebugTurn>): VoiceDebugTurn[] {
  return turns.map(turn => turn.id === id ? { ...turn, ...patch, stages: { ...turn.stages, ...patch.stages } } : turn);
}

export function transportRemainder(turn: VoiceDebugTurn): number | undefined {
  const { endpointing, preparation, buffering } = turn.stages;
  if (turn.total === undefined || turn.serverFirstAudio === undefined || !endpointing || !preparation || !buffering) return undefined;
  const remainder = turn.total - turn.serverFirstAudio - endpointing.duration - preparation.duration - buffering.duration;
  // Small rounding differences are harmless; large negatives indicate incompatible clocks/data.
  return remainder >= -2 ? Math.max(0, remainder) : undefined;
}

export const DEBUG_STAGES = [
  ["endpointing", "Détection de fin de parole"],
  ["preparation", "Préparation et envoi de l’audio"],
  ["stt", "Transcription (STT)"],
  ["retrieval", "Recherche documentaire (RAG)"],
  ["llm_wait", "IA : attente du premier token"],
  ["llm_phrase", "IA : génération de la phrase"],
  ["tts", "Synthèse : premier audio (TTS)"],
  ["buffering", "Navigateur : début de lecture"],
] as const;
