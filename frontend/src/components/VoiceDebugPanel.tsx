"use client";

import { useState } from "react";
import { DEBUG_STAGES, STT_COMPARE_PHRASES, llmLabel, sttRouteLabel, transportRemainder, type VoiceDebugTurn } from "@/lib/voiceDebug";

const duration = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;

export function VoiceDebugPanel({ turns, llmProvider, llmModel }: { turns: VoiceDebugTurn[]; llmProvider?: string; llmModel?: string }) {
  const [selected, setSelected] = useState("");
  const turn = turns.find(item => item.id === selected) || turns[0];
  const model = llmLabel(turn?.llmProvider || llmProvider, turn?.llmModel || llmModel);
  const remainder = turn ? transportRemainder(turn) : undefined;
  const longest = turn ? Math.max(1, ...Object.values(turn.stages).map(stage => stage.duration)) : 1;
  const exportTimings = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(turns, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "voice-debug.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return <details open className="mt-4 rounded-xl border border-primary/15 bg-white/60 p-4 text-ink">
    <summary className="cursor-pointer text-sm font-semibold">Debug vocal · temps par étape</summary>
    {model && <p className="mt-2 text-xs text-soft">Modèle IA : <span className="font-semibold text-ink">{model}</span></p>}
    {turn && <p className="mt-1 text-xs text-soft">STT : <span className="font-semibold text-ink">{sttRouteLabel(turn)}</span></p>}
    {!turn ? <p className="mt-3 text-xs text-soft">Les mesures apparaîtront après votre première intervention.</p> : <>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <label className="text-xs">Intervention <select aria-label="Intervention à inspecter" value={turn.id} onChange={event => setSelected(event.target.value)} className="ml-1 rounded border bg-white p-1">
          {turns.map((item, index) => <option key={item.id} value={item.id}>{index === 0 ? "Dernière · " : ""}{new Date(item.createdAt).toLocaleTimeString("fr-FR")}</option>)}
        </select></label>
        <button type="button" onClick={exportTimings} className="text-xs font-semibold underline">Exporter JSON</button>
      </div>
      <p className="mt-3 text-lg font-semibold tabular-nums">{turn.speechEndToAudio === undefined && turn.total === undefined ? (turn.status === "pending" ? "Mesure en cours…" : "Indisponible") : duration(turn.speechEndToAudio ?? turn.total ?? 0)}
        <span className="block text-xs font-normal text-soft">Fin de parole (micro) → première lecture de la réponse</span>
      </p>
      <p className="mt-1 text-xs text-soft">Fin de parole (micro) → transcript envoyé : <span className="font-semibold text-ink tabular-nums">{turn.speechEndToSent === undefined ? "—" : duration(turn.speechEndToSent)}</span>
        <span className="block">Origine fin de parole : {turn.speechEndSource === "microphone" ? "activité micro (RMS)" : "indisponible"}</span>
      </p>
      {turn.heardText && <p className="mt-1 text-xs text-soft">Texte entendu : <span className="text-ink">{turn.heardText}</span></p>}
      {turn.sttAudioMs !== undefined && <p className="mt-1 text-xs text-soft">Audio transcrit : <span className="font-semibold text-ink tabular-nums">{duration(turn.sttAudioMs)}</span>
        {turn.sttSpeechMs !== undefined ? ` · parole ${duration(turn.sttSpeechMs)}` : ""}
        {turn.sttLeadSilenceMs !== undefined ? ` · silence début ${duration(turn.sttLeadSilenceMs)}` : ""}
        {turn.sttTrailSilenceMs !== undefined ? ` · silence fin ${duration(turn.sttTrailSilenceMs)}` : ""}
        {turn.sttStartupWarmed === false ? " · démarrage Whisper non terminé" : ""}
        {turn.sttIdleMs !== undefined ? ` · idle GPU ${duration(turn.sttIdleMs)}` : ""}
        {turn.sttGpuPstate ? ` · ${turn.sttGpuPstate}${turn.sttGpuClockMhz ? ` ${Math.round(turn.sttGpuClockMhz)} MHz` : ""}` : ""}
        {turn.sttKeepAwake ? " · GPU maintenu" : ""}
      </p>}
      <table className="mt-3 w-full text-left text-xs">
        <thead><tr className="text-soft"><th scope="col" className="pb-2 font-normal">Étape</th><th scope="col" className="pb-2 text-right font-normal">Durée</th></tr></thead>
        <tbody>{DEBUG_STAGES.filter(([key]) => (!key.startsWith("stt_") && !key.startsWith("rag_")) || turn.stages[key]).map(([key, label]) => {
          const stage = turn.stages[key];
          return <tr key={key}><th scope="row" className="py-1.5 pr-3 font-normal">
            {label}
            {stage && <div className="mt-1 h-1 rounded bg-primary/5"><div className={`h-1 rounded ${stage.failed ? "bg-red-400" : "bg-accent"}`} style={{ width: `${Math.max(1, stage.duration / longest * 100)}%` }} /></div>}
          </th><td className="whitespace-nowrap text-right tabular-nums">{stage ? `${duration(stage.duration)}${stage.failed ? " · échec" : ""}` : key === "stt" && turn.input === "browser" ? "Pendant la parole" : "—"}</td></tr>;
        })}
          <tr><th scope="row" className="py-2 pr-3 font-normal">Transport + temps non instrumenté²</th><td className="text-right tabular-nums">{remainder === undefined ? "—" : duration(remainder)}</td></tr>
        </tbody>
      </table>
      {turn.error && <p role="status" className="mt-2 text-xs text-red-700">{turn.error}</p>}
      <p className="mt-3 text-[11px] leading-relaxed text-soft">Les étapes peuvent se chevaucher : ne pas additionner les barres. Les accusés d’attente sont exclus. Le total s’arrête au début de lecture, pas à la fin de la réponse. « — » : mesure indisponible (pas d’activité micro mesurée, ou horloge incompatible).</p>
      <p className="mt-1 text-[11px] text-soft">Fin de parole = dernière trame micro au-dessus du seuil RMS, pas le dernier événement de transcription.</p>
      <p className="mt-1 text-[11px] text-soft">Résidu estimé, comprenant le réseau et l’ordonnancement ; ce n’est pas une mesure réseau isolée.</p>
      <details className="mt-3 text-[11px] text-soft">
        <summary className="cursor-pointer font-semibold text-ink">Protocole de comparaison (5 phrases)</summary>
        <ol className="mt-1 list-decimal pl-4">
          {STT_COMPARE_PHRASES.map(phrase => <li key={phrase} className="text-ink">{phrase}</li>)}
        </ol>
        <p className="mt-1">A : case cochée. B : case décochée avant de relancer, STT = faster-whisper · cuda. Exporter JSON après chaque série.</p>
      </details>
      <p className="mt-2 break-all font-mono text-[10px] text-soft">Turn {turn.id}{turn.traceId ? ` · Trace ${turn.traceId}` : ""}</p>
    </>}
  </details>;
}
