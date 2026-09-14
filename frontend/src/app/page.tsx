"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowRight, Mic, MicOff, PhoneOff, Play } from "lucide-react";
import {
  chatStream,
  createTenant,
  getStatus,
  loadDemoKnowledge,
  type Source,
  type TenantStatus,
} from "@/lib/api";
import { useVoiceCall } from "@/hooks/useVoiceCall";
import { VoiceDebugPanel } from "@/components/VoiceDebugPanel";

function formatTimer(seconds: number): string {
  const m = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const s = (seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export default function HomePage() {
  const [tenantId, setTenantId] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [status, setStatus] = useState<TenantStatus | null>(null);
  const [booting, setBooting] = useState(true);
  const [uiError, setUiError] = useState<string | null>(null);
  const [textInput, setTextInput] = useState("");
  const [textReply, setTextReply] = useState("");
  const [textSources, setTextSources] = useState<Source[]>([]);
  const [textBusy, setTextBusy] = useState(false);
  const [polling, setPolling] = useState(false);

  const voice = useVoiceCall(tenantId, token);

  const DEMO_DOC = "FAQ_Orange_TN_Demo_v2.pdf";
  const STORAGE_KEY = "auralis_demo_v3";

  const persistTenant = useCallback((tenant_id: string, tok: string) => {
    setTenantId(tenant_id);
    setToken(tok);
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ tenant_id, token: tok }),
    );
  }, []);

  const markProcessing = useCallback((tenant_id: string) => {
    setStatus({
      tenant_id,
      status: "processing",
      pipeline: { extracting: false, creating_knowledge_base: true, ready: false },
      document: null,
      error: null,
    });
    setPolling(true);
  }, []);

  const needsDemoKb = useCallback((s: TenantStatus) => {
    if (s.status !== "ready" || !s.document) return true;
    return s.document.original_name !== DEMO_DOC;
  }, []);

  // Auto-create tenant + internal FAQ (no PDF upload).
  useEffect(() => {
    let cancelled = false;
    localStorage.removeItem("auralis_demo");
    localStorage.removeItem("auralis_demo_v2"); // drop legacy sessions

    const boot = async () => {
      setBooting(true);
      setUiError(null);
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          try {
            const parsed = JSON.parse(raw) as { tenant_id: string; token: string };
            const s = await getStatus(parsed.tenant_id, parsed.token);
            if (cancelled) return;
            persistTenant(parsed.tenant_id, parsed.token);
            if (s.status === "processing") {
              try {
                await loadDemoKnowledge(parsed.tenant_id, parsed.token);
              } catch {
                // Status polling / backend resume will retry after a reload.
              }
              if (cancelled) return;
              markProcessing(parsed.tenant_id);
              return;
            }
            if (!needsDemoKb(s)) {
              setStatus(s);
              return;
            }
            await loadDemoKnowledge(parsed.tenant_id, parsed.token);
            if (cancelled) return;
            markProcessing(parsed.tenant_id);
            return;
          } catch {
            localStorage.removeItem(STORAGE_KEY);
          }
        }

        const created = await createTenant();
        if (cancelled) return;
        persistTenant(created.tenant_id, created.token);
        markProcessing(created.tenant_id);
      } catch (err) {
        if (!cancelled) {
          setUiError(err instanceof Error ? err.message : "Impossible de démarrer la démo");
        }
      } finally {
        if (!cancelled) setBooting(false);
      }
    };

    void boot();
    return () => {
      cancelled = true;
    };
  }, [markProcessing, needsDemoKb, persistTenant]);

  useEffect(() => {
    if (!tenantId || !token || !polling) return;
    const id = window.setInterval(async () => {
      try {
        const s = await getStatus(tenantId, token);
        setStatus(s);
        if (s.status === "ready") setPolling(false);
        if (s.status === "error") {
          setUiError(s.error || "Erreur de traitement");
          setPolling(false);
        }
      } catch (err) {
        setUiError(err instanceof Error ? err.message : "Erreur statut");
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [tenantId, token, polling]);

  const onRestart = async () => {
    voice.endCall();
    setTextReply("");
    setTextSources([]);
    voice.setTranscript([]);
    voice.setSources([]);
    localStorage.removeItem(STORAGE_KEY);
    setTenantId(null);
    setToken(null);
    setBooting(true);
    setUiError(null);
    try {
      const created = await createTenant();
      persistTenant(created.tenant_id, created.token);
      markProcessing(created.tenant_id);
    } catch (err) {
      setUiError(err instanceof Error ? err.message : "Redémarrage impossible");
    } finally {
      setBooting(false);
    }
  };

  const onTextChat = async () => {
    if (!tenantId || !token || !textInput.trim()) return;
    setTextBusy(true);
    setTextReply("");
    setTextSources([]);
    let full = "";
    try {
      await chatStream(
        tenantId,
        token,
        textInput.trim(),
        (t) => {
          full += t;
          setTextReply((prev) => prev + t);
        },
        (s) => setTextSources(s),
      );
      if (full.trim()) void voice.speakNeural(full);
    } catch (err) {
      setUiError(err instanceof Error ? err.message : "Erreur chat");
    } finally {
      setTextBusy(false);
    }
  };

  const inCall =
    voice.state === "connecting" ||
    voice.state === "listening" ||
    voice.state === "thinking" ||
    voice.state === "speaking";
  const canStartCall = status?.status === "ready" && !!tenantId && !!token;
  const displaySources = voice.sources.length ? voice.sources : textSources;
  const kbBusy = booting || polling || status?.status === "processing";

  const pipelineSteps = useMemo(
    () => [
      {
        label: "Chargement FAQ interne",
        done:
          !!status &&
          (status.pipeline.ready ||
            status.status === "ready" ||
            status.pipeline.creating_knowledge_base),
        active: kbBusy && status?.status !== "ready",
      },
      {
        label: "Index vectoriel (pgvector)",
        done: !!status && (status.pipeline.ready || status.status === "ready"),
        active: !!status?.pipeline.creating_knowledge_base && status.status !== "ready",
      },
      {
        label: "Agent prêt",
        done: status?.status === "ready",
        active: status?.status === "ready",
      },
    ],
    [kbBusy, status],
  );

  return (
    <div className="bg-scene relative min-h-screen w-full overflow-hidden font-body">
      <div className="pointer-events-none absolute -left-24 -top-24 size-[380px] rounded-full bg-accent/25 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 right-0 size-[420px] rounded-full bg-mint/25 blur-3xl" />

      <div className="relative mx-auto max-w-6xl px-6 py-6">
        <nav className="glass flex items-center justify-between rounded-2xl px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="grid size-9 place-items-center rounded-xl bg-primary font-display text-sm font-semibold text-primary-foreground">
              V
            </div>
            <span className="font-display text-lg font-semibold tracking-tight text-ink">
              Vantage AI
            </span>
            <span className="ml-1 hidden text-[10px] font-semibold uppercase tracking-[0.18em] text-soft sm:inline">
              Call Center
            </span>
          </div>
          <div className="hidden items-center gap-7 text-sm font-medium text-soft md:flex">
            <a href="#demo" className="text-ink">
              Live Demo
            </a>
            <a href="#features">Features</a>
            <a href="#console">Console</a>
            <Link href="/dashboard">Dashboard</Link>
          </div>
          <a
            href="#demo"
            className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90"
          >
            Open Console
          </a>        </nav>

        {(uiError || voice.error) && (
          <div className="glass mt-4 rounded-2xl border border-red-200/80 px-4 py-3 text-sm text-ink">
            {uiError || voice.error}
          </div>
        )}

        {/* Hero */}
        <div className="mt-10 grid items-center gap-8 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <div className="glass-soft inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium text-ink">
              <span className="size-2 animate-pulse rounded-full bg-mint" />
              Agent vocal RAG · centres d&apos;appels francophones
            </div>
            <h1 className="mt-5 font-display text-5xl font-semibold leading-[1.05] tracking-tight text-ink lg:text-6xl">
              Every call, <span className="text-accent">understood</span> in real time.
            </h1>
            <p className="mt-5 max-w-lg text-lg leading-relaxed text-soft">
              FAQ Fibre Tunisie chargée automatiquement. Parlez à un agent IA français ancré
              uniquement dans cette base de connaissances.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <a
                href="#demo"
                className="rounded-xl bg-accent px-6 py-3 font-semibold text-accent-foreground shadow-lg shadow-accent/30 transition hover:brightness-105"
              >
                Tester la démo
              </a>
              <a
                href="#console"
                className="glass-soft inline-flex items-center gap-2 rounded-xl px-6 py-3 font-semibold text-ink transition hover:bg-white/60"
              >
                <Play className="size-4" /> Voir la console
              </a>
            </div>
            <div className="mt-8 flex items-center gap-6 text-sm text-soft">
              <div>
                <span className="block font-display text-2xl font-semibold text-ink">FAQ</span>
                interne
              </div>
              <div className="h-8 w-px bg-primary/10" />
              <div>
                <span className="block font-display text-2xl font-semibold text-ink">FR</span>
                voice agent
              </div>
              <div className="h-8 w-px bg-primary/10" />
              <div>
                <span className="block font-display text-2xl font-semibold text-ink">KB</span>
                citations
              </div>
            </div>
          </div>

          {/* Live console preview / active call */}
          <div className="lg:col-span-5" id="console">
            <div className="glass rounded-3xl p-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className={`size-2.5 rounded-full ${
                      inCall ? "animate-pulse bg-mint" : "bg-soft/40"
                    }`}
                  />
                  <span className="font-display text-sm font-semibold text-ink">
                    {inCall ? "Live Call · AI Agent" : "Live Call · Preview"}
                  </span>
                </div>
                <span className="text-xs font-medium tabular-nums text-soft">
                  {inCall ? formatTimer(voice.elapsed) : "00:00"}
                </span>
              </div>

              <div className="mt-4 max-h-[280px] space-y-3 overflow-y-auto text-sm">
                {voice.transcript.length === 0 && !textReply ? (
                  <>
                    <div className="glass-soft rounded-2xl rounded-tl-sm px-3.5 py-2.5 leading-snug text-ink">
                      « Bonjour, ma connexion internet ne fonctionne plus. »
                    </div>
                    <div className="ml-auto max-w-[85%] rounded-2xl rounded-tr-sm bg-primary px-3.5 py-2.5 leading-snug text-primary-foreground">
                      « Je vais vous aider. D&apos;après la procédure, redémarrez le routeur
                      pendant 2 minutes… »
                    </div>
                  </>
                ) : (
                  <>
                    {voice.transcript.map((item) =>
                      item.role === "user" ? (
                        <div
                          key={item.id}
                          className="glass-soft rounded-2xl rounded-tl-sm px-3.5 py-2.5 leading-snug text-ink"
                        >
                          {item.text}
                        </div>
                      ) : (
                        <div
                          key={item.id}
                          className="ml-auto max-w-[90%] rounded-2xl rounded-tr-sm bg-primary px-3.5 py-2.5 leading-snug text-primary-foreground"
                        >
                          {item.text}
                        </div>
                      ),
                    )}
                    {textReply && (
                      <div className="ml-auto max-w-[90%] rounded-2xl rounded-tr-sm bg-primary px-3.5 py-2.5 leading-snug text-primary-foreground">
                        {textReply}
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-white/60 pt-4">
                <div className="flex flex-wrap items-center gap-2 text-xs font-medium text-soft">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-lg px-2 py-1 font-semibold ${
                      status?.status === "ready"
                        ? "bg-mint/15 text-mint"
                        : "bg-primary/5 text-soft"
                    }`}
                  >
                    {status?.status === "ready" ? "KB Ready" : "Préparation KB"}
                  </span>
                  <span className="capitalize">{voice.state === "idle" ? "idle" : voice.state}</span>
                </div>
                <span className="text-xs font-semibold text-accent">
                  {displaySources[0]
                    ? `${displaySources[0].document_name} · p.${displaySources[0].page}`
                    : "Auto-transcribed"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Interactive demo */}
        <div className="mt-14 grid gap-5 lg:grid-cols-12" id="demo">
          <div className="glass rounded-3xl p-6 lg:col-span-7">
            <div className="flex items-center gap-2">
              <div className="grid size-11 place-items-center rounded-2xl bg-accent/15 font-display text-lg font-semibold text-accent">
                1
              </div>
              <div>
                <h2 className="font-display text-lg font-semibold text-ink">
                  Base de connaissances
                </h2>
                <p className="text-sm text-soft">
                  FAQ Fibre Tunisie chargée automatiquement (sans PDF)
                </p>
              </div>
            </div>

            <div className="mt-5 rounded-2xl border border-primary/10 bg-white/35 px-4 py-4 text-sm text-ink">
              {kbBusy ? (
                <p>Indexation en cours… pannes, Wi-Fi, forfaits, facturation, résiliation.</p>
              ) : status?.status === "ready" ? (
                <p>
                  Prêt — essayez: « Combien coûte la Fibre 100 ? » ou « Ma connexion ne marche
                  plus ».
                </p>
              ) : (
                <p>Préparation de la session démo…</p>
              )}
            </div>

            <div className="mt-5 space-y-2.5">
              {pipelineSteps.map((item) => (
                <div key={item.label} className="flex items-center gap-3 text-sm">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${
                      item.done
                        ? "bg-mint"
                        : item.active
                          ? "pipeline-bar h-2.5 w-8 rounded-full"
                          : "bg-primary/15"
                    }`}
                  />
                  <span className={item.done || item.active ? "text-ink" : "text-soft"}>
                    {item.label}
                  </span>
                </div>
              ))}
              {status?.document && status.status === "ready" && (
                <p className="pt-1 text-xs text-soft">
                  {status.document.original_name} · {status.document.page_count} pages ·{" "}
                  {status.document.chunk_count} segments
                </p>
              )}
            </div>

            <div className="mt-6 border-t border-white/60 pt-5">
              <div className="mb-3 flex items-center gap-2">
                <div className="grid size-11 place-items-center rounded-2xl bg-mint/15 font-display text-lg font-semibold text-mint">
                  2
                </div>
                <div>
                  <h2 className="font-display text-lg font-semibold text-ink">
                    Tester l&apos;agent
                  </h2>
                  <p className="text-sm text-soft">Appel vocal ou question texte</p>
                </div>
              </div>

              <label className="mb-3 block text-xs text-soft">
                <span className="flex items-center gap-2 font-semibold text-ink">
                  <input type="checkbox" checked={voice.preferBrowserStt}
                    onChange={event => voice.setPreferBrowserStt(event.target.checked)}
                    disabled={inCall || !voice.browserSpeechAvailable} />
                  Reconnaissance en direct · Chrome / Edge
                </span>
                <span className="mt-1 block">Activé : Web Speech. Désactivé : audio enregistré → Whisper serveur. Décochez avant de relancer l’appel. Retour au moteur local si la connexion échoue.</span>
              </label>

              {!inCall ? (
                <button
                  type="button"
                  disabled={!canStartCall}
                  onClick={() => {
                    setUiError(null);
                    void voice.startCall();
                  }}
                  className="w-full rounded-xl bg-accent px-6 py-3.5 font-semibold text-accent-foreground shadow-lg shadow-accent/25 transition hover:brightness-105 disabled:cursor-not-allowed disabled:bg-primary/20 disabled:text-soft disabled:shadow-none"
                >
                  Démarrer l&apos;appel IA
                </button>
              ) : (
                <div className="space-y-3">
                  {voice.state === "connecting" && (
                    <p className="text-sm text-soft">
                      Connexion… autorisez le micro si demandé.
                    </p>
                  )}
                  {voice.state === "listening" && (
                    <p className="text-sm text-soft">
                      Micro actif — parlez, puis faites une courte pause. Utilisez Chrome ou Edge.
                    </p>
                  )}
                  {voice.state !== "connecting" && <p className="text-xs text-soft">Transcription : {voice.serverStt ? "serveur (audio enregistré)" : "navigateur en direct · Web Speech"}</p>}
                  {voice.state === "thinking" && (
                    <p className="text-sm text-soft">Réflexion en cours…</p>
                  )}
                  {voice.state === "speaking" && (
                    <p className="text-sm text-soft">L&apos;agent parle…</p>
                  )}
                  {voice.interim && (
                    <p className="rounded-xl bg-white/50 px-3 py-2 text-sm italic text-ink">
                      {voice.interim}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => voice.setMuted(!voice.muted)}
                      className="glass-soft inline-flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold text-ink"
                    >
                      {voice.muted ? <MicOff className="size-4" /> : <Mic className="size-4" />}
                      {voice.muted ? "Réactiver le micro" : "Muet"}
                    </button>
                    <button
                      type="button"
                      onClick={() => voice.endCall()}
                      className="inline-flex items-center gap-2 rounded-xl bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground"
                    >
                      <PhoneOff className="size-4" /> Terminer
                    </button>
                  </div>
                </div>
              )}

              <VoiceDebugPanel turns={voice.debugTurns} llmProvider={voice.llmProvider} llmModel={voice.llmModel} />

              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => void onRestart()}
                  className="rounded-lg px-3 py-2 text-xs font-semibold text-soft hover:bg-white/40"
                >
                  Nouvelle session
                </button>
              </div>

              <div className="mt-4">
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  rows={2}
                  placeholder="Ou posez une question en texte…"
                  className="w-full resize-none rounded-xl border border-white/70 bg-white/50 px-3 py-2 text-sm text-ink outline-none ring-accent placeholder:text-soft/70 focus:ring-2"
                  disabled={status?.status !== "ready"}
                />
                <button
                  type="button"
                  disabled={status?.status !== "ready" || textBusy || !textInput.trim()}
                  onClick={() => void onTextChat()}
                  className="mt-2 rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-40"
                >
                  {textBusy ? "Génération…" : "Envoyer"}
                </button>
              </div>
            </div>
          </div>

          <div className="space-y-5 lg:col-span-5">
            <div className="glass rounded-3xl p-6">
              <h3 className="font-display text-lg font-semibold text-ink">Sources utilisées</h3>
              <ul className="mt-4 space-y-3">
                {displaySources.length === 0 && (
                  <li className="text-sm text-soft">
                    Les citations document / page apparaîtront après une réponse.
                  </li>
                )}
                {displaySources.map((s, i) => (
                  <li
                    key={`${s.document_name}-${s.page}-${i}`}
                    className="glass-soft rounded-2xl px-3.5 py-2.5"
                  >
                    <p className="text-sm font-semibold text-ink">{s.document_name}</p>
                    <p className="text-xs text-soft">Page {s.page}</p>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass rounded-3xl p-6">
              <h3 className="font-display text-lg font-semibold text-ink">Isolation multi-tenant</h3>
              <p className="mt-2 text-sm leading-relaxed text-soft">
                Chaque session démo a ses propres embeddings, filtrés par tenant_id. Aucune
                retrieval croisée entre entreprises.
              </p>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="mt-14 grid gap-5 md:grid-cols-3" id="features">
          {[
            {
              letter: "A",
              iconClass: "bg-accent/15 text-accent",
              title: "AI Agent Handoff",
              body: "Si l'information n'est pas dans la base, l'agent recommande une escalation vers un humain — sans inventer de politiques.",
            },
            {
              letter: "T",
              iconClass: "bg-mint/15 text-mint",
              title: "Real-time Transcripts",
              body: "Transcript live client / IA, sources citées (document + page), latence mesurée de bout en bout.",
            },
            {
              letter: "R",
              iconClass: "bg-primary/10 text-primary",
              title: "Grounded RAG",
              body: "La FAQ interne est indexée au démarrage. Chaque question recherche uniquement la knowledge base du tenant.",
            },
          ].map((f) => (
            <div key={f.title} className="glass rounded-3xl p-6">
              <div
                className={`grid size-11 place-items-center rounded-2xl font-display text-lg font-semibold ${f.iconClass}`}
              >
                {f.letter}
              </div>
              <h3 className="mt-4 font-display text-lg font-semibold text-ink">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-soft">{f.body}</p>
            </div>
          ))}
        </div>

        <div
          className="glass mt-14 flex flex-col justify-between gap-5 rounded-3xl px-8 py-8 md:flex-row md:items-center"
          id="cta"
        >
          <div>
            <h2 className="font-display text-2xl font-semibold tracking-tight text-ink">
              Put your call center on autopilot.
            </h2>
            <p className="mt-1.5 text-sm text-soft">
              FAQ interne → knowledge base → appel IA. Prêt pour SIP / Twilio plus tard.
            </p>
          </div>
          <a
            href="#demo"
            className="inline-flex items-center gap-2 whitespace-nowrap rounded-xl bg-primary px-6 py-3 font-semibold text-primary-foreground transition hover:bg-primary/90"
          >
            Lancer la démo <ArrowRight className="size-4" />
          </a>
        </div>

        <footer className="mt-10 flex flex-col items-center justify-between gap-3 pb-4 text-xs text-soft sm:flex-row">
          <span className="font-display font-semibold text-ink">Vantage AI</span>
          <span>© 2026 Vantage AI · Privacy · Security · Status</span>
        </footer>
      </div>
    </div>
  );
}
