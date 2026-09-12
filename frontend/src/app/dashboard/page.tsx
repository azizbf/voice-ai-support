"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  getDashboardOverview,
  getDashboardStatus,
  getDashboardTenantDetail,
  getDashboardTenants,
  type DashboardOverview,
  type DashboardTenant,
} from "@/lib/api";

export default function DashboardPage() {
  const [status, setStatus] = useState<{
    postgres_configured: boolean;
    postgres_connected: boolean;
    vector_store: string;
    note: string;
  } | null>(null);
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [tenants, setTenants] = useState<DashboardTenant[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const st = await getDashboardStatus();
      setStatus(st);
      const ov = await getDashboardOverview();
      setOverview(ov);
      if (st.postgres_connected && ov.enabled) {
        const list = await getDashboardTenants();
        setTenants(list);
      } else {
        setTenants([]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur dashboard");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    void getDashboardTenantDetail(selected)
      .then(setDetail)
      .catch((err) => setError(err instanceof Error ? err.message : "Erreur détail"));
  }, [selected]);

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
            <span className="ml-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-soft">
              Dashboard
            </span>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => void refresh()}
              className="rounded-xl bg-white/50 px-4 py-2 text-sm font-semibold text-ink"
            >
              Refresh
            </button>
            <Link
              href="/"
              className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground"
            >
              Demo
            </Link>
          </div>
        </nav>

        {error && (
          <div className="glass mt-4 rounded-2xl px-4 py-3 text-sm text-ink">{error}</div>
        )}

        <div className="mt-8">
          <h1 className="font-display text-3xl font-semibold text-ink">Operations console</h1>
          <p className="mt-2 max-w-2xl text-soft">
            PostgreSQL holds tenants, documents, chunk text, embeddings (pgvector), conversations
            and messages. FAISS files are used only if Postgres is unavailable.
          </p>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="glass rounded-3xl p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-soft">PostgreSQL</p>
            <p className="mt-2 font-display text-2xl font-semibold text-ink">
              {status?.postgres_connected ? "Connected" : status?.postgres_configured ? "Configured" : "Offline"}
            </p>
            <p className="mt-1 text-xs text-soft">
              {status?.postgres_connected
                ? "Schema ready"
                : "Set DATABASE_URL then restart API"}
            </p>
          </div>
          <div className="glass rounded-3xl p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-soft">Vector store</p>
            <p className="mt-2 font-display text-2xl font-semibold text-ink">
              {(status?.vector_store || "faiss").toUpperCase()}
            </p>
            <p className="mt-1 text-xs text-soft">
              {(status?.vector_store || "faiss") === "pgvector"
                ? "Embeddings in PostgreSQL"
                : "Per-tenant index on disk"}
            </p>
          </div>
          <div className="glass rounded-3xl p-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-soft">Ready tenants</p>
            <p className="mt-2 font-display text-2xl font-semibold text-ink">
              {overview?.tenants_ready ?? "—"} / {overview?.tenants ?? "—"}
            </p>
            <p className="mt-1 text-xs text-soft">KB indexed sessions</p>
          </div>
        </div>

        {overview?.enabled && (
          <div className="mt-4 grid gap-4 sm:grid-cols-4">
            {[
              ["Documents", overview.documents],
              ["Chunks", overview.chunks],
              ["Conversations", overview.conversations],
              ["Messages", overview.messages],
            ].map(([label, value]) => (
              <div key={String(label)} className="glass-soft rounded-2xl px-4 py-3">
                <p className="text-xs text-soft">{label}</p>
                <p className="font-display text-xl font-semibold text-ink">{value as number}</p>
              </div>
            ))}
          </div>
        )}

        {!status?.postgres_connected && !loading && (
          <div className="glass mt-6 rounded-3xl p-6 text-sm leading-relaxed text-ink">
            <p className="font-display text-lg font-semibold">Connect PostgreSQL</p>
            <ol className="mt-3 list-decimal space-y-2 pl-5 text-soft">
              <li>Finish installing PostgreSQL and start the service.</li>
              <li>
                Create a database:{" "}
                <code className="rounded bg-white/60 px-1.5 py-0.5 text-ink">
                  CREATE DATABASE vantage_ai;
                </code>
              </li>
              <li>
                Set in <code className="rounded bg-white/60 px-1.5 py-0.5 text-ink">backend/.env</code>:
                <br />
                <code className="mt-1 block rounded bg-white/60 px-2 py-1 text-xs text-ink">
                  DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/vantage_ai
                </code>
              </li>
              <li>Restart the FastAPI backend — tables are created automatically.</li>
              <li>Run a demo upload/chat — rows will appear here.</li>
            </ol>
            <p className="mt-4 text-xs text-soft">
              Or with Docker: <code>docker compose up db -d</code> then use
              <code className="ml-1">postgresql+asyncpg://vantage:vantage@localhost:5432/vantage_ai</code>
            </p>
          </div>
        )}

        {status?.postgres_connected && (
          <div className="mt-6 grid gap-5 lg:grid-cols-12">
            <div className="glass rounded-3xl p-5 lg:col-span-5">
              <h2 className="font-display text-lg font-semibold text-ink">Tenants</h2>
              <div className="mt-4 max-h-[480px] space-y-2 overflow-y-auto">
                {tenants.length === 0 && (
                  <p className="text-sm text-soft">No tenants yet — use the demo first.</p>
                )}
                {tenants.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setSelected(t.id)}
                    className={`w-full rounded-2xl px-3.5 py-3 text-left transition ${
                      selected === t.id ? "bg-primary text-primary-foreground" : "glass-soft text-ink"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold">{t.name}</span>
                      <span className="text-xs opacity-80">{t.status}</span>
                    </div>
                    <p className="mt-1 truncate text-xs opacity-70">{t.id}</p>
                    <p className="mt-1 text-xs opacity-70">
                      {t.document_count} docs · {t.message_count} msgs
                    </p>
                  </button>
                ))}
              </div>
            </div>

            <div className="glass rounded-3xl p-5 lg:col-span-7">
              <h2 className="font-display text-lg font-semibold text-ink">Tenant detail</h2>
              {!selected && (
                <p className="mt-4 text-sm text-soft">Select a tenant to inspect documents and messages.</p>
              )}
              {detail && (
                <div className="mt-4 space-y-5">
                  <div>
                    <p className="text-xs font-semibold uppercase text-soft">Documents</p>
                    <ul className="mt-2 space-y-2">
                      {((detail.documents as Array<Record<string, unknown>>) || []).map((d) => (
                        <li key={String(d.id)} className="glass-soft rounded-xl px-3 py-2 text-sm">
                          <span className="font-medium text-ink">{String(d.original_name)}</span>
                          <span className="ml-2 text-soft">
                            {String(d.page_count)} pages · {String(d.chunk_count)} chunks
                          </span>
                        </li>
                      ))}
                      {!(detail.documents as unknown[])?.length && (
                        <li className="text-sm text-soft">No documents</li>
                      )}
                    </ul>
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase text-soft">Recent messages</p>
                    <ul className="mt-2 max-h-[320px] space-y-2 overflow-y-auto">
                      {((detail.recent_messages as Array<Record<string, unknown>>) || []).map((m) => (
                        <li key={String(m.id)} className="glass-soft rounded-xl px-3 py-2 text-sm">
                          <span className="text-xs font-semibold uppercase text-accent">
                            {String(m.role)}
                          </span>
                          <p className="mt-1 text-ink">{String(m.content)}</p>
                        </li>
                      ))}
                      {!(detail.recent_messages as unknown[])?.length && (
                        <li className="text-sm text-soft">No messages yet</li>
                      )}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <footer className="mt-10 pb-4 text-xs text-soft">
          Dashboard key header: <code>X-Dashboard-Key</code> · env{" "}
          <code>NEXT_PUBLIC_DASHBOARD_KEY</code>
        </footer>
      </div>
    </div>
  );
}
