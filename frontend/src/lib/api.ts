export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type PipelineStatus = {
  extracting: boolean;
  creating_knowledge_base: boolean;
  ready: boolean;
};

export type TenantStatus = {
  tenant_id: string;
  status: "empty" | "processing" | "ready" | "error";
  pipeline: PipelineStatus;
  document: {
    doc_id: string;
    original_name: string;
    page_count: number;
    chunk_count: number;
  } | null;
  error: string | null;
};

export type Source = {
  document_name: string;
  page: number;
  score: number;
  chunk_id?: string | null;
};

export type TranscriptItem = {
  id: string;
  role: "user" | "assistant";
  text: string;
};

function authHeaders(token: string): HeadersInit {
  return { "X-Tenant-Token": token };
}

export async function createTenant(): Promise<{
  tenant_id: string;
  token: string;
  expires_at: string;
}> {
  const res = await fetch(`${API_URL}/api/v1/tenants`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(tenantId: string, token: string): Promise<TenantStatus> {
  const res = await fetch(`${API_URL}/api/v1/tenants/${tenantId}/status`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadPdf(
  tenantId: string,
  token: string,
  file: File,
): Promise<{ page_count: number; status: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/v1/tenants/${tenantId}/documents`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || "Échec de l'import PDF");
  }
  return res.json();
}

export async function loadDemoKnowledge(
  tenantId: string,
  token: string,
): Promise<{ page_count: number; status: string; original_name: string }> {
  const res = await fetch(`${API_URL}/api/v1/tenants/${tenantId}/demo-knowledge`, {
    method: "POST",
    headers: authHeaders(token),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || "Échec du chargement de la FAQ démo");
  }
  return res.json();
}

export async function clearKnowledge(tenantId: string, token: string): Promise<void> {
  const res = await fetch(`${API_URL}/api/v1/tenants/${tenantId}/knowledge`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function synthesizeSpeech(
  tenantId: string,
  token: string,
  text: string,
): Promise<ArrayBuffer> {
  const res = await fetch(`${API_URL}/api/v1/tts`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ tenant_id: tenantId, text }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || "Échec de la synthèse vocale");
  }
  return res.arrayBuffer();
}

export async function chatStream(
  tenantId: string,
  token: string,
  message: string,
  onToken: (t: string) => void,
  onSources: (s: Source[]) => void,
): Promise<void> {
  const res = await fetch(`${API_URL}/api/v1/chat`, {
    method: "POST",
    headers: {
      ...authHeaders(token),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ tenant_id: tenantId, message, stream: true }),
  });
  if (!res.ok || !res.body) throw new Error("Échec du chat");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const lines = part.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const parsed = JSON.parse(data);
      if (event === "token") onToken(parsed.text);
      if (event === "sources" || event === "meta") {
        if (parsed.sources) onSources(parsed.sources);
      }
    }
  }
}

export function voiceWsUrl(tenantId: string, token: string): string {
  const base = API_URL.replace(/^http/, "ws");
  return `${base}/api/v1/voice/${tenantId}?token=${encodeURIComponent(token)}`;
}

const DASHBOARD_KEY = process.env.NEXT_PUBLIC_DASHBOARD_KEY || "dev-dashboard-key";

function dashboardHeaders(): HeadersInit {
  return { "X-Dashboard-Key": DASHBOARD_KEY };
}

export type DashboardOverview = {
  enabled: boolean;
  message?: string;
  tenants?: number;
  tenants_ready?: number;
  documents?: number;
  chunks?: number;
  conversations?: number;
  messages?: number;
};

export type DashboardTenant = {
  id: string;
  name: string;
  status: string;
  status_detail: string;
  created_at: string | null;
  expires_at: string | null;
  document_count: number;
  message_count: number;
};

export async function getDashboardStatus(): Promise<{
  postgres_configured: boolean;
  postgres_connected: boolean;
  vector_store: string;
  note: string;
}> {
  const res = await fetch(`${API_URL}/api/v1/dashboard/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDashboardOverview(): Promise<DashboardOverview> {
  const res = await fetch(`${API_URL}/api/v1/dashboard/overview`, {
    headers: dashboardHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDashboardTenants(): Promise<DashboardTenant[]> {
  const res = await fetch(`${API_URL}/api/v1/dashboard/tenants`, {
    headers: dashboardHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.tenants || [];
}

export async function getDashboardTenantDetail(tenantId: string) {
  const res = await fetch(`${API_URL}/api/v1/dashboard/tenants/${tenantId}`, {
    headers: dashboardHeaders(),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
