# Auralis — AI Voice Customer Support

Production-oriented MVP for Tunisian call centers serving French-speaking customers.

**Flow:** Upload PDF → build tenant RAG knowledge base once → talk to a Claude-powered French voice agent grounded only in that document → see live transcript and page citations.

> **Note:** Anthropic Claude is used for reasoning. Anthropic has no Realtime speech API or embeddings API, so voice uses a low-latency chained pipeline (STT → RAG → Claude stream → TTS), and embeddings use Voyage (optional) or local `sentence-transformers`.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
frontend (Next.js)  ←→  backend (FastAPI)
                           ├─ PDF extract / chunk
                           ├─ Embeddings → FAISS (per tenant)
                           ├─ Claude chat + SSE stream
                           └─ WebSocket voice (STT → Claude → TTS)
```

## Quick start (local)

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy ..\.env.example .env   # Windows
# cp ../.env.example .env   # macOS/Linux

# Set at least:
# ANTHROPIC_API_KEY=...

uvicorn app.main:app --reload --port 8000
```

First run downloads the local embedding model (and Whisper if no Deepgram key).

### 2. Frontend

```bash
cd frontend
npm install
copy ..\.env.example .env.local
# ensure NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000

### Docker

```bash
# Put secrets in backend/.env
docker compose up --build
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `ANTHROPIC_MODEL` | No | Default `claude-sonnet-5` |
| `VOYAGE_API_KEY` | No | If set, used for embeddings |
| `DEEPGRAM_API_KEY` | No | Streaming-quality French STT |
| `ELEVENLABS_API_KEY` | No | Higher-quality French TTS |
| `DEMO_SIGNING_SECRET` | Yes in prod | Signs tenant demo tokens |
| `CORS_ORIGINS` | No | Default `http://localhost:3000` |
| `TENANT_TTL_HOURS` | No | Demo cleanup (default 24) |
| `PERSIST_DOCUMENTS` | No | Keep PDFs on disk (`false` by default) |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend base URL |

Fallbacks without optional keys:

- Embeddings → local multilingual MiniLM
- STT → `faster-whisper` (French)
- TTS → Microsoft Edge neural voice via `edge-tts` (`fr-FR-DeniseNeural`)

## How RAG works

1. PDF validated (type, size, magic bytes).
2. Text extracted **with page numbers**.
3. Cleaned and chunked (~1800 chars, overlap).
4. Embedded once and stored in a **tenant-scoped FAISS** index + `chunks.jsonl` metadata.
5. Queries embed the question, search that tenant only, pass top chunks to Claude.
6. If nothing relevant: Claude is instructed to say it lacks information and recommend human escalation.

Indexes are persisted under `backend/data/{tenant_id}/` and **not** rebuilt per question.

## How voice works

1. Browser obtains mic permission and opens `WS /api/v1/voice/{tenant_id}`.
2. Speech is transcribed in the browser (Web Speech API, French) and sent as text — low latency, no Whisper required.
3. Optional server STT: set `DEEPGRAM_API_KEY` (or install `faster-whisper`) and send audio frames instead.
4. Tenant RAG retrieval runs.
5. Claude streams text; completed sentences are flushed to TTS immediately (first audio before full answer).
6. Client plays audio chunks; barge-in sends `interrupt` and stops playback.

Use **Chrome or Edge** for the best Web Speech experience.

## Multi-tenant design

Each demo session creates a `tenant_id`. FAISS files and metadata are isolated. Retrieval asserts tenant match. Never shares chunks across companies.

## Security

- API keys only on the backend
- PDF validation + size limits + safe filenames
- Rate limits (SlowAPI) on upload / chat / voice paths
- Signed tenant tokens (`X-Tenant-Token`)
- Temporary demo data with TTL cleanup

## PostgreSQL + dashboard

1. Install PostgreSQL and create a database:
   ```sql
   CREATE DATABASE vantage_ai;
   ```
2. Set in `backend/.env` (adjust user/password):
   ```env
   DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/vantage_ai
   DASHBOARD_API_KEY=dev-dashboard-key
   ```
3. Restart the API — tables are created automatically.
4. Open http://localhost:3000/dashboard

Postgres stores tenants, documents, chunk text, conversations, messages.  
Vectors stay in FAISS for now (pgvector can replace them later).

Docker DB only: `docker compose up db -d` → URL  
`postgresql+asyncpg://vantage:vantage@localhost:5432/vantage_ai`

## Deploy

- Backend: any Linux host / container with Python 3.11+, expose 8000, set secrets, persistent volume optional for `DATA_DIR`.
- Frontend: Vercel/Node host with `NEXT_PUBLIC_API_URL` pointing at the API; configure CORS accordingly.
- Prefer putting TLS termination (nginx/Caddy) in front of both.

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- [DATABASE_SCHEMA.md](docs/DATABASE_SCHEMA.md)
- [API_SPEC.md](docs/API_SPEC.md)

## Extension roadmap

Real phone calls (SIP/Twilio), CRM, human transfer, call recording, analytics, Tunisian Arabic, company dashboards, usage billing — same Claude + RAG core with new adapters.
