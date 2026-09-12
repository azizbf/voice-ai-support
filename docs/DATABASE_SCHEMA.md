# Database / Persistence Schema

## Current architecture

| Concern | Store |
|---------|--------|
| Tenants, documents, chunk text, embeddings, conversations, messages, usage | **PostgreSQL + pgvector** (when `DATABASE_URL` is set) |
| Vector search fallback | **FAISS** files under `backend/data/{tenant_id}/index/` if Postgres/pgvector is unavailable |
| Runtime demo status (fast local) | `meta.json` (still written; mirrored to Postgres) |

## PostgreSQL tables

Created automatically on API startup via SQLAlchemy:

- `tenants` — id, name, status, expires_at, timestamps
- `documents` — per-tenant uploaded PDF metadata
- `chunks` — chunk text, page, document_name, `embedding vector(384)` (pgvector)
- `conversations` — chat/voice sessions
- `messages` — user/assistant turns + sources JSON + latency
- `usage_events` — upload/chat/voice events for billing later

Enable the extension once per database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The API also runs this on startup, then creates an HNSW cosine index on `chunks.embedding`.

## Connect locally

```sql
CREATE DATABASE vantage_ai;
CREATE EXTENSION IF NOT EXISTS vector;
```

```env
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/vantage_ai
DASHBOARD_API_KEY=dev-dashboard-key
VECTOR_STORE=auto
```

Docker: `docker compose up db -d` uses `pgvector/pgvector:pg16` →  
`postgresql+asyncpg://vantage:vantage@localhost:5432/vantage_ai`

## Search

Embeddings are L2-normalized (same as the old FAISS inner-product index). Retrieval uses cosine distance:

```sql
ORDER BY embedding <=> :query
```

Score returned to the app is `1 - cosine_distance`.
