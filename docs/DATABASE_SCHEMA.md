# Database / Persistence Schema

## Current architecture (hybrid)

| Concern | Store |
|---------|--------|
| Tenants, documents, chunk text, conversations, messages, usage | **PostgreSQL** (when `DATABASE_URL` is set) |
| Vector embeddings / similarity search | **FAISS** files under `backend/data/{tenant_id}/index/` |
| Runtime demo status (fast local) | `meta.json` (still written; mirrored to Postgres) |

## PostgreSQL tables

Created automatically on API startup via SQLAlchemy:

- `tenants` — id, name, status, expires_at, timestamps
- `documents` — per-tenant uploaded PDF metadata
- `chunks` — chunk text, page, document_name, `faiss_row` index
- `conversations` — chat/voice sessions
- `messages` — user/assistant turns + sources JSON + latency
- `usage_events` — upload/chat/voice events for billing later

## Connect locally

```sql
CREATE DATABASE vantage_ai;
```

```env
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/vantage_ai
DASHBOARD_API_KEY=dev-dashboard-key
```

Or Docker: `docker compose up db -d` then use  
`postgresql+asyncpg://vantage:vantage@localhost:5432/vantage_ai`

## Future: pgvector

Replace FAISS by storing `embedding VECTOR(n)` on `chunks` and using cosine search in SQL. The `VectorStore` abstraction is already in place for that swap.
