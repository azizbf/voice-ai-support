# API Specification

Base URL: `http://localhost:8000`

Auth (demo): `X-Tenant-Token` header (signed token issued on tenant create) or returned cookie.

## Health

`GET /health` → `{ "status": "ok" }`

## Tenants

### Create demo tenant
`POST /api/v1/tenants`

Response:
```json
{
  "tenant_id": "uuid",
  "token": "signed-token",
  "expires_at": "ISO-8601"
}
```

### Status
`GET /api/v1/tenants/{tenant_id}/status`

```json
{
  "tenant_id": "uuid",
  "status": "processing",
  "pipeline": {
    "extracting": true,
    "creating_knowledge_base": true,
    "ready": false
  },
  "document": null,
  "error": null
}
```

### Clear knowledge base
`DELETE /api/v1/tenants/{tenant_id}/knowledge` → `{ "ok": true }`

## Documents

`POST /api/v1/tenants/{tenant_id}/documents`  
`multipart/form-data` field `file` (PDF, max 10MB)

Response:
```json
{
  "doc_id": "uuid",
  "original_name": "faq.pdf",
  "page_count": 12,
  "status": "processing"
}
```

Processing continues asynchronously; poll status until `ready`.

## RAG retrieve (internal / voice tool)

`POST /api/v1/rag/retrieve`

```json
{
  "tenant_id": "uuid",
  "query": "Ma connexion ne fonctionne plus",
  "top_k": 5
}
```

```json
{
  "chunks": [
    {
      "chunk_id": "uuid",
      "document_name": "support_internet.pdf",
      "page": 14,
      "text": "...",
      "score": 0.82
    }
  ],
  "latency_ms": 45
}
```

## Chat (text RAG)

`POST /api/v1/chat`

```json
{
  "tenant_id": "uuid",
  "message": "Comment réinitialiser mon routeur ?",
  "stream": false
}
```

Non-stream response:
```json
{
  "answer": "Selon la procédure...",
  "sources": [{"document_name": "faq.pdf", "page": 3, "score": 0.79}],
  "latency": {
    "retrieval_ms": 40,
    "llm_ms": 820,
    "total_ms": 880
  }
}
```

`stream: true` → `text/event-stream` events: `meta`, `token`, `sources`, `done`, `error`.

## Voice WebSocket

`WS /api/v1/voice/{tenant_id}?token=...`

Client → server:
- `{ "type": "user_transcript", "text": "..." }` (browser Web Speech API, default)
- binary audio frames + `{ "type": "audio_end" }` (optional server STT: Deepgram / Whisper)
- `{ "type": "interrupt" }`
- `{ "type": "ping" }`

Server → client:
- `{ "type": "transcript", "role": "user"|"assistant", "text": "...", "final": true }`
- `{ "type": "sources", "items": [...] }`
- `{ "type": "latency", ... }`
- `{ "type": "status", "state": "listening"|"thinking"|"speaking" }`
- binary TTS audio frames (PCM/mp3 depending on provider)
- `{ "type": "error", "message": "..." }`

## Metrics (debug)

`GET /api/v1/metrics/recent` → last N latency samples (dev only).
