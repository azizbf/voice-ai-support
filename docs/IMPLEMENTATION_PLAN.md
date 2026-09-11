# Implementation Plan

## Phase 0 — Scaffold
Monorepo, docs, Docker, env examples, empty FastAPI + Next.js apps boot.

## Phase 1 — PDF upload + extraction
Validate PDF, extract text with page numbers, clean text, status API, frontend Step 1.

## Phase 2 — Chunking + embeddings + FAISS
Intelligent chunking, embedding provider abstraction, persist/load FAISS per tenant.

## Phase 3 — Text RAG chat
`POST /chat` grounded on tenant KB; refuse when insufficient context.

## Phase 4 — Streaming text
SSE/token stream from Claude to UI.

## Phase 5 — Browser voice
WebSocket voice session: STT → RAG → Claude → TTS, barge-in, French agent.

## Phase 6 — Citations
Return and display document name + page for every answer.

## Phase 7 — Polished demo UI
3-step French B2B SaaS experience (upload → prepare → call).

## Phase 8 — Multi-tenant foundations
Tenant create/clear/restart; isolation asserts; session cookies/tokens.

## Phase 9 — Security + rate limits
Limits, cleanup job, safe uploads, CORS, secrets.

## Phase 10 — Latency metrics
Structured logs: speech_end → retrieval → llm_first_token → tts_first_audio → playback.

## Definition of done
No fake core paths. Each phase smoke-tested before the next.
