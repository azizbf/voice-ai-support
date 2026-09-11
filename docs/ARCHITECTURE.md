# Architecture

## Product

Browser demo for Tunisian French-speaking call centers:

1. Upload a company PDF FAQ/procedure document.
2. Process once into a tenant-scoped RAG knowledge base (FAISS).
3. Start a voice call in the browser.
4. Speak to a Claude-powered French support agent grounded only in that KB.
5. See live transcript and document/page citations.

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js (App Router), TypeScript, Tailwind |
| Backend | FastAPI, Python 3.11+ |
| LLM | Anthropic Claude (streaming) |
| Embeddings | Voyage AI if `VOYAGE_API_KEY`, else local `sentence-transformers` |
| Vector store | FAISS behind `VectorStore` abstraction |
| STT | Deepgram streaming if key set, else `faster-whisper` |
| TTS | ElevenLabs streaming if key set, else `edge-tts` |
| Voice transport | Browser mic + WebSocket audio/events pipeline |

Anthropic has no speech-to-speech Realtime API and no embeddings API. Voice is a **chained** pipeline optimized for first-audio latency.

## High-level flow

```
PDF upload → extract (pages) → clean → chunk → embed → FAISS persist (once)
                                                              │
Customer speech → STT → retrieve(tenant) → Claude stream → TTS stream → speaker
                              │                              │
                         citations UI                   barge-in cancel
```

## Multi-tenancy

Every demo session creates a `tenant_id` (UUID). Indexes, uploads, metadata, and retrieval are strictly scoped to that ID. Cross-tenant search is impossible by construction (separate FAISS files + metadata filter + assert).

## RAG contract

- Ingest runs **once** after upload; queries never re-embed the PDF.
- Retrieval returns top-k chunks with scores, document name, and page.
- Claude system prompt forbids inventing policies; if context is insufficient, answer in French recommending human escalation.

## Voice contract

1. Client opens `WS /api/v1/voice/{tenant_id}`.
2. Default STT: browser Web Speech API (`fr-FR`) sends `{ type: "user_transcript", text }`.
3. Optional: client streams audio; server uses Deepgram or faster-whisper when configured.
4. On final transcript: retrieval → Claude stream → sentence-flush TTS → binary audio frames.
5. On barge-in (`interrupt` or new speech while speaking): cancel TTS/Claude generation, clear playback.

## Security

- API keys only on backend.
- PDF validation, size limits, sanitized filenames.
- Rate limits on upload/chat/voice.
- Signed demo tenant tokens.
- TTL cleanup of `data/{tenant_id}` unless `PERSIST_DOCUMENTS=true`.

## Extension points

- Replace `FaissVectorStore` with pgvector / Pinecone / Qdrant.
- Swap STT/TTS providers via service interfaces.
- Add Twilio/SIP media bridge on the same Claude+RAG core.
- Company dashboards, billing meters, CRM webhooks, Tunisian Arabic locale packs.
