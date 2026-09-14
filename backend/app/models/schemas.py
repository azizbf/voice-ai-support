from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TenantStatus = Literal["empty", "processing", "ready", "error"]


class PipelineStatus(BaseModel):
    extracting: bool = False
    creating_knowledge_base: bool = False
    ready: bool = False


class DocumentInfo(BaseModel):
    doc_id: str
    original_name: str
    page_count: int
    chunk_count: int = 0


class TenantCreateResponse(BaseModel):
    tenant_id: str
    token: str
    expires_at: datetime


class TenantStatusResponse(BaseModel):
    tenant_id: str
    status: TenantStatus
    pipeline: PipelineStatus
    document: DocumentInfo | None = None
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    original_name: str
    page_count: int
    status: TenantStatus


class SourceCitation(BaseModel):
    document_name: str
    page: int
    score: float
    chunk_id: str | None = None


class RetrieveRequest(BaseModel):
    tenant_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_name: str
    page: int
    text: str
    score: float


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    latency_ms: float


class ChatRequest(BaseModel):
    tenant_id: str
    message: str
    stream: bool = False


class TtsRequest(BaseModel):
    tenant_id: str
    text: str


class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    latency: LatencyBreakdown


class OkResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    status: str = "ok"
    embedding_provider: str | None = None
    stt_provider: str | None = None
    stt_device: str | None = None
    tts_provider: str | None = None
    embeddings_ready: bool = False
    stt_ready: bool = False
    voice_ready: bool = False
    stt_impl: str | None = None
    whisper_keepalive: bool | None = None


class PageText(BaseModel):
    page: int
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    tenant_id: str
    doc_id: str
    document_name: str
    page: int
    text: str
    token_estimate: int = 0


class TenantMeta(BaseModel):
    tenant_id: str
    created_at: datetime
    expires_at: datetime
    status: TenantStatus = "empty"
    status_detail: str = ""
    document: DocumentInfo | None = None
    pipeline: PipelineStatus = Field(default_factory=PipelineStatus)
    extra: dict[str, Any] = Field(default_factory=dict)
