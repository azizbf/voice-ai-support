from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    # Faster model for live voice turns (override with ANTHROPIC_VOICE_MODEL)
    anthropic_voice_model: str = "claude-haiku-4-5"

    voyage_api_key: str = ""
    embedding_provider: str = "auto"  # auto | voyage | local
    voyage_model: str = "voyage-4-lite"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    # Neural French voice (Microsoft edge-tts)
    edge_tts_voice: str = "fr-FR-VivienneMultilingualNeural"

    cors_origins: str = "http://localhost:3000"
    data_dir: str = "./data"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 200
    tenant_ttl_hours: int = 24
    persist_documents: bool = False
    rate_limit_upload: str = "10/minute"
    rate_limit_chat: str = "30/minute"
    rate_limit_voice: str = "20/minute"
    demo_signing_secret: str = "change-me-in-production"
    rag_top_k: int = 3
    rag_min_score: float = 0.25
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 300

    # PostgreSQL — leave empty to run file/FAISS-only mode
    database_url: str = ""
    # Example: postgresql+asyncpg://postgres:postgres@localhost:5432/vantage_ai
    dashboard_api_key: str = "dev-dashboard-key"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def postgres_enabled(self) -> bool:
        return bool(self.database_url.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
