from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.dashboard import router as dashboard_router
from app.api.rag import router as rag_router
from app.api.system import router as system_router
from app.api.tenants import router as tenants_router
from app.api.voice import router as voice_router
from app.core.config import get_settings
from app.core.rate_limit import configure_logging, limiter, request_logging_middleware
from app.db.session import close_db, init_db
from app.services.session.service import session_service

logger = logging.getLogger(__name__)


async def _cleanup_loop() -> None:
    while True:
        try:
            removed = await session_service.cleanup_expired()
            if removed:
                logger.info("Cleaned %s expired tenants", removed)
        except Exception:  # noqa: BLE001
            logger.exception("Cleanup failed")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.data_path.mkdir(parents=True, exist_ok=True)
    try:
        await init_db()
    except Exception:  # noqa: BLE001
        logger.exception("PostgreSQL init failed — continuing in file/FAISS mode")

    # Warm embeddings in background so API accepts requests immediately
    async def _warm_embeddings() -> None:
        try:
            from app.services.rag.embeddings import get_embedding_provider

            await asyncio.to_thread(get_embedding_provider)
            logger.info("Embedding provider ready")
        except Exception:  # noqa: BLE001
            logger.exception("Embedding warm-up failed")

    warm_task = asyncio.create_task(_warm_embeddings())
    async def _warm_voice() -> None:
        try:
            from app.services.voice.stt import warm_stt

            await warm_stt()
            logger.info("Speech recognition ready")
        except Exception:
            logger.exception("Speech recognition warm-up failed")

    voice_warm_task = asyncio.create_task(_warm_voice())
    task = asyncio.create_task(_cleanup_loop())
    logger.info("Voice AI Support API starting (env=%s)", settings.app_env)
    yield
    warm_task.cancel()
    voice_warm_task.cancel()
    task.cancel()
    try:
        await voice_warm_task
    except asyncio.CancelledError:
        pass
    try:
        await warm_task
    except asyncio.CancelledError:
        pass
    try:
        await task
    except asyncio.CancelledError:
        pass
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Voice Customer Support",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS must be outermost so error responses also get ACAO headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.middleware("http")(request_logging_middleware)

    app.include_router(system_router)
    app.include_router(tenants_router)
    app.include_router(rag_router)
    app.include_router(voice_router)
    app.include_router(dashboard_router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        from starlette.exceptions import HTTPException as StarletteHTTPException

        # Prefer FastAPI's HTTPException handler for 4xx/5xx raised on purpose.
        if isinstance(exc, StarletteHTTPException):
            origin = request.headers.get("origin")
            headers: dict[str, str] = {}
            if origin and (
                origin in settings.cors_origin_list
                or origin.startswith("http://localhost:")
                or origin.startswith("http://127.0.0.1:")
            ):
                headers["Access-Control-Allow-Origin"] = origin
                headers["Access-Control-Allow-Credentials"] = "true"
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=headers,
            )

        logger.exception("Unhandled error on %s: %s", request.url.path, exc)
        origin = request.headers.get("origin")
        headers = {}
        if origin and (
            origin in settings.cors_origin_list
            or origin.startswith("http://localhost:")
            or origin.startswith("http://127.0.0.1:")
        ):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        return JSONResponse(
            status_code=500,
            content={"detail": f"Erreur interne: {exc}"},
            headers=headers,
        )

    return app


app = create_app()
