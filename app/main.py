"""FastAPI entry point. Run: uvicorn app.main:app --reload"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import Settings, get_settings
from app.database import init_db
from app.logging_config import setup_logging
from app.routers import auth, research, stocks, watchlist

logger = logging.getLogger("nepse")


def check_settings(settings: Settings) -> None:
    """Refuse to start in production with a weak secret key."""
    if settings.environment == "production" and (
        settings.secret_key == "change-me" or len(settings.secret_key) < 32
    ):
        raise RuntimeError("SECRET_KEY must be at least 32 characters in production")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    check_settings(settings)
    init_db()
    yield


app = FastAPI(
    title="NEPSE Stock API",
    version="1.2.0",
    description="Price history, next-day forecasts and watchlists for NEPSE stocks. "
    "Educational project, not financial advice.",
    lifespan=lifespan,
)

_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(auth.router)
app.include_router(stocks.router)
app.include_router(research.router)
app.include_router(watchlist.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}