import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import db
from app.config import settings
from app.errors import register_error_handlers
from app.llm import create_llm_client
from app.routers import recipe
from app.runs import PgRunStore
from app.seed import load_catalog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    app.state.pool = await db.create_pool(settings.database_url)
    applied = await db.apply_migrations(app.state.pool, settings.migrations_dir)
    logger.info("migrations applied: %s", applied)
    # The validated YAML catalog is the solver's source (the DB copy from `app.seed` is for SQL).
    app.state.catalog = load_catalog()
    app.state.llm = create_llm_client(settings)
    app.state.runs = PgRunStore(app.state.pool)
    yield
    await app.state.pool.close()


app = FastAPI(title="Recipe under Constraints", version="0.1.0", lifespan=lifespan)
register_error_handlers(app)
app.include_router(recipe.router)


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    db_ok = await db.ping(request.app.state.pool)
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={
            "status": "ok" if db_ok else "degraded",
            "db": "ok" if db_ok else "error",
            "llm_provider": settings.llm_provider,
        },
    )
