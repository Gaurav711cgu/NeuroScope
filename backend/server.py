"""NeuroScope FastAPI server — mounts /api/* and /api/v1/* endpoints.

Lightweight entrypoint routing domain endpoints to modular API routers.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

from core.logging_config import configure_logging
from db import session
from api.routers import meta, runs, experiments, steering
from neuroscope.worker import start_worker, stop_worker

# Initialize logging config
configure_logging()
logger = logging.getLogger("neuroscope.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database connection pool
    await session.init_db()
    # Start the transactional outbox background worker
    start_worker()
    logger.info("NeuroScope server startup complete.")
    yield
    # Stop background polling worker
    stop_worker()
    # Close database pool
    await session.close_pool()
    logger.info("NeuroScope server shutdown complete.")


app = FastAPI(title="NeuroScope API", version="3.1.0", lifespan=lifespan)

# Build unified internal router
core_router = APIRouter()
core_router.include_router(meta.router)
core_router.include_router(runs.router)
core_router.include_router(experiments.router)
core_router.include_router(steering.router)

# Mount /api/ prefix
api_router = APIRouter(prefix="/api")
api_router.include_router(core_router)

# Mount /api/v1/ prefix
v1_router = APIRouter(prefix="/v1")
v1_router.include_router(core_router)
api_router.include_router(v1_router)

# Expose composite routers to application
app.include_router(api_router)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception):
    logger.exception("Unhandled server exception")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"}
    )
