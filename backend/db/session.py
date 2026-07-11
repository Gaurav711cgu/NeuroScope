"""PostgreSQL database session and pool lifecycle manager."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import asyncpg

logger = logging.getLogger("neuroscope.db.session")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/neuroscope")

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the global asyncpg connection pool singleton, initializing if necessary."""
    global _pool
    if _pool is None:
        logger.info("Initializing asyncpg connection pool with DSN: %s", DATABASE_URL)
        _pool = await asyncpg.create_pool(DATABASE_URL)
    return _pool


async def close_pool():
    """Gracefully shutdown the global asyncpg connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed.")


async def init_db():
    """Execute schema.sql to ensure all relational tables are initialized."""
    pool = await get_pool()
    schema_path = Path(__file__).parent.parent / "neuroscope" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    logger.info("Database schema initialized successfully.")
