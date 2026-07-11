"""PostgreSQL client database operations utilizing asyncpg.

Delegates connection pool lifecycle to db/session.py.
"""
from __future__ import annotations

import json
import logging
import uuid
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
import asyncpg
from fastapi import HTTPException

from db.session import get_pool, init_db, close_pool

logger = logging.getLogger("neuroscope.db")


def _key_to_int(key: str | uuid.UUID) -> int:
    """Hash a string/UUID to a signed 31-bit integer for PG advisory locks."""
    if isinstance(key, uuid.UUID):
        return key.int & 0x7FFFFFFF
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) & 0x7FFFFFFF


@asynccontextmanager
async def advisory_lock(key: str | uuid.UUID):
    """Context manager acquiring a PostgreSQL session-level advisory lock.
    
    Fails fast with HTTP 409 if lock is not available.
    """
    lock_id = _key_to_int(key)
    pool = await get_pool()
    async with pool.acquire() as conn:
        locked = await conn.fetchval("SELECT pg_try_advisory_lock($1)", lock_id)
        if not locked:
            logger.warning("Failed to acquire advisory lock for resource %s", key)
            raise HTTPException(
                status_code=409,
                detail=f"Resource {key} is currently locked by another background worker."
            )
        logger.debug("Acquired advisory lock for resource %s", key)
        try:
            yield conn
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", lock_id)
            logger.debug("Released advisory lock for resource %s", key)


# --- Outbox Pattern Operators ---

async def save_outbox_event(conn, event_id: uuid.UUID, event_type: str, payload: dict):
    """Atomic outbox insertion, designed to run within an existing connection/transaction."""
    payload_json = json.dumps(payload)
    await conn.execute(
        """
        INSERT INTO outbox (id, event_type, payload, status)
        VALUES ($1, $2, $3, 'pending')
        """,
        event_id, event_type, payload_json
    )


async def claim_next_outbox_event() -> dict | None:
    """Fetch next pending outbox task using SELECT FOR UPDATE SKIP LOCKED to prevent race conditions."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, event_type, payload
                FROM outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """
            )
            if not row:
                return None
            
            await conn.execute(
                """
                UPDATE outbox
                SET status = 'processing', processed_at = NOW()
                WHERE id = $1
                """,
                row["id"]
            )
            return {
                "id": str(row["id"]),
                "event_type": row["event_type"],
                "payload": json.loads(row["payload"])
            }


async def update_outbox_status(event_id: str, status: str, error: str = None):
    """Mark an outbox task execution status."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE outbox
        SET status = $2, error = $3, processed_at = NOW()
        WHERE id = $1
        """,
        uuid.UUID(event_id), status, error
    )


# --- Relational Query Operations ---

async def save_run(run_id: str, task: str, model_name: str, n_steps: int, sae_layer: int, status: str = "queued", progress: dict = None, correct: bool = None):
    pool = await get_pool()
    progress_json = json.dumps(progress) if progress else None
    await pool.execute(
        """
        INSERT INTO runs (id, task, model_name, n_steps, sae_layer, status, progress, correct)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (id) DO UPDATE SET
            status = EXCLUDED.status,
            progress = EXCLUDED.progress,
            correct = COALESCE(EXCLUDED.correct, runs.correct)
        """,
        uuid.UUID(run_id), task, model_name, n_steps, sae_layer, status, progress_json, correct
    )


async def update_run(run_id: str, fields: dict):
    """Update runs matching fields. Uses advisory_lock or FOR UPDATE constraints where needed."""
    pool = await get_pool()
    keys = list(fields.keys())
    if not keys:
        return
    values = []
    set_clauses = []
    for i, k in enumerate(keys):
        val = fields[k]
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        values.append(val)
        set_clauses.append(f"{k} = ${i+2}")
    
    query = f"UPDATE runs SET {', '.join(set_clauses)} WHERE id = $1"
    await pool.execute(query, uuid.UUID(run_id), *values)


async def get_run_with_lock(run_id: str) -> dict | None:
    """Retrieve run record utilizing SELECT FOR UPDATE NOWAIT to fail fast on conflict."""
    pool = await get_pool()
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        return None
    
    try:
        r = await pool.fetchrow("SELECT * FROM runs WHERE id = $1 FOR UPDATE NOWAIT", run_uuid)
    except asyncpg.exceptions.LockNotAvailableError:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is currently locked and processing. Please try again later."
        )
    if not r:
        return None
    return await _inflate_run(r, run_uuid)


async def get_run(run_id: str) -> dict | None:
    pool = await get_pool()
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        return None
    r = await pool.fetchrow("SELECT * FROM runs WHERE id = $1", run_uuid)
    if not r:
        return None
    return await _inflate_run(r, run_uuid)


async def _inflate_run(r: asyncpg.Record, run_uuid: uuid.UUID) -> dict:
    pool = await get_pool()
    step_rows = await pool.fetch("SELECT * FROM steps WHERE run_id = $1 ORDER BY step_n", run_uuid)
    steps = []
    for s in step_rows:
        feat_rows = await pool.fetch("SELECT feature_id, activation, drift_score FROM step_features WHERE step_id = $1 ORDER BY activation DESC", s["id"])
        top_feats = [{
            "feature_id": f["feature_id"],
            "activation": f["activation"],
            "drift_score": f["drift_score"]
        } for f in feat_rows]
        
        steps.append({
            "step_n": s["step_n"],
            "prompt": s["prompt"],
            "output": s["output"],
            "tool_called": s["tool_called"],
            "n_active_features": s["n_active_features"],
            "sae_l2_norm": s["sae_l2_norm"],
            "elapsed_ms": s["elapsed_ms"],
            "activation_path": s["activation_path"],
            "layer_l2_norms": json.loads(s["layer_l2_norms"]) if s["layer_l2_norms"] else [],
            "hallucination": json.loads(s["hallucination"]) if s["hallucination"] else {},
            "top_features": top_feats
        })
    
    patch_rows = await pool.fetch("SELECT * FROM patch_results WHERE run_id = $1 ORDER BY source_step, target_step", run_uuid)
    patch_matrix = []
    for p in patch_rows:
        patch_matrix.append({
            "source_step": p["source_step"],
            "target_step": p["target_step"],
            "patch_layer": p["patch_layer"],
            "kl": p["kl"],
            "significant": p["significant"],
            "top_token_change": json.loads(p["top_token_change"]) if p["top_token_change"] else None
        })
    
    return {
        "id": str(r["id"]),
        "task": r["task"],
        "model_name": r["model_name"],
        "n_steps": r["n_steps"],
        "sae_layer": r["sae_layer"],
        "status": r["status"],
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "total_elapsed_ms": r["total_elapsed_ms"],
        "correct": r["correct"],
        "progress": json.loads(r["progress"]) if r["progress"] else None,
        "error": r["error"],
        "steps": steps,
        "feature_timelines": json.loads(r["feature_timelines"]) if r["feature_timelines"] else [],
        "patch_matrix": patch_matrix,
        "patch_matrix_summary": json.loads(r["patch_matrix_summary"]) if r["patch_matrix_summary"] else None
    }


async def save_step(
    step_id: str,
    run_id: str,
    step_n: int,
    prompt: str,
    output: str,
    tool_called: str,
    n_active_features: int,
    sae_l2_norm: float,
    hallucination: dict,
    elapsed_ms: int,
    activation_path: str,
    top_features: list[dict],
    layer_l2_norms: list[float]
):
    pool = await get_pool()
    h_score = hallucination.get("composite", 0.0)
    entropy = hallucination.get("entropy", 0.0)
    attn_diffusion = hallucination.get("attention_diffusion", 0.0)
    drift_score = hallucination.get("drift_proxy", 0.0)
    
    layer_l2_norms_json = json.dumps(layer_l2_norms)
    hallucination_json = json.dumps(hallucination)
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO steps (id, run_id, step_n, prompt, output, tool_called, n_active_features, sae_l2_norm, hallucination_score, entropy, attn_diffusion, drift_score, elapsed_ms, activation_path, layer_l2_norms, hallucination)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (id) DO NOTHING
                """,
                uuid.UUID(step_id), uuid.UUID(run_id), step_n, prompt, output, tool_called, n_active_features, sae_l2_norm, h_score, entropy, attn_diffusion, drift_score, elapsed_ms, activation_path, layer_l2_norms_json, hallucination_json
            )
            if top_features:
                records = [
                    (uuid.UUID(step_id), int(f["feature_id"]), float(f["activation"]), float(f.get("drift_score", 0.0)))
                    for f in top_features
                ]
                await conn.executemany(
                    """
                    INSERT INTO step_features (step_id, feature_id, activation, drift_score)
                    VALUES ($1, $2, $3, $4)
                    """,
                    records
                )


async def save_patch(
    run_id: str,
    source_step: int,
    target_step: int,
    patch_layer: int,
    kl: float,
    significant: bool,
    top_token_change: dict
):
    pool = await get_pool()
    top_token_change_json = json.dumps(top_token_change) if top_token_change else None
    await pool.execute(
        """
        INSERT INTO patch_results (run_id, source_step, target_step, patch_layer, kl, significant, top_token_change)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        uuid.UUID(run_id), source_step, target_step, patch_layer, kl, significant, top_token_change_json
    )


async def list_runs(limit: int = 30) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, task, model_name, status, created_at, n_steps, sae_layer, progress, error, total_elapsed_ms, correct
        FROM runs
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit
    )
    res = []
    for r in rows:
        res.append({
            "id": str(r["id"]),
            "task": r["task"],
            "model_name": r["model_name"],
            "status": r["status"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "n_steps": r["n_steps"],
            "sae_layer": r["sae_layer"],
            "progress": json.loads(r["progress"]) if r["progress"] else None,
            "error": r["error"],
            "total_elapsed_ms": r["total_elapsed_ms"],
            "correct": r["correct"]
        })
    return res


async def get_experiment(slug: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM experiments WHERE slug = $1", slug)
    if not row:
        return None
    data = json.loads(row["data"]) if row["data"] else {}
    return {
        "id": row["slug"],
        "slug": row["slug"],
        "title": row["title"],
        "category": row["category"],
        "hypothesis": row["hypothesis"],
        "task": row["task"],
        "n_steps": row["n_steps"],
        "sae_layer": row["sae_layer"],
        "model": row["model"],
        "finding_seed": row["finding_seed"],
        "finding": row["finding"],
        "total_elapsed_ms": row["total_elapsed_ms"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "steps": data.get("steps", []),
        "feature_timelines": data.get("feature_timelines", []),
        "patch_matrix": data.get("patch_matrix", []),
        "patch_matrix_summary": data.get("patch_matrix_summary", {}),
        "_is_experiment": True
    }


async def list_experiments() -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT slug, title, category, hypothesis, finding, task, n_steps, sae_layer, model, total_elapsed_ms, created_at FROM experiments ORDER BY created_at DESC")
    experiments = []
    for r in rows:
        experiments.append({
            "id": r["slug"],
            "slug": r["slug"],
            "title": r["title"],
            "category": r["category"],
            "hypothesis": r["hypothesis"],
            "finding": r["finding"],
            "task": r["task"],
            "n_steps": r["n_steps"],
            "sae_layer": r["sae_layer"],
            "model": r["model"],
            "total_elapsed_ms": r["total_elapsed_ms"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None
        })
    return experiments


async def save_experiment(
    slug: str,
    title: str,
    category: str,
    hypothesis: str,
    task: str,
    n_steps: int,
    sae_layer: int,
    model: str,
    finding_seed: str,
    finding: str,
    total_elapsed_ms: int,
    data: dict
):
    pool = await get_pool()
    data_json = json.dumps(data)
    await pool.execute(
        """
        INSERT INTO experiments (slug, title, category, hypothesis, task, n_steps, sae_layer, model, finding_seed, finding, total_elapsed_ms, data)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (slug) DO UPDATE SET
            title = EXCLUDED.title,
            category = EXCLUDED.category,
            hypothesis = EXCLUDED.hypothesis,
            task = EXCLUDED.task,
            n_steps = EXCLUDED.n_steps,
            sae_layer = EXCLUDED.sae_layer,
            model = EXCLUDED.model,
            finding_seed = EXCLUDED.finding_seed,
            finding = EXCLUDED.finding,
            total_elapsed_ms = EXCLUDED.total_elapsed_ms,
            data = EXCLUDED.data
        """,
        slug, title, category, hypothesis, task, n_steps, sae_layer, model, finding_seed, finding, total_elapsed_ms, data_json
    )


async def save_query(query_id: str, run_id: str, query: str, answer: str):
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO queries (id, run_id, query, answer)
        VALUES ($1, $2, $3, $4)
        """,
        uuid.UUID(query_id), run_id, query, answer
    )


async def list_queries(run_id: str) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, run_id, query, answer, created_at
        FROM queries
        WHERE run_id = $1
        ORDER BY created_at DESC
        """,
        run_id
    )
    return [{
        "id": str(r["id"]),
        "run_id": r["run_id"],
        "query": r["query"],
        "answer": r["answer"],
        "created_at": r["created_at"].isoformat() if r["created_at"] else None
    } for r in rows]


async def get_feature_label(layer: int, feature_id: int) -> dict | None:
    pool = await get_pool()
    doc_id = f"l{layer}_f{feature_id}"
    row = await pool.fetchrow("SELECT * FROM feature_labels WHERE id = $1", doc_id)
    if not row:
        return None
    return {
        "layer": row["layer"],
        "feature_id": row["feature_id"],
        "label": row["label"],
        "neuronpedia_url": row["neuronpedia_url"]
    }


async def save_feature_label(layer: int, feature_id: int, label: str, neuronpedia_url: str):
    pool = await get_pool()
    doc_id = f"l{layer}_f{feature_id}"
    await pool.execute(
        """
        INSERT INTO feature_labels (id, layer, feature_id, label, neuronpedia_url)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (id) DO UPDATE SET
            label = EXCLUDED.label,
            neuronpedia_url = EXCLUDED.neuronpedia_url
        """,
        doc_id, layer, feature_id, label, neuronpedia_url
    )
