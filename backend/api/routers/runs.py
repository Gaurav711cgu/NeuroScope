"""Runs router mounting endpoints for run telemetry, patching, causal attribution, and NL query logs."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neuroscope import db
from neuroscope import llm as ns_llm
from neuroscope.loader import get_model, model_info
from neuroscope.patching import cross_step_patch, causal_attribution_for_step
from neuroscope.runner import patch_matrix

logger = logging.getLogger("neuroscope.api.runs")

router = APIRouter()


# --- Schemas ---

class RunCreate(BaseModel):
    task: str
    n_steps: int = Field(default=3, ge=2, le=6)
    sae_layer: int = Field(default=12, ge=0, le=25)
    inject_observation: Optional[dict] = None


class PatchRequest(BaseModel):
    source_step: int
    target_step: int
    patch_layer: int


class PatchSweepRequest(BaseModel):
    layers: Optional[list[int]] = None


class QueryRequest(BaseModel):
    query: str


class AttributionRequest(BaseModel):
    step_n: int
    layer: int = 12
    top_k: int = 12
    real: Optional[bool] = True


# --- Helpers ---

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


async def _get_run(run_id: str) -> dict:
    doc = await db.get_run(run_id)
    if not doc:
        doc = await db.get_experiment(run_id)
    if not doc:
        raise HTTPException(status_code=404, detail="run not found")
    return doc


def _summary_for_llm(run: dict, max_steps: int = 8) -> dict:
    return {
        "task": run["task"],
        "model": run.get("model", "gemma-2-2b-it"),
        "n_steps": run["n_steps"],
        "sae_layer": run["sae_layer"],
        "capture_layers": [6, 12, 18, 24],
        "steps": [
            {
                "step_n": s["step_n"],
                "output": s["output"][:160],
                "tool_called": s.get("tool_called"),
                "top_features": s["top_features"][:5],
                "hallucination": s["hallucination"],
                "layer_l2_norms_sample": s["layer_l2_norms"],
            }
            for s in run.get("steps", [])[:max_steps]
        ],
        "top_drifting_features": [
            {"feature_id": f["feature_id"], "drift_score": f["drift_score"], "activations": f["activations"]}
            for f in run.get("feature_timelines", [])[:8]
        ],
        "patch_summary": run.get("patch_matrix_summary"),
    }


# --- Endpoints ---

@router.post("/runs")
async def create_run(payload: RunCreate):
    run_id = str(uuid.uuid4())
    model_name = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
    is_gpt2 = "gpt2" in model_name.lower()
    sae_layer = payload.sae_layer
    if is_gpt2:
        if sae_layer == 12:
            sae_layer = 7
        else:
            sae_layer = min(sae_layer, 11)

    pool = await db.get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Enqueue run entry
            progress_json = json.dumps({"stage": "queued", "completed_steps": 0})
            await conn.execute(
                """
                INSERT INTO runs (id, task, model_name, n_steps, sae_layer, status, progress)
                VALUES ($1, $2, $3, $4, $5, 'queued', $6)
                """,
                uuid.UUID(run_id), payload.task, model_name, payload.n_steps, sae_layer, progress_json
            )

            # 2. Transactional Outbox Pattern insertion
            event_id = uuid.uuid4()
            event_payload = {
                "run_id": run_id,
                "task": payload.task,
                "n_steps": payload.n_steps,
                "sae_layer": sae_layer,
                "inject_observation": payload.inject_observation or {}
            }
            await db.save_outbox_event(conn, event_id, "run_trajectory", event_payload)

    logger.info("Enqueued run trajectory event %s in transaction outbox.", run_id)
    return {"run_id": run_id, "status": "queued"}


@router.get("/runs")
async def list_runs():
    rows = await db.list_runs()
    return {"runs": rows}


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    return await _get_run(run_id)


@router.get("/runs/{run_id}/steps/{step_n}")
async def get_step(run_id: str, step_n: int):
    run = await _get_run(run_id)
    for s in run.get("steps", []):
        if s["step_n"] == step_n:
            return s
    raise HTTPException(status_code=404, detail="step not found")


@router.post("/runs/{run_id}/patch")
async def run_patch(run_id: str, payload: PatchRequest):
    async with db.advisory_lock(run_id):
        run = await _get_run(run_id)
        src = next((s for s in run.get("steps", []) if s["step_n"] == payload.source_step), None)
        tgt = next((s for s in run.get("steps", []) if s["step_n"] == payload.target_step), None)
        if not src or not tgt:
            raise HTTPException(status_code=404, detail="step missing")
            
        model_name = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
        is_gpt2 = "gpt2" in model_name.lower()
        patch_layer = payload.patch_layer
        if is_gpt2:
            if patch_layer == 12:
                patch_layer = 7
            else:
                patch_layer = min(patch_layer, 11)
                
        max_layer = 11 if is_gpt2 else 25
        if not (0 <= payload.patch_layer <= max_layer) and not (is_gpt2 and payload.patch_layer == 12):
            raise HTTPException(status_code=400, detail=f"patch_layer must be 0–{max_layer}")

        model = get_model()
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None, cross_step_patch,
            model, src["activation_path"], tgt["prompt"], patch_layer,
        )
        
        await db.save_patch(
            run_id=run_id,
            source_step=payload.source_step,
            target_step=payload.target_step,
            patch_layer=payload.patch_layer,
            kl=res["kl"],
            significant=res["significant"],
            top_token_change=res.get("top_token_change")
        )
        return {
            "run_id": run_id,
            "source_step": payload.source_step,
            "target_step": payload.target_step,
            "patch_layer": payload.patch_layer,
            **res
        }


@router.post("/runs/{run_id}/patch-matrix")
async def run_patch_matrix(run_id: str, payload: PatchSweepRequest):
    async with db.advisory_lock(run_id):
        run = await _get_run(run_id)
        steps = run.get("steps", [])
        if not steps:
            raise HTTPException(status_code=400, detail="run not done")

        def target_prompt_fn(step_n: int) -> str:
            for s in steps:
                if s["step_n"] == step_n:
                    return s["prompt"]
            return ""

        model_name = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
        is_gpt2 = "gpt2" in model_name.lower()
        
        layers = payload.layers or ([3, 7, 10] if is_gpt2 else [6, 12, 18])
        if is_gpt2 and payload.layers == [6, 12, 18]:
            layers = [3, 7, 10]

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, patch_matrix, steps, target_prompt_fn, layers)
        summary = {
            "layers": layers,
            "n_results": len(results),
            "max_kl": max((r["kl"] for r in results), default=0.0),
            "significant_count": sum(1 for r in results if r["significant"]),
        }
        
        for r in results:
            await db.save_patch(
                run_id=run_id,
                source_step=r["source_step"],
                target_step=r["target_step"],
                patch_layer=r["patch_layer"],
                kl=r["kl"],
                significant=r["significant"],
                top_token_change=r.get("top_token_change")
            )
            
        await db.update_run(
            run_id,
            {
                "patch_matrix": results,
                "patch_matrix_summary": summary
            }
        )
        return {"patch_matrix": results, "layers": layers}


@router.get("/runs/{run_id}/patches")
async def list_patches(run_id: str):
    pool = await db.get_pool()
    rows = await pool.fetch("SELECT * FROM patch_results WHERE run_id = $1 ORDER BY id DESC", uuid.UUID(run_id))
    patches = [{
        "id": str(r["id"]),
        "run_id": str(r["run_id"]),
        "source_step": r["source_step"],
        "target_step": r["target_step"],
        "patch_layer": r["patch_layer"],
        "kl": r["kl"],
        "significant": r["significant"],
        "top_token_change": json.loads(r["top_token_change"]) if r["top_token_change"] else None
    } for r in rows]
    return {"patches": patches}


@router.post("/runs/{run_id}/attribution")
async def run_attribution(run_id: str, payload: AttributionRequest):
    async with db.advisory_lock(run_id):
        run = await _get_run(run_id)
        step = next((s for s in run.get("steps", []) if s["step_n"] == payload.step_n), None)
        if not step:
            raise HTTPException(status_code=404, detail="step not found")

        model_name = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
        is_gpt2 = "gpt2" in model_name.lower()
        layer = payload.layer
        if is_gpt2:
            if layer == 12:
                layer = 7
            else:
                layer = min(layer, 11)

        model = None
        real_run = payload.real
        if real_run:
            try:
                model = get_model()
            except Exception as e:
                logger.error("Failed to load model for real causal attribution, falling back to mock: %s", e)
                real_run = False

        loop = asyncio.get_running_loop()
        top_feats = step.get("top_features", [])[:payload.top_k]

        graph = await loop.run_in_executor(
            None,
            lambda: causal_attribution_for_step(model, step["prompt"], layer, top_feats, real_run)
        )

        graph_id = uuid.uuid4()
        pool = await db.get_pool()
        await pool.execute(
            """
            INSERT INTO attribution_graphs (id, run_id, step_n, layer, graph)
            VALUES ($1, $2, $3, $4, $5)
            """,
            graph_id, uuid.UUID(run_id), payload.step_n, payload.layer, json.dumps(graph)
        )

        return {
            "id": str(graph_id),
            "run_id": run_id,
            "step_n": payload.step_n,
            "layer": payload.layer,
            "graph": graph,
            "created_at": _now(),
        }


@router.post("/runs/{run_id}/query")
async def run_query(run_id: str, payload: QueryRequest):
    run = await _get_run(run_id)
    ctx = _summary_for_llm(run)
    answer = await ns_llm.ask(payload.query, ctx, session_id=f"run-{run_id}")
    
    query_id = str(uuid.uuid4())
    await db.save_query(query_id, run_id, payload.query, answer)
    
    return {
        "id": query_id,
        "run_id": run_id,
        "query": payload.query,
        "answer": answer,
        "created_at": _now(),
    }


@router.get("/runs/{run_id}/queries")
async def list_queries(run_id: str):
    rows = await db.list_queries(run_id)
    return {"queries": rows}


@router.get("/feature/{layer}/{feature_id}")
async def get_feature(layer: int, feature_id: int):
    label_doc = await db.get_feature_label(layer, feature_id)
    if label_doc:
        return label_doc
    return {
        "layer": layer,
        "feature_id": feature_id,
        "label": f"gemma_l{layer}_f{feature_id}",
        "neuronpedia_url": f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_id}",
    }
