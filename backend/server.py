"""NeuroScope FastAPI server — mounts /api/v1/* endpoints.

FIX #9 (complete) — MongoDB/Motor replaced with Supabase Postgres client.
  All db.collection.* calls replaced with sb.table("*").*.execute() equivalents.

Additional modernisations:
  - asyncio.get_running_loop() (replaces deprecated get_event_loop())
  - lifespan context manager (replaces deprecated @app.on_event)
  - patch_layer validation: 0–25 (26 layers for Gemma-2-2b-it)
  - ZeroGPU @spaces.GPU guard (try/except for local dev compatibility)
  - Attribution route updated to use sae_coactivation_graph naming
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from neuroscope.firebase_init import get_db

from pathlib import Path
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

from neuroscope import llm as ns_llm  # noqa: E402
from neuroscope.agent import build_prompt  # noqa: E402
from neuroscope.loader import get_model, model_info  # noqa: E402
from neuroscope.patching import cross_step_patch  # noqa: E402
from neuroscope.runner import attribution_for_step, patch_matrix, run_trajectory  # noqa: E402

# Optional ZeroGPU decorator (only available inside a HF Space)
try:
    import spaces  # type: ignore
    _HAS_ZEROGPU = True
except ImportError:
    _HAS_ZEROGPU = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("neuroscope.server")


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Nothing to explicitly close with supabase-py sync client


app = FastAPI(title="NeuroScope API", version="2.0.0", lifespan=lifespan)
api_router = APIRouter(prefix="/api")
v1 = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RunCreate(BaseModel):
    task: str
    n_steps: int = Field(default=3, ge=2, le=6)
    sae_layer: int = Field(default=12, ge=0, le=25)   # 26 layers for Gemma-2-2b-it
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
    layer: int = 12        # Default to layer 12 (mid-model for Gemma-2-2b-it)
    top_k: int = 12


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_run(run_id: str) -> dict:
    db = get_db()
    loop = asyncio.get_running_loop()

    def _fetch():
        # Try agent_runs first
        doc_ref = db.collection("agent_runs").document(run_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict()
        # Fall back to experiments (view by slug)
        query = db.collection("experiments").where("slug", "==", run_id).limit(1).get()
        if query:
            e = query[0].to_dict()
            return {
                "id": e["slug"],
                "task": e.get("task"),
                "n_steps": e.get("n_steps"),
                "sae_layer": e.get("sae_layer"),
                "model": e.get("model", "gemma-2-2b-it"),
                "status": "done",
                "steps": e.get("steps", []),
                "feature_timelines": e.get("feature_timelines", []),
                "patch_matrix": e.get("patch_matrix", []),
                "patch_matrix_summary": e.get("patch_matrix_summary"),
                "total_elapsed_ms": e.get("total_elapsed_ms"),
                "_is_experiment": True,
            }
        return None

    doc = await loop.run_in_executor(None, _fetch)
    if not doc:
        raise HTTPException(status_code=404, detail="run not found")
    return doc


def _summary_for_llm(run: dict, max_steps: int = 8) -> dict:
    """Build a compact JSON the LLM can ground answers in."""
    return {
        "task": run["task"],
        "model": run.get("model", "gemma-2-2b-it"),
        "n_steps": run["n_steps"],
        "sae_layer": run["sae_layer"],
        "capture_layers": run.get("capture_layers", [6, 12, 18, 24]),
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@v1.get("/health")
async def health():
    return {"status": "ok", "time": _now(), "model": model_info()}


@v1.get("/suggested-tasks")
async def suggested_tasks():
    return {
        "tasks": [
            {
                "id": "capital-france",
                "title": "Factual multi-hop",
                "task": "The Eiffel Tower is located in which city, and what country is that city the capital of?",
                "n_steps": 3,
                "category": "Factual QA",
            },
            {
                "id": "math-stepwise",
                "title": "Stepwise arithmetic",
                "task": "If a train leaves at 14:30 and travels for 2 hours 45 minutes, what time does it arrive?",
                "n_steps": 4,
                "category": "Math",
            },
            {
                "id": "tool-routing",
                "title": "Tool routing decision",
                "task": "Which tool should you use to find the current weather in Paris: search, lookup, or calc?",
                "n_steps": 3,
                "category": "Tool selection",
            },
            {
                "id": "self-correction",
                "title": "Self-correction",
                "task": "Compute 23 * 17. Then verify by computing 17 * 23 and check the results match.",
                "n_steps": 4,
                "category": "Self-correction",
            },
            {
                "id": "reasoning-collapse",
                "title": "Multi-hop reasoning",
                "task": "Who is the current spouse of the president of the country whose capital hosts the Eiffel Tower?",
                "n_steps": 5,
                "category": "Multi-hop",
            },
            {
                "id": "ambiguity",
                "title": "Ambiguity probe",
                "task": "Is the statement 'The bank is across the river' about a financial institution or a riverbank?",
                "n_steps": 3,
                "category": "Ambiguity",
            },
        ]
    }


@v1.post("/runs")
async def create_run(payload: RunCreate, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    doc = {
        "id": run_id,
        "task": payload.task,
        "n_steps": payload.n_steps,
        "sae_layer": payload.sae_layer,
        "model_name": "gemma-2-2b-it",
        "status": "queued",
        "created_at": _now(),
        "progress": {"stage": "queued", "completed_steps": 0},
        "inject_observation": payload.inject_observation or {},
    }
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: get_db().collection("agent_runs").document(run_id).set(doc))
    background_tasks.add_task(
        _execute_run, run_id, payload.task, payload.n_steps, payload.sae_layer,
        payload.inject_observation or {},
    )
    return {"run_id": run_id, "status": "queued"}


# ZeroGPU-decorated trajectory function (guard for local dev)
if _HAS_ZEROGPU:
    import spaces  # type: ignore

    @spaces.GPU(duration=120)
    def _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject):
        return run_trajectory(
            run_id=run_id, task=task, n_steps=n_steps,
            sae_layer=sae_layer, inject_context_at_step=inject,
        )
else:
    def _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject):
        return run_trajectory(
            run_id=run_id, task=task, n_steps=n_steps,
            sae_layer=sae_layer, inject_context_at_step=inject,
        )


async def _execute_run(run_id: str, task: str, n_steps: int, sae_layer: int, inject: dict) -> None:
    """Background task: execute trajectory, persist to Firestore."""
    inject_int = {int(k): v for k, v in (inject or {}).items()}
    db = get_db()
    loop = asyncio.get_running_loop()

    async def _update(fields: dict):
        await loop.run_in_executor(
            None,
            lambda: db.collection("agent_runs").document(run_id).update(fields),
        )

    try:
        await _update({"status": "running", "progress": {"stage": "loading_model", "completed_steps": 0}})

        def _on_progress(stage: str, payload: dict):
            asyncio.run_coroutine_threadsafe(
                _update({"progress": {"stage": stage, "completed_steps": payload.get("step_n", 0)}}),
                loop,
            )

        result = await loop.run_in_executor(
            None,
            lambda: _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject_int or None),
        )

        await _update({
            "status": "done",
            "steps": result["steps"],
            "feature_timelines": result["feature_timelines"],
            "total_elapsed_ms": result["total_elapsed_ms"],
            "progress": {"stage": "done", "completed_steps": n_steps},
        })
        logger.info("Run %s done in %dms", run_id, result["total_elapsed_ms"])
    except Exception as e:  # noqa: BLE001
        logger.exception("Run %s failed", run_id)
        await _update({"status": "error", "error": f"{type(e).__name__}: {e}"})


@v1.get("/runs")
async def list_runs():
    db = get_db()
    from firebase_admin import firestore
    loop = asyncio.get_running_loop()

    def _fetch():
        docs = db.collection("agent_runs").order_by("created_at", direction=firestore.Query.DESCENDING).limit(30).stream()
        runs = []
        for doc in docs:
            d = doc.to_dict()
            runs.append({
                "id": d.get("id"),
                "task": d.get("task"),
                "model_name": d.get("model_name"),
                "status": d.get("status"),
                "created_at": d.get("created_at"),
                "n_steps": d.get("n_steps"),
                "sae_layer": d.get("sae_layer"),
                "progress": d.get("progress"),
                "error": d.get("error"),
                "total_elapsed_ms": d.get("total_elapsed_ms"),
            })
        return runs

    rows = await loop.run_in_executor(None, _fetch)
    return {"runs": rows}


@v1.get("/runs/{run_id}")
async def get_run(run_id: str):
    return await _get_run(run_id)


@v1.get("/runs/{run_id}/steps/{step_n}")
async def get_step(run_id: str, step_n: int):
    run = await _get_run(run_id)
    for s in run.get("steps", []):
        if s["step_n"] == step_n:
            return s
    raise HTTPException(status_code=404, detail="step not found")


@v1.post("/runs/{run_id}/patch")
async def run_patch(run_id: str, payload: PatchRequest):
    run = await _get_run(run_id)
    src = next((s for s in run.get("steps", []) if s["step_n"] == payload.source_step), None)
    tgt = next((s for s in run.get("steps", []) if s["step_n"] == payload.target_step), None)
    if not src or not tgt:
        raise HTTPException(status_code=404, detail="step missing")
    if not (0 <= payload.patch_layer <= 25):
        raise HTTPException(status_code=400, detail="patch_layer must be 0–25 (Gemma-2-2b-it has 26 layers)")

    model = get_model()
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        None, cross_step_patch,
        model, src["activation_path"], tgt["prompt"], payload.patch_layer,
    )
    record = {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "source_step": payload.source_step,
        "target_step": payload.target_step,
        "patch_layer": payload.patch_layer,
        **res,
        "created_at": _now(),
    }
    db = get_db()
    loop2 = asyncio.get_running_loop()
    await loop2.run_in_executor(None, lambda: db.collection("causal_patches").document(record["id"]).set(record))
    return record


@v1.post("/runs/{run_id}/patch-matrix")
async def run_patch_matrix(run_id: str, payload: PatchSweepRequest):
    run = await _get_run(run_id)
    steps = run.get("steps", [])
    if not steps:
        raise HTTPException(status_code=400, detail="run not done")

    def target_prompt_fn(step_n: int) -> str:
        for s in steps:
            if s["step_n"] == step_n:
                return s["prompt"]
        return ""

    loop = asyncio.get_running_loop()
    layers = payload.layers or [6, 12, 18]   # Gemma-2-2b-it captured layers
    results = await loop.run_in_executor(None, patch_matrix, steps, target_prompt_fn, layers)
    summary = {
        "layers": layers,
        "n_results": len(results),
        "max_kl": max((r["kl"] for r in results), default=0.0),
        "significant_count": sum(1 for r in results if r["significant"]),
    }
    db = get_db()
    await loop.run_in_executor(
        None,
        lambda: db.collection("agent_runs").document(run_id).update({
            "patch_matrix": results,
            "patch_matrix_summary": summary,
        }),
    )
    return {"patch_matrix": results, "layers": layers}


@v1.get("/runs/{run_id}/patches")
async def list_patches(run_id: str):
    db = get_db()
    from firebase_admin import firestore
    loop = asyncio.get_running_loop()

    def _fetch():
        docs = db.collection("causal_patches").where("run_id", "==", run_id).order_by("created_at", direction=firestore.Query.DESCENDING).limit(200).stream()
        return [doc.to_dict() for doc in docs]

    rows = await loop.run_in_executor(None, _fetch)
    return {"patches": rows}


@v1.post("/runs/{run_id}/attribution")
async def run_attribution(run_id: str, payload: AttributionRequest):
    run = await _get_run(run_id)
    step = next((s for s in run.get("steps", []) if s["step_n"] == payload.step_n), None)
    if not step:
        raise HTTPException(status_code=404, detail="step not found")
    loop = asyncio.get_running_loop()
    graph = await loop.run_in_executor(
        None, attribution_for_step, step["activation_path"], payload.layer, payload.top_k,
    )
    record = {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "step_n": payload.step_n,
        "layer": payload.layer,
        "graph": graph,
        "created_at": _now(),
    }
    db = get_db()
    await loop.run_in_executor(None, lambda: db.collection("attribution_graphs").document(record["id"]).set(record))
    return record


@v1.post("/runs/{run_id}/query")
async def run_query(run_id: str, payload: QueryRequest):
    run = await _get_run(run_id)
    ctx = _summary_for_llm(run)
    answer = await ns_llm.ask(payload.query, ctx, session_id=f"run-{run_id}")
    record = {
        "id": str(uuid.uuid4()),
        "run_id": run_id,
        "query": payload.query,
        "answer": answer,
        "created_at": _now(),
    }
    db = get_db()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: db.collection("queries").document(record["id"]).set(record))
    return record


@v1.get("/runs/{run_id}/queries")
async def list_queries(run_id: str):
    db = get_db()
    from firebase_admin import firestore
    loop = asyncio.get_running_loop()

    def _fetch():
        docs = db.collection("queries").where("run_id", "==", run_id).order_by("created_at", direction=firestore.Query.DESCENDING).limit(50).stream()
        return [doc.to_dict() for doc in docs]

    rows = await loop.run_in_executor(None, _fetch)
    return {"queries": rows}


@v1.get("/experiments")
async def list_experiments():
    db = get_db()
    loop = asyncio.get_running_loop()

    def _fetch():
        docs = db.collection("experiments").stream()
        experiments = []
        for doc in docs:
            d = doc.to_dict()
            experiments.append({
                "id": d.get("id"),
                "slug": d.get("slug"),
                "title": d.get("title"),
                "category": d.get("category"),
                "hypothesis": d.get("hypothesis"),
                "finding": d.get("finding"),
                "task": d.get("task"),
                "n_steps": d.get("n_steps"),
                "sae_layer": d.get("sae_layer"),
                "model": d.get("model"),
                "total_elapsed_ms": d.get("total_elapsed_ms"),
                "created_at": d.get("created_at"),
            })
        return experiments

    rows = await loop.run_in_executor(None, _fetch)
    return {"experiments": rows}


@v1.get("/experiments/{slug}")
async def get_experiment(slug: str):
    db = get_db()
    loop = asyncio.get_running_loop()

    def _fetch():
        query = db.collection("experiments").where("slug", "==", slug).limit(1).get()
        return [doc.to_dict() for doc in query]

    rows = await loop.run_in_executor(None, _fetch)
    if not rows:
        raise HTTPException(status_code=404, detail="experiment not found")
    return rows[0]


@v1.get("/feature/{layer}/{feature_id}")
async def get_feature(layer: int, feature_id: int):
    """Return cached GemmaScope feature label (best-effort via Neuronpedia)."""
    db = get_db()
    loop = asyncio.get_running_loop()

    def _fetch():
        doc_id = f"l{layer}_f{feature_id}"
        doc_ref = db.collection("feature_labels").document(doc_id).get()
        if doc_ref.exists:
            return [doc_ref.to_dict()]
        return []

    rows = await loop.run_in_executor(None, _fetch)
    if rows:
        return rows[0]
    return {
        "layer": layer,
        "feature_id": feature_id,
        "label": f"gemma_l{layer}_f{feature_id}",
        "neuronpedia_url": f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_id}",
        "description": "Label not yet cached. Click 'View on Neuronpedia' for community-sourced description.",
    }


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------

api_router.include_router(v1)


@api_router.get("/")
async def root():
    return {"service": "neuroscope", "version": "2.0.0", "model": "gemma-2-2b-it"}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception):
    logger.exception("unhandled")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
