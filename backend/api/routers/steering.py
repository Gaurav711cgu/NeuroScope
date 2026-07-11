"""Steering router mounting endpoints for active residual vector steering, probe training, and safety shield comparisons."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from neuroscope import db
from neuroscope.steering import steer_and_regenerate
from neuroscope.shield_runner import run_shield_trajectory

logger = logging.getLogger("neuroscope.api.steering")

router = APIRouter()


# --- Schemas ---

class SteerRequest(BaseModel):
    prompt: str
    layer: int
    feature_id: int
    alpha: float = 10.0
    real: Optional[bool] = True


class ProbeTrainRequest(BaseModel):
    layer: int = 12
    real: Optional[bool] = False


class ShieldRequest(BaseModel):
    task: str
    rules: list[dict]
    real: Optional[bool] = False


# --- Endpoints ---

@router.post("/steer")
async def steer_feature_endpoint(payload: SteerRequest):
    """Run residual steering generation on a prompt, amplifying the decoder direction of feature_id."""
    baseline = None
    steered = None

    if payload.real:
        try:
            res = await steer_and_regenerate(payload.prompt, payload.layer, payload.feature_id, payload.alpha)
            baseline = res["baseline"]
            steered = res["steered"]
        except Exception as e:
            logger.error("Real steering failed, falling back to mock: %s", e)
            payload.real = False

    if not baseline or not steered:
        low_p = payload.prompt.lower()
        if "einstein" in low_p:
            baseline = "Thought: Albert Einstein won the Nobel Prize in Physics in 1921."
            if payload.feature_id % 3 == 0:
                steered = "Thought: Wait, let me double check. Albert Einstein won the Nobel Prize in Physics in 1921 (awarded in 1922) for his explanation of the photoelectric effect, not relativity. I must verify this fact."
            elif payload.feature_id % 3 == 1:
                steered = "Thought: Albert Einstein won the Nobel Prize in Physics in 1921. Let's compute: 1921 + 100 = 2021 was the centenary year of his prize."
            else:
                steered = "Thought: Albert Einstein (the famous physicist who developed the theory of relativity) won the Nobel Prize in Physics in 1921 for the photoelectric effect."
        else:
            baseline = "Thought: Solving the reasoning task step-by-step."
            steered = f"Thought: Solving the reasoning task step-by-step. [Steering active: Feature {payload.feature_id} amplified by {payload.alpha}x in Layer {payload.layer} residual stream]"

    return {
        "baseline": baseline,
        "steered": steered,
        "feature_id": payload.feature_id,
        "alpha": payload.alpha,
        "layer": payload.layer,
        "real": payload.real
    }


@router.post("/probe/train")
async def train_probe_endpoint(payload: ProbeTrainRequest):
    """Train a sparse linear probe and Cox hazard model to predict factual assertions and representation decay."""
    from neuroscope.probe import train_hallucination_probe
    if not payload.real:
        return await train_hallucination_probe([], layer=payload.layer)
        
    pool = await db.get_pool()
    rows = await pool.fetch("SELECT id FROM runs WHERE status = 'done' AND correct IS NOT NULL")
    run_ids = [str(r["id"]) for r in rows]
    
    result = await train_hallucination_probe(run_ids, layer=payload.layer)
    return result


@router.post("/shield/run")
async def run_shield_endpoint(payload: ShieldRequest):
    """Run a comparative agent trajectory with/without the active steering safety shield."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_shield_trajectory(payload.task, payload.rules, payload.real)
    )
    return result
