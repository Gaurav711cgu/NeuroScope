"""Background worker polling and executing transactional outbox events with Hugging Face ZeroGPU support."""
from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
import os

from .db import claim_next_outbox_event, update_outbox_status, save_step
from .runner import run_trajectory, patch_matrix
from .db import update_run

logger = logging.getLogger("neuroscope.worker")

# Optional ZeroGPU decorator support
try:
    import spaces  # type: ignore
    _HAS_ZEROGPU = True
except ImportError:
    _HAS_ZEROGPU = False


if _HAS_ZEROGPU:
    @spaces.GPU(duration=120)
    def _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject):
        return run_trajectory(
            run_id=run_id, task=task, n_steps=n_steps,
            sae_layer=sae_layer, inject_observation=inject,
        )
else:
    def _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject):
        return run_trajectory(
            run_id=run_id, task=task, n_steps=n_steps,
            sae_layer=sae_layer, inject_context_at_step=inject,
        )


def grade_suggested_task(task: str, final_output: str) -> bool:
    """Evaluate reasoning correctness for suggested tasks using case-insensitive heuristics."""
    task_lower = task.lower()
    out_lower = final_output.lower()
    if "eiffel tower" in task_lower:
        return "paris" in out_lower and "france" in out_lower
    if "leaves at 14:30" in task_lower or "train" in task_lower:
        return "17:15" in out_lower
    if "weather in paris" in task_lower or "paris: search" in task_lower:
        return "search" in out_lower
    if "23 * 17" in task_lower:
        return "391" in out_lower
    if "spouse" in task_lower:
        return "macron" in out_lower or "brigitte" in out_lower
    if "bank" in task_lower:
        return "riverbank" in out_lower or "financial" in out_lower
    return "hallucinat" not in out_lower


_worker_task: asyncio.Task | None = None
_running = False


async def execute_outbox_event(event: dict):
    """Execute a claimed outbox task using strictly typed backend methods."""
    event_id = event["id"]
    event_type = event["event_type"]
    payload = event["payload"]
    
    logger.info("Executing outbox event %s of type %s", event_id, event_type)
    
    try:
        if event_type == "run_trajectory":
            run_id = payload["run_id"]
            task = payload["task"]
            n_steps = payload["n_steps"]
            sae_layer = payload["sae_layer"]
            inject_obs = payload.get("inject_observation")
            
            await update_run(run_id, {"status": "running", "progress": {"stage": "loading_model", "completed_steps": 0}})
            
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: _run_trajectory_gpu(run_id, task, n_steps, sae_layer, inject_obs or None)
            )
            
            correct = grade_suggested_task(task, result["steps"][-1]["output"])
            
            await update_run(run_id, {
                "status": "done",
                "correct": correct,
                "total_elapsed_ms": result["total_elapsed_ms"],
                "progress": {"stage": "done", "completed_steps": n_steps},
                "feature_timelines": result["feature_timelines"]
            })
            
            for s in result["steps"]:
                step_uuid = str(uuid.uuid4())
                await save_step(
                    step_id=step_uuid,
                    run_id=run_id,
                    step_n=s["step_n"],
                    prompt=s["prompt"],
                    output=s["output"],
                    tool_called=s.get("tool_called", "none"),
                    n_active_features=s.get("n_active_features", 0),
                    sae_l2_norm=s.get("sae_l2_norm", 0.0),
                    hallucination=s.get("hallucination", {}),
                    elapsed_ms=s.get("elapsed_ms", 0),
                    activation_path=s.get("activation_path", ""),
                    top_features=s.get("top_features", []),
                    layer_l2_norms=s.get("layer_l2_norms", [])
                )
            logger.info("Run %s done in %dms", run_id, result["total_elapsed_ms"])
            
        elif event_type == "patch_matrix":
            run_id = payload["run_id"]
            layers = payload.get("layers")
            await patch_matrix(run_id=run_id, layers=layers)
            
        else:
            raise ValueError(f"Unknown outbox event type: {event_type}")
            
        await update_outbox_status(event_id, "completed")
        logger.info("Outbox event %s completed successfully.", event_id)
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error("Outbox event %s failed: %s", event_id, error_msg)
        await update_outbox_status(event_id, "failed", error=error_msg)
        if "run_id" in payload:
            await update_run(payload["run_id"], {"status": "failed", "error": str(e)})


async def worker_loop():
    """Continuous polling worker loop utilizing database transaction limits."""
    global _running
    logger.info("Outbox worker loop started.")
    _running = True
    while _running:
        try:
            event = await claim_next_outbox_event()
            if event:
                await execute_outbox_event(event)
            else:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Error in outbox worker loop: %s", e)
            await asyncio.sleep(2.0)
            
    logger.info("Outbox worker loop stopped.")


def start_worker():
    """Start the background outbox polling loop task."""
    global _worker_task, _running
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(worker_loop())
        logger.info("Background outbox worker task launched.")


def stop_worker():
    """Gracefully cancel and stop the outbox worker loop task."""
    global _worker_task, _running
    _running = False
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        logger.info("Background outbox worker task cancelled.")
