"""End-to-end trajectory orchestration: agent loop → hook capture → SAE → hallucination.
Used both by the API background task and the seed_experiments script.

Updated for v2:
  - Gemma-2-2b-it (26 layers, d_model=2304)
  - GemmaScope SAEs at layer 12 by default
  - resid_l2_per_captured_layer (4 values) instead of resid_l2_per_layer (12/26 values)
  - sae_coactivation_graph (renamed from sae_attribution_graph)
  - Hallucination scorer receives top_drifting_feature_ids dynamically (no hardcoded IDs)
  - Patch matrix defaults to layers [6, 12, 18] (Gemma mid-model)
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from .agent import build_prompt, greedy_decode, tool_for_output
from .hallucination import hallucination_score
from .hooks import capture_forward, default_hook_names, resid_l2_per_captured_layer
from .loader import CAPTURE_LAYERS, get_model
from .sae import (
    build_feature_timelines,
    compute_drift_scores,
    decompose_step,
    sae_coactivation_graph,
)
from .storage import save_step_activations

logger = logging.getLogger(__name__)


def run_trajectory(
    run_id: str,
    task: str,
    n_steps: int = 3,
    sae_layer: int = 12,   # Layer 12 of 26 for Gemma-2-2b-it
    inject_context_at_step: dict | None = None,
    progress_cb: Callable[[str, dict], None] | None = None,
) -> dict:
    """Run a multi-step ReAct-style agent trajectory using Gemma-2-2b-it.

    inject_context_at_step: {step_n: text} — injects false context to trigger
    hallucination in controlled experiments (e.g. hallucination-propagation).
    """
    t_start = time.time()
    model = get_model()
    n_layers = model.cfg.n_layers  # 26 for Gemma-2-2b-it

    if progress_cb:
        progress_cb("model_ready", {"n_layers": n_layers})

    hook_names = default_hook_names(n_layers)
    history: list[str] = []
    steps: list[dict] = []
    per_step_top: list[list[dict]] = []
    top_drifting_ids: list[int] = []  # updated after each step for hallucination scorer

    for n in range(1, n_steps + 1):
        # Optional adversarial context injection (for hallucination-propagation experiment)
        if inject_context_at_step and n in inject_context_at_step:
            history.append(f"Observation: {inject_context_at_step[n]}")

        prompt = build_prompt(task, n, history)
        t0 = time.time()

        captured, last_logits, n_tokens = capture_forward(model, prompt, hook_names)
        output = greedy_decode(model, prompt, max_new=40)
        history.append(f"Step {n}:\nThought:{output}")

        activation_path = save_step_activations(run_id, n, captured)
        sae_top = decompose_step(captured, layer=sae_layer, top_k=25)
        per_step_top.append(sae_top["top"])

        # Hallucination score — pass current top-drifting features for drift proxy
        # top_drifting_ids is empty on step 1 (harmless — drift_proxy falls back to entropy*0.5)
        attn_key = f"blocks.{n_layers - 1}.attn.hook_pattern"
        h = hallucination_score(
            last_logits=last_logits,
            last_attn_pattern=captured.get(attn_key),
            top_features=sae_top["top"],
            top_drifting_feature_ids=top_drifting_ids,
        )

        # Update drift IDs for next step's hallucination scorer (after ≥2 steps)
        if len(per_step_top) >= 2:
            timelines = build_feature_timelines(per_step_top)
            drift = compute_drift_scores(timelines)
            top_drifting_ids = sorted(drift, key=drift.get, reverse=True)[:8]

        layer_l2 = resid_l2_per_captured_layer(captured, CAPTURE_LAYERS)

        step_data = {
            "step_n": n,
            "prompt_tokens": n_tokens,
            "prompt": prompt,
            "output": output,
            "tool_called": tool_for_output(output),
            "activation_path": activation_path,
            "top_features": sae_top["top"],
            "n_active_features": sae_top["n_active"],
            "sae_l2_norm": sae_top["l2_norm"],
            "layer_l2_norms": layer_l2,          # 4 values (CAPTURE_LAYERS), not 12/26
            "capture_layers": CAPTURE_LAYERS,     # [6, 12, 18, 24]
            "hallucination": h,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
        steps.append(step_data)

        if progress_cb:
            progress_cb("step_done", {"step_n": n, "elapsed_ms": step_data["elapsed_ms"]})

    # Build feature timelines and drift scores for the full trajectory
    timelines = build_feature_timelines(per_step_top)
    drift = compute_drift_scores(timelines)
    top_drifting = sorted(drift, key=drift.get, reverse=True)[:25]

    feature_timelines_payload = [
        {
            "feature_id": int(fid),
            "layer": sae_layer,
            "activations": [round(float(v), 4) for v in timelines[fid]],
            "drift_score": round(float(drift[fid]), 4),
        }
        for fid in top_drifting
    ]

    return {
        "run_id": run_id,
        "task": task,
        "n_steps": n_steps,
        "sae_layer": sae_layer,
        "model": "gemma-2-2b-it",
        "capture_layers": CAPTURE_LAYERS,
        "steps": steps,
        "feature_timelines": feature_timelines_payload,
        "total_elapsed_ms": int((time.time() - t_start) * 1000),
    }


def patch_matrix(
    steps: list[dict],
    target_prompt_fn: Callable[[int], str],
    layers: list[int] | None = None,
) -> list[dict]:
    """Sweep (source, target, layer) triples — returns flat list of patch results."""
    from .patching import cross_step_patch
    model = get_model()
    # For Gemma-2-2b-it (26 layers), use mid-model and late-model captured layers
    if layers is None:
        layers = [6, 12, 18]
    out: list[dict] = []
    for s in steps:
        for t in steps:
            if s["step_n"] == t["step_n"]:
                continue
            for L in layers:
                tgt_prompt = target_prompt_fn(t["step_n"])
                res = cross_step_patch(
                    model=model,
                    source_activation_path=s["activation_path"],
                    target_prompt=tgt_prompt,
                    patch_layer=L,
                )
                out.append({
                    "source_step": s["step_n"],
                    "target_step": t["step_n"],
                    "patch_layer": L,
                    "kl": res["kl"],
                    "significant": res["significant"],
                    "top_token_change": res["token_changes"][0] if res["token_changes"] else None,
                })
    return out


def attribution_for_step(activation_path: str, layer: int, top_k: int = 12) -> dict:
    """Compute SAE co-activation graph for a single step (renamed from attribution)."""
    from .storage import load_step_activations
    acts = load_step_activations(activation_path)
    return sae_coactivation_graph(acts, layer=layer, top_k=top_k)
