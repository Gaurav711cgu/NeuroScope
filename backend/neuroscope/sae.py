"""SAE decomposition + feature drift computation for GemmaScope.

FIX #7 — Attribution graph renamed to sae_coactivation_graph:
  The v1 function 'sae_attribution_graph' computed Pearson correlation of feature
  activations — NOT causal attribution. Renaming to accurately describe what it
  computes. The Anthropic circuit-tracer integration is Phase 3 work (optional).
  All callers (runner.py, server.py) updated accordingly.

GemmaScope notes:
  - 16,384 features per layer (vs 24,576 in gpt2-small-res-jb)
  - JumpReLU activation: sparse by design, typically 50-200 features active
  - Trained on hook_resid_POST (hooks.py matches this convention)
  - sae.encode() handles JumpReLU correctly via SAELens abstraction
"""
from __future__ import annotations

import numpy as np
import torch

from .loader import get_sae


def decompose_step(
    activations: dict[str, np.ndarray],
    layer: int,
    top_k: int = 25,
) -> dict:
    """SAE-decompose the last-token residual at the given layer.

    Returns top-k GemmaScope features with their activation values.
    GemmaScope: 16,384 features per layer, JumpReLU activation.
    """
    sae, _cfg = get_sae(layer=layer)
    key = f"blocks.{layer}.hook_resid_post"

    if key not in activations:
        available = [k for k in activations if "resid" in k]
        return {
            "layer": layer,
            "top": [],
            "n_active": 0,
            "l2_norm": 0.0,
            "error": (
                f"Layer {layer} not captured. "
                f"Captured layers: {available}"
            ),
        }

    resid = torch.tensor(activations[key].astype(np.float32))  # [1, seq_len, d_model]

    with torch.no_grad():
        feat = sae.encode(resid)  # [1, seq_len, d_sae=16384]

    last = feat[0, -1]  # [16384] — last token position
    n_active = int((last > 0).sum().item())

    top = last.topk(min(top_k, n_active or top_k))

    items = [
        {"feature_id": int(i), "activation": round(float(v), 4)}
        for i, v in zip(top.indices.tolist(), top.values.tolist())
        if v > 0  # GemmaScope JumpReLU: only include active (positive) features
    ]

    return {
        "layer": layer,
        "top": items,
        "n_active": n_active,
        "l2_norm": round(float(torch.linalg.norm(last).item()), 4),
    }


def build_feature_timelines(
    per_step_top: list[list[dict]],
) -> dict[int, list[float]]:
    """Construct {feature_id: [activation_at_step_1, ..., activation_at_step_n]}."""
    n = len(per_step_top)
    timelines: dict[int, list[float]] = {}
    for step_idx, top in enumerate(per_step_top):
        for f in top:
            fid = int(f["feature_id"])
            if fid not in timelines:
                timelines[fid] = [0.0] * n
            timelines[fid][step_idx] = float(f["activation"])
    return timelines


def compute_drift_scores(timelines: dict[int, list[float]]) -> dict[int, float]:
    """Variance-based drift score across trajectory steps.

    Limitation: variance with 3-6 steps has high variance itself. This is a
    starting point; with enough runs, a normalized drift score against a
    population baseline is the correct next step.
    """
    return {
        fid: float(np.var(np.asarray(t, dtype=np.float32)))
        for fid, t in timelines.items()
    }


def sae_coactivation_graph(
    activations: dict[str, np.ndarray],
    layer: int,
    top_k: int = 12,
) -> dict:
    """Compute GemmaScope feature co-activation graph via Pearson correlation.

    NAMING NOTE: This is NOT causal attribution. It computes pairwise correlation
    of top-k SAE features across the last 8 token positions. High correlation =
    features tend to activate together = possible circuit membership (exploratory).
    True causal attribution requires the Anthropic circuit-tracer (Phase 3).

    Use this for: exploratory feature relationship visualization.
    Do NOT claim this as: 'the circuit responsible for behavior X'.
    """
    sae, _ = get_sae(layer=layer)
    key = f"blocks.{layer}.hook_resid_post"

    if key not in activations:
        return {
            "nodes": [],
            "edges": [],
            "layer": layer,
            "method": "coactivation_pearson",
        }

    resid = torch.tensor(activations[key].astype(np.float32))

    with torch.no_grad():
        feat = sae.encode(resid)[0]  # [seq_len, 16384]

    last = feat[-1]  # [16384]
    active_mask = last > 0
    n_active = int(active_mask.sum().item())

    if n_active < 2:
        return {
            "nodes": [],
            "edges": [],
            "layer": layer,
            "method": "coactivation_pearson",
        }

    top_idx = last.topk(min(top_k, n_active)).indices.tolist()

    # Co-activation over last 8 token positions
    pos = min(8, feat.shape[0])
    window = feat[-pos:][:, top_idx].cpu().numpy()  # [pos, top_k]

    # Pearson correlation
    win = window - window.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(win, axis=0) + 1e-8
    corr = (win.T @ win) / (denom[:, None] * denom[None, :])

    nodes = [
        {
            "id": int(fid),
            "layer": layer,
            "activation": round(float(last[fid].item()), 4),
            "size": round(float(last[fid].item()), 4),
        }
        for fid in top_idx
    ]

    edges = []
    for i in range(len(top_idx)):
        for j in range(i + 1, len(top_idx)):
            w = float(corr[i, j])
            if abs(w) > 0.25:
                edges.append({
                    "source": int(top_idx[i]),
                    "target": int(top_idx[j]),
                    "weight": round(w, 3),
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "layer": layer,
        "method": "coactivation_pearson",
        "disclaimer": (
            "Co-activation correlation (Pearson), not causal attribution. "
            "Use Anthropic circuit-tracer for causal graphs (Phase 3)."
        ),
    }
