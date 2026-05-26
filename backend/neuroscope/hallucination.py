"""Three-signal hallucination early-warning scorer for Gemma-2-2b-it + GemmaScope.
Framed as a diagnostic, not a correction mechanism.

FIX #5 — Hardcoded 'uncertainty feature' IDs removed:
  v1 used FALLBACK_UNCERT_FEATURES = [12059, 4521, 7291, 1842], which were
  GPT-2 specific IDs with no scientific basis for Gemma-2-2b-it.
  Replacement: dynamically identify uncertainty-signal features from the
  current run's own top-drifting features. High drift = high variance across
  reasoning steps = proxy for instability/uncertainty.
  This approach is self-referential (no fixed lookup table needed) and
  scientifically defensible: it only claims 'these features are unstable
  in this run', not 'these are universal hallucination features'.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hallucination_score(
    last_logits: np.ndarray,
    last_attn_pattern: np.ndarray | None,
    top_features: list[dict],
    top_drifting_feature_ids: list[int] | None = None,
) -> dict:
    """Compute three-signal hallucination risk for one agent step.

    Signals:
    1. Entropy (0..1): softmax entropy of next-token distribution.
       High entropy = model is uncertain about the next token.

    2. Attention diffusion (0..1): entropy of attention weights across keys.
       High diffusion = model attends broadly, not to specific context.
       (Wentao Shi et al., 2023 — correlated with hallucination in factual QA)

    3. Feature drift proxy (0..1): mean activation of the top-drifting features
       at this step. High drift features that are also highly active at this step
       indicate the model's representation is shifting — a proxy for instability.

       NOTE: This replaces the hardcoded list from v1.
       top_drifting_feature_ids comes from drift computation in runner.py,
       which uses THIS run's own feature timelines — no fixed lookup needed.

    Composite: 0.4 * entropy + 0.3 * attn_diffusion + 0.3 * drift_proxy
    Threshold 0.65 → flag for review (conservative — reduces false positives)

    IMPORTANT: Scores > 0.65 are not binary hallucination labels. They are
    risk indicators to flag trajectories for human review. Intervention via SAE
    steering has approximately 20-30% correction success rate (literature, Mar 2026).
    """
    # ── Signal 1: Next-token distribution entropy ─────────────────────────────
    logits = torch.tensor(last_logits.astype(np.float32))
    probs = F.softmax(logits, dim=-1)
    entropy = float(-(probs * (probs + 1e-10).log()).sum().item())
    # Gemma-2-2b vocab ~256k → log(256k) ≈ 12.5; normalize to [0, 1] with headroom at 10
    entropy_score = float(min(entropy / 10.0, 1.0))

    # ── Signal 2: Attention weight diffusion (last-layer attention) ───────────
    if last_attn_pattern is not None:
        a = torch.tensor(last_attn_pattern.astype(np.float32))
        if a.ndim == 4:
            a = a[0]  # [heads, q, k]
        # Entropy across key dimension, averaged over heads and query positions
        a_safe = a + 1e-10
        a_entropy = float(-(a_safe * a_safe.log()).sum(-1).mean().item())
        # Normalize: max entropy for uniform over k keys is log(k)
        # k is typically 512+ for long prompts; use 6.0 as normalization (log(~400))
        attn_score = float(min(a_entropy / 6.0, 1.0))
    else:
        attn_score = 0.5  # fallback if attention not captured

    # ── Signal 3: Feature drift proxy (dynamic, from THIS run's features) ─────
    if top_drifting_feature_ids and len(top_drifting_feature_ids) > 0:
        by_id = {int(f["feature_id"]): float(f["activation"]) for f in top_features}
        drift_activations = [by_id.get(int(fid), 0.0) for fid in top_drifting_feature_ids[:8]]
        mean_drift_act = float(np.mean(drift_activations))
        # GemmaScope activations are JumpReLU-gated, typically 0..20 range
        drift_score = float(min(mean_drift_act / 8.0, 1.0))
    else:
        # No drift features yet (step 1 of trajectory) — use entropy as proxy
        drift_score = entropy_score * 0.5

    composite = 0.4 * entropy_score + 0.3 * attn_score + 0.3 * drift_score

    return {
        "composite": round(composite, 3),
        "entropy": round(entropy_score, 3),
        "attention_diffusion": round(attn_score, 3),
        "drift_proxy": round(drift_score, 3),
        "flag": composite > 0.65,
        "note": (
            "Early-warning diagnostic only. Composite > 0.65 flags for review. "
            "SAE steering intervention: ~20-30% correction success (lit. Mar 2026). "
            "Drift proxy is computed from this run's own top-drifting GemmaScope features — "
            "no hardcoded feature IDs."
        ),
    }
