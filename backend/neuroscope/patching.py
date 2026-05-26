"""Cross-step causal patching with correct KL divergence.

FIX #3 — KL direction was backwards in v1:
  v1: F.kl_div(patched.log(), baseline) = KL(baseline || patched)  ← WRONG
  v2: F.kl_div(baseline.log(), patched) = KL(patched || baseline)  ← CORRECT
  Correct interpretation: how much does patching source into target change
  the output distribution? KL(patched || baseline) measures divergence FROM
  the baseline TOWARDS the patched distribution.

FIX #6 — Patch length mismatch:
  v1 patched src[:, :min_len, :] into target — misaligns token semantics when
  the source prompt is shorter than the target (step N+1 includes N's history).
  v2 patches only the LAST TOKEN position: src[:, -1:, :].
  The last token represents the model's "current reasoning state" and is
  semantically comparable across steps regardless of prompt length.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .storage import load_step_activations


def cross_step_patch(
    model,
    source_activation_path: str,
    target_prompt: str,
    patch_layer: int,
) -> dict:
    """Patch source step's last-token residual into target step's forward pass.

    Patches ONLY the last token position of the source into the corresponding
    last-token position in the target. This avoids the token-length mismatch
    problem: regardless of prompt length, the last token position represents
    the model's 'current state' at that reasoning step.

    KL divergence: KL(patched_dist || baseline_dist)
    Interpretation: how much does injecting source state redirect target's next token?
    """
    src = load_step_activations(source_activation_path)
    hook_name = f"blocks.{patch_layer}.hook_resid_post"

    if hook_name not in src:
        available = [k for k in src if "resid" in k]
        return {
            "patch_layer": patch_layer,
            "kl": 0.0,
            "significant": False,
            "token_changes": [],
            "error": (
                f"Layer {patch_layer} not in source activations. "
                f"Captured layers: {available}"
            ),
        }

    # Extract ONLY the last-token residual from source
    # src[hook_name]: float16 ndarray [1, seq_len, d_model] -> [1, 1, d_model]
    src_last_token = torch.tensor(
        src[hook_name].astype(np.float32)
    )[:, -1:, :]  # shape [1, 1, d_model]

    tgt_tokens = model.to_tokens(target_prompt)

    # Baseline: unpatched forward pass through target prompt
    with torch.no_grad():
        baseline_logits = model(tgt_tokens, return_type="logits")
    baseline_probs = F.softmax(baseline_logits[0, -1], dim=-1)

    # Patched: inject source's last-token residual into target's last-token position
    def patch_last_token(value, hook):
        # value: [1, seq_len, d_model] — replace ONLY the final token position
        value[:, -1:, :] = src_last_token
        return value

    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, patch_last_token)]):
            patched_logits = model(tgt_tokens, return_type="logits")
    patched_probs = F.softmax(patched_logits[0, -1], dim=-1)

    # KL(patched || baseline) — correct direction (FIX #3)
    # F.kl_div(log_reference, target) = KL(target || reference)
    # So: F.kl_div(baseline.log(), patched) = KL(patched || baseline)
    kl = float(
        F.kl_div(
            (baseline_probs + 1e-10).log(),  # log of reference (baseline)
            patched_probs,                    # target distribution (patched)
            reduction="sum",
        ).item()
    )

    # Top-5 token probability shifts
    delta = (patched_probs - baseline_probs).abs()
    top_idx = delta.topk(5).indices.tolist()
    token_changes = []
    for idx in top_idx:
        tok = model.tokenizer.decode([idx])
        token_changes.append({
            "token": tok,
            "token_id": int(idx),
            "baseline_p": round(float(baseline_probs[idx].item()), 4),
            "patched_p": round(float(patched_probs[idx].item()), 4),
            "delta": round(float((patched_probs[idx] - baseline_probs[idx]).item()), 4),
        })

    significant = kl > 0.05

    return {
        "patch_layer": patch_layer,
        "kl": round(kl, 4),
        "significant": significant,
        "token_changes": token_changes,
        "interpretation": (
            f"Patching source step's last-token residual (layer {patch_layer}) "
            f"into the target step "
            f"{'significantly changes' if significant else 'does not significantly change'} "
            f"the output distribution (KL(patched||baseline)={kl:.4f}). "
            f"{'Causal influence from source step confirmed at this layer.' if significant else 'No strong causal influence at this layer.'}"
        ),
    }
