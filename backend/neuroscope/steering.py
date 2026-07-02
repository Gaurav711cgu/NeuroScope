"""Activation steering capability: amplify or suppress specific SAE feature directions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from .loader import get_model, get_sae


def steer_and_regenerate(
    prompt: str,
    layer: int,
    feature_id: int,
    alpha: float,
    max_new_tokens: int = 40,
) -> dict:
    """Amplify a specific SAE feature via residual addition during generation.

    Steering vector: h_new = h + alpha * W_dec[feature_id]
    """
    model = get_model()
    sae, _ = get_sae(layer=layer)

    # Get decoders directions W_dec
    W_dec_feature = sae.W_dec[feature_id].to(
        device=next(model.parameters()).device,
        dtype=model.cfg.dtype,
    )  # shape [d_model]

    def steer_hook(value, hook):
        # value: [batch, seq_len, d_model]
        # Add steering vector at all positions (broadcast)
        return value + alpha * W_dec_feature.unsqueeze(0).unsqueeze(0)

    # Use the correct hook name
    hook_name = sae.cfg.hook_name if (hasattr(sae, "cfg") and hasattr(sae.cfg, "hook_name")) else f"blocks.{layer}.hook_resid_post"

    # Baseline (unsteered) completion
    with torch.no_grad():
        baseline_tokens = model.generate(
            model.to_tokens(prompt),
            max_new_tokens=max_new_tokens,
            verbose=False,
        )
    baseline_text = model.to_string(baseline_tokens[0])

    # Steered completion
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
            steered_tokens = model.generate(
                model.to_tokens(prompt),
                max_new_tokens=max_new_tokens,
                verbose=False,
            )
    steered_text = model.to_string(steered_tokens[0])

    return {
        "layer": layer,
        "feature_id": feature_id,
        "alpha": alpha,
        "baseline": baseline_text,
        "steered": steered_text,
        "hook": hook_name,
    }
