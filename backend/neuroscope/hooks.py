"""Forward-pass hook capture utilities for Gemma-2-2b-it.
Stores activations as float16 numpy arrays.

FIX #4 — resid_pre/post mismatch resolved:
  GemmaScope SAEs are trained on hook_resid_POST — the residual stream AFTER
  the block's contribution is added. We hook resid_POST throughout to match.
  v1 hooked resid_POST but used gpt2-small-res-jb which was trained on resid_PRE,
  creating a systematic mismatch. By switching to GemmaScope (trained on resid_POST)
  the hook and SAE now use the same tensor.

Storage optimisation:
  Gemma-2-2b-it has 26 layers. Full capture = ~61MB per step. We capture only
  CAPTURE_LAYERS = [6, 12, 18, 24] for residuals, reducing to ~10MB per step.
"""
from __future__ import annotations

import numpy as np
import torch

from .loader import CAPTURE_LAYERS


def default_hook_names(n_layers: int) -> list[str]:
    """Return hook names for Gemma-2-2b-it.

    We hook:
    - hook_resid_post at CAPTURE_LAYERS (6, 12, 18, 24) — matches GemmaScope SAE convention
    - hook_attn_pattern at the last layer only — for hallucination attention diffusion signal
    - hook_mlp_out at layers 6 and 12 — mid-model mechanistic diagnostics
    """
    names: list[str] = []
    # Residual stream at selected layers (matches GemmaScope SAE training convention)
    for i in CAPTURE_LAYERS:
        if i < n_layers:
            names.append(f"blocks.{i}.hook_resid_post")
    # Last-layer attention for hallucination diffusion signal
    names.append(f"blocks.{n_layers - 1}.attn.hook_pattern")
    # Mid-model MLP outputs for mechanistic diagnostics
    for i in [6, 12]:
        if i < n_layers:
            names.append(f"blocks.{i}.hook_mlp_out")
    return names


def capture_forward(
    model,
    prompt: str,
    hook_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Run a single forward pass, capture all listed hooks as float16 numpy.

    Returns:
        captured:     dict of hook_name -> float16 numpy array
        last_logits:  float32 logits at final token position (for hallucination scoring)
        n_tokens:     length of tokenized prompt
    """
    tokens = model.to_tokens(prompt)
    captured: dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def fn(value, hook):
            # Store as float16 — halves storage vs float32
            captured[name] = value.detach().to(torch.float16).cpu().numpy()
        return fn

    fwd_hooks = [(n, make_hook(n)) for n in hook_names]

    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(tokens, return_type="logits")

    # Keep last_logits as float32 for numerically stable softmax
    last_logits = logits[0, -1].detach().cpu().to(torch.float32).numpy()
    return captured, last_logits, int(tokens.shape[1])


def resid_l2_per_captured_layer(
    activations: dict[str, np.ndarray],
    capture_layers: list[int],
) -> list[float]:
    """Return L2 norm of last-token residual at each captured layer.

    Returns a list aligned to capture_layers (length 4 for [6, 12, 18, 24]),
    not all 26 Gemma layers.
    """
    out: list[float] = []
    for i in capture_layers:
        key = f"blocks.{i}.hook_resid_post"
        if key not in activations:
            out.append(0.0)
            continue
        arr = activations[key].astype(np.float32)
        last = arr[0, -1]  # [d_model]
        out.append(float(np.linalg.norm(last)))
    return out
