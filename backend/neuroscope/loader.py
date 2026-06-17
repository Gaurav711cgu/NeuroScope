"""Lazy singleton loaders for Gemma-2-2b-it HookedTransformer and GemmaScope SAEs.
Model and SAEs are loaded once per process; thread-safe via lock.

CRITICAL ARCHITECTURE DECISION (locked):
- Model:   google/gemma-2-2b-it via TransformerLens HookedTransformer
- SAE:     google/gemma-scope-2b-pt-res (residual stream SAEs, 16k features/layer)
- Same model used for both agent execution AND activation analysis.
  Using different models for these would be scientifically invalid.
- Claude/other LLMs are NEVER used here — only for NL query explanations (llm.py).

FIX #1: Replaces GPT-2 Small (MODEL_NAME=gpt2) with Gemma-2-2b-it.
FIX #2: Replaces gpt2-small-res-jb SAEs with GemmaScope (gemma-scope-2b-pt-res).
"""
from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

_model = None
_sae_cache: dict[tuple[str, int], object] = {}
_lock = threading.Lock()

MODEL_NAME = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
SAE_RELEASE = os.environ.get("NEUROSCOPE_SAE_RELEASE", "gemma-scope-2b-pt-res")
DEFAULT_SAE_LAYER = int(os.environ.get("NEUROSCOPE_SAE_LAYER", "12"))

# Gemma-2-2b-it: 26 transformer layers (0-indexed: 0..25).
# We capture residual stream at layers 6, 12, 18, 24 (4 of 26) to limit storage.
# Full capture: 26 * 512 * 2304 * 2 bytes ≈ 61MB per step. Too large for Supabase free tier.
# 4-layer capture: ~10MB per step — well within the 1GB bucket limit.
is_gemma = "gemma" in MODEL_NAME.lower()
CAPTURE_LAYERS = [6, 12, 18, 24] if is_gemma else [3, 6, 7, 9, 10]

torch.set_num_threads(int(os.environ.get("NEUROSCOPE_THREADS", "4")))


def get_model():
    """Load Gemma-2-2b-it as a HookedTransformer on CPU.

    On HF ZeroGPU Spaces the @spaces.GPU decorator on the caller moves it to GPU.
    fold_ln=True and center_* are required for SAE compatibility with GemmaScope.
    dtype=torch.float16 is required for the TransformerLens ↔ Gemma-2 bridge.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                quant = os.environ.get("NEUROSCOPE_QUANTIZATION", "").lower()
                load_in_8bit = quant == "8bit"
                load_in_4bit = quant == "4bit"

                logger.info("Loading HookedTransformer %s on %s ...", MODEL_NAME, device)
                from transformer_lens import HookedTransformer

                model_kwargs = {
                    "device": device,
                    "fold_ln": True,
                    "center_writing_weights": True,
                    "center_unembed": True,
                    # Required for Gemma-2 via TransformerLens bridge
                    "dtype": torch.bfloat16 if "gemma" in MODEL_NAME.lower() else None,
                }

                if device == "cuda" and (load_in_8bit or load_in_4bit):
                    try:
                        import bitsandbytes  # noqa: F401
                        if load_in_8bit:
                            model_kwargs["load_in_8bit"] = True
                            logger.info("Enabling 8-bit quantization via bitsandbytes")
                        elif load_in_4bit:
                            model_kwargs["load_in_4bit"] = True
                            logger.info("Enabling 4-bit quantization via bitsandbytes")
                    except ImportError:
                        logger.warning("bitsandbytes import failed, falling back to unquantized loading")

                _model = HookedTransformer.from_pretrained(
                    MODEL_NAME,
                    **model_kwargs
                )
                _model.eval()
                logger.info(
                    "Loaded %s: n_layers=%d d_model=%d n_heads=%d",
                    MODEL_NAME,
                    _model.cfg.n_layers,
                    _model.cfg.d_model,
                    _model.cfg.n_heads,
                )
    return _model


def get_sae(layer: int = DEFAULT_SAE_LAYER):
    """Load GemmaScope residual SAE for a given layer.

    GemmaScope SAEs are trained on hook_resid_POST — matches our hook convention.
    GemmaScope: 16,384 features per layer (16k width, canonical).

    CRITICAL: GemmaScope SAEs are trained on resid_POST (unlike gpt2-small-res-jb
    which was trained on resid_PRE). This means our hooks and SAEs use the same
    tensor — no mismatch. This is why we MUST use GemmaScope and cannot substitute
    gpt2-small-res-jb. (FIX #4 upstream cause resolved here.)

    Activation: JumpReLU (not vanilla ReLU). sae.encode() handles this correctly.
    """
    key = (SAE_RELEASE, layer)
    if key in _sae_cache:
        return _sae_cache[key]
    with _lock:
        if key in _sae_cache:
            return _sae_cache[key]
        logger.info("Loading SAE %s @ layer %d ...", SAE_RELEASE, layer)
        from sae_lens import SAE
        is_gemma = "gemma" in MODEL_NAME.lower()
        sae_id = f"layer_{layer}/width_16k/canonical" if is_gemma else f"blocks.{layer}.hook_resid_pre"
        sae, cfg, _sparsity = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
            device="cpu",
        )
        sae.eval()
        _sae_cache[key] = (sae, cfg)
        logger.info(
            "Loaded GemmaScope layer %d: d_in=%d d_sae=%d",
            layer, sae.cfg.d_in, sae.cfg.d_sae,
        )
        return _sae_cache[key]


def model_info() -> dict:
    """Best-effort metadata without forcing a model load."""
    if _model is None:
        return {
            "model": MODEL_NAME,
            "loaded": False,
            "sae_release": SAE_RELEASE,
            "sae_default_layer": DEFAULT_SAE_LAYER,
            "capture_layers": CAPTURE_LAYERS,
        }
    return {
        "model": MODEL_NAME,
        "loaded": True,
        "n_layers": _model.cfg.n_layers,   # 26 for Gemma-2-2b
        "d_model": _model.cfg.d_model,     # 2304
        "n_heads": _model.cfg.n_heads,     # 8
        "d_vocab": _model.cfg.d_vocab,
        "capture_layers": CAPTURE_LAYERS,
        "sae_release": SAE_RELEASE,
        "sae_default_layer": DEFAULT_SAE_LAYER,
        "sae_layers_cached": [k[1] for k in _sae_cache.keys()],
    }
