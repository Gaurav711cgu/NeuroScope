"""Lazy loaders for Gemma-2-2b-it HookedTransformer and GemmaScope SAEs.
Delegates to centralized core/state.py state manager.
"""
from __future__ import annotations

import logging
import os
import torch

from core import state

logger = logging.getLogger(__name__)

# Re-expose configurations for backward compatibility
MODEL_NAME = state.MODEL_NAME
SAE_RELEASE = state.SAE_RELEASE
DEFAULT_SAE_LAYER = state.DEFAULT_SAE_LAYER
CAPTURE_LAYERS = state.CAPTURE_LAYERS


def get_model():
    """Load Gemma-2-2b-it as a HookedTransformer on CPU/GPU."""
    if state.model is None:
        with state.state_lock:
            if state.model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                quant = os.environ.get("NEUROSCOPE_QUANTIZATION", "").lower()
                load_in_8bit = quant == "8bit"
                load_in_4bit = quant == "4bit"

                logger.info("Loading HookedTransformer %s on %s ...", state.MODEL_NAME, device)
                from transformer_lens import HookedTransformer

                model_kwargs = {
                    "device": device,
                    "fold_ln": True,
                    "center_writing_weights": True,
                    "center_unembed": True,
                    "dtype": torch.bfloat16 if "gemma" in state.MODEL_NAME.lower() else None,
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

                state.model = HookedTransformer.from_pretrained(
                    state.MODEL_NAME,
                    **model_kwargs
                )
                state.model.eval()
                logger.info(
                    "Loaded %s: n_layers=%d d_model=%d n_heads=%d",
                    state.MODEL_NAME,
                    state.model.cfg.n_layers,
                    state.model.cfg.d_model,
                    state.model.cfg.n_heads,
                )
    return state.model


def get_sae(layer: int = DEFAULT_SAE_LAYER):
    """Load GemmaScope residual SAE for a given layer."""
    key = (state.SAE_RELEASE, layer)
    if key in state.sae_cache:
        return state.sae_cache[key]
    with state.state_lock:
        if key in state.sae_cache:
            return state.sae_cache[key]
        logger.info("Loading SAE %s @ layer %d ...", state.SAE_RELEASE, layer)
        from sae_lens import SAE
        is_gemma = "gemma" in state.MODEL_NAME.lower()
        sae_id = f"layer_{layer}/width_16k/canonical" if is_gemma else f"blocks.{layer}.hook_resid_pre"
        sae, cfg, _sparsity = SAE.from_pretrained(
            release=state.SAE_RELEASE,
            sae_id=sae_id,
            device="cpu",
        )
        sae.eval()
        state.sae_cache[key] = (sae, cfg)
        logger.info(
            "Loaded GemmaScope layer %d: d_in=%d d_sae=%d",
            layer, sae.cfg.d_in, sae.cfg.d_sae,
        )
        return state.sae_cache[key]


def model_info() -> dict:
    """Best-effort metadata without forcing a model load."""
    if state.model is None:
        return {
            "model": state.MODEL_NAME,
            "loaded": False,
            "sae_release": state.SAE_RELEASE,
            "sae_default_layer": state.DEFAULT_SAE_LAYER,
            "capture_layers": state.CAPTURE_LAYERS,
        }
    return {
        "model": state.MODEL_NAME,
        "loaded": True,
        "n_layers": state.model.cfg.n_layers,
        "d_model": state.model.cfg.d_model,
        "n_heads": state.model.cfg.n_heads,
        "d_vocab": state.model.cfg.d_vocab,
        "capture_layers": state.CAPTURE_LAYERS,
        "sae_release": state.SAE_RELEASE,
        "sae_default_layer": state.DEFAULT_SAE_LAYER,
        "sae_layers_cached": [k[1] for k in state.sae_cache.keys()],
    }
