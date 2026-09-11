"""
OpenAI-Style Automated SAE Feature Auto-Interpretation & Validation Pipeline
=============================================================================
Replicates OpenAI's "Language models can explain neurons in language models"
(Bills et al., 2023) paper for Sparse Autoencoders:

1. Top-Snippet Extraction: Extracts top-activating text sequences for feature f_i.
2. Hypothesis Generation: Prompts LLM Explainer (Gemini / OpenAI API) for feature explanation.
3. Scoring & Validation: Predicts feature activation on held-out test sequences,
   computing Quantitative F1-score & Pearson Correlation (r).
"""
from __future__ import annotations

import json
import logging
import re
import numpy as np
from scipy import stats

logger = logging.getLogger("neuroscope.auto_interp")

AUTO_INTERP_EXPLAINER_PROMPT = """You are an automated interpretability explainer evaluating Sparse Autoencoder (SAE) feature activations in a large language model.

Below are top text snippets where Feature #{feature_id} (Layer {layer}) activated most strongly. The target token where activation peaked is surrounded by brackets `[[token]]`.

Top Activating Snippets:
{activating_snippets}

Your Task:
Provide a concise, 1-sentence hypothesis explaining what specific semantic, syntactic, or logical concept activates this feature.
Respond ONLY with the JSON object:
{{"feature_id": {feature_id}, "explanation": "your explanation here", "confidence": 0.95}}
"""


def extract_top_activating_snippets(
    feature_id: int,
    activations_by_prompt: list[dict],
    top_k: int = 5
) -> list[dict]:
    """Extract top-k text sequences where feature_id exhibited maximum activation."""
    scored_snippets = []
    for item in activations_by_prompt:
        prompt = item["prompt"]
        acts = item.get("activations", [])
        if len(acts) > 0:
            max_act = float(np.max(acts))
            max_idx = int(np.argmax(acts))
            scored_snippets.append({
                "prompt": prompt,
                "max_activation": max_act,
                "token_index": max_idx
            })
            
    scored_snippets.sort(key=lambda x: x["max_activation"], reverse=True)
    return scored_snippets[:top_k]


def evaluate_auto_interp_prediction_score(
    true_activations: np.ndarray,
    predicted_activations: np.ndarray
) -> dict:
    """Quantitative evaluation (Bills et al., 2023) comparing predicted vs true feature activations.
    
    Computes Pearson correlation r and binary F1-score at threshold 0.10.
    """
    assert len(true_activations) == len(predicted_activations), "Array lengths must match"
    
    # 1. Pearson Correlation
    if len(np.unique(true_activations)) < 2 or len(np.unique(predicted_activations)) < 2:
        r_val = 0.0
    else:
        r_val, _ = stats.pearsonr(true_activations, predicted_activations)
        if np.isnan(r_val):
            r_val = 0.0
            
    # 2. Binary F1 Score (threshold >= 0.10)
    true_binary = (true_activations >= 0.10).astype(int)
    pred_binary = (predicted_activations >= 0.10).astype(int)
    
    tp = np.sum((true_binary == 1) & (pred_binary == 1))
    fp = np.sum((true_binary == 0) & (pred_binary == 1))
    fn = np.sum((true_binary == 1) & (pred_binary == 0))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        "pearson_r": round(float(r_val), 4),
        "f1_score": round(float(f1_score), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "auto_interp_valid": bool(r_val >= 0.60 and f1_score >= 0.65)
    }
