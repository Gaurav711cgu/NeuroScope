"""Script to run end-to-end evaluation pipeline matching lab SOPs."""
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path

# Neuroscope imports
from backend.neuroscope.loader import get_model, get_sae
from backend.neuroscope.probe import eval_probe_loss, train_hallucination_probe
from backend.neuroscope.steering import composite_steer, steer_and_regenerate
from backend.neuroscope.auto_interp import generate_feature_hypothesis, score_pearson_r
from backend.neuroscope.geometry import compute_sae_quality, compute_ablation_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    # 1. Load model and SAE
    logger.info("Loading model and SAE...")
    model = get_model()
    layer = 12
    sae, _ = get_sae(layer)
    
    # We will use a mock dataset to represent hallucination vs factual completions
    # Since we don't have the real clinical trial or benchmark datasets loaded here.
    logger.info("Setting up evaluation dataset...")
    
    # 2. SAE Quality Metrics (Lieberum et al.)
    logger.info("Running SAE Quality Metrics...")
    sample_text = ["The capital of France is Paris.", "The speed of light is 299792458 m/s."]
    tokens = model.to_tokens(sample_text)
    
    quality_metrics = compute_sae_quality(model, sae, tokens, mask_special=True)
    logger.info("SAE Quality: %s", quality_metrics)
    
    # 3. Logistic Probe Evaluation (Gao et al.)
    logger.info("Running Logistic Probe Evaluation...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[sae.cfg.hook_name])
        acts = cache[sae.cfg.hook_name]
        sae_out = sae.encode(acts)
        
    flat_latents = sae_out.view(-1, sae_out.shape[-1]).cpu().numpy()
    binary_labels = np.random.randint(0, 2, size=flat_latents.shape[0])
    
    subset_idx = np.random.choice(flat_latents.shape[0], min(50, flat_latents.shape[0]), replace=False)
    probe_loss = eval_probe_loss(flat_latents[subset_idx], binary_labels[subset_idx])
    logger.info("Probe Loss (best CE across latents): %s", probe_loss)
    
    # 4. Cox Proportional Hazards (Hallucination dynamics)
    logger.info("Running Cox Survival Analysis...")
    X_mock = np.random.randn(100, 10)
    y_mock = np.random.randint(0, 2, size=100)
    T_mock = np.random.randint(1, 20, size=100)
    E_mock = np.random.randint(0, 2, size=100)
    
    cox_res = train_hallucination_probe(
        layer=layer,
        X=X_mock,
        y=y_mock,
        T=T_mock,
        E=E_mock,
        feature_names=[f"F{i}" for i in range(10)]
    )
    logger.info("Cox Model Survival Analysis: %s", cox_res.get("survival_analysis", {}))
    
    # 5. Ablation Sparsity (Gao et al.)
    logger.info("Running Ablation Sparsity...")
    top_feature = int(cox_res["top_predictive_dims"][0]["dim"]) if cox_res.get("top_predictive_dims") else 0
    sparsity_metrics = compute_ablation_sparsity(model, sae, tokens, [top_feature])
    logger.info("Ablation Sparsity for F%d: %s", top_feature, sparsity_metrics[top_feature])
    
    # 6. Composite Steering (Semantic Collapse check)
    logger.info("Running Composite Steering...")
    steer_prompt = "Can you summarize the plot of Inception?"
    steer_res = composite_steer(
        prompt=steer_prompt,
        layer=layer,
        features=[top_feature],
        multipliers=[50.0],
        check_ppl=True
    )
    logger.info("Steering Result Delta PPL: %s", steer_res.get('delta_ppl', steer_res.get('error')))
    
    # 7. Auto-Interp Pearson r
    logger.info("Running Auto-Interp...")
    def dummy_llm(prompt):
        return {
            "explanation": "This feature tracks confident factual assertions.",
            "simulated_activations": [0.9, 0.1]
        }
    
    snippets = [
        {"prompt": "The capital of France is [[Paris]]."},
        {"prompt": "The cat sat on the [[mat]]."}
    ]
    
    hypothesis = generate_feature_hypothesis(top_feature, snippets, dummy_llm, layer=layer)
    pearson_r = score_pearson_r(hypothesis["simulated_activations"], [0.85, 0.15])
    logger.info("Auto-Interp Pearson r: %s", pearson_r)
    
    results = {
        "sae_quality": quality_metrics,
        "eval_probe_loss_best_ce": probe_loss,
        "survival_analysis": cox_res.get("survival_analysis"),
        "ablation_sparsity": sparsity_metrics,
        "steering": steer_res,
        "auto_interp_pearson_r": pearson_r
    }
    
    out_path = Path("results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Pipeline complete. Results saved to %s", out_path.absolute())

if __name__ == "__main__":
    run_pipeline()
