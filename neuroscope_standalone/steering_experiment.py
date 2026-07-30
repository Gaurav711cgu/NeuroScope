"""
NeuroScope v2 — Steering Intervention Experiment
Tests whether amplifying causal features during generation reduces entropy-spike rate.
Outputs: comparison table — unsteered vs steered entropy trajectories.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_lens import HookedTransformer
from sae_lens import SAE


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    return float(-(probs * (probs + 1e-10).log()).sum().item())


def run_steered_trajectory(
    model: HookedTransformer,
    sae: SAE,
    question: str,
    causal_feature_ids: list[int],
    alpha: float = 10.0,
    sae_layer: int = 12,
    n_steps: int = 5,
    device: str = None,
) -> dict:
    """Run trajectory with causal features amplified. Returns entropy per step."""
    if device is None:
        device = get_default_device()
        
    hook_name = f"blocks.{sae_layer}.hook_resid_post"
    dtype = model.cfg.dtype
    
    # Build combined steering vector: sum of top feature decoder directions
    W_steer = torch.zeros(model.cfg.d_model, device=device, dtype=dtype)
    for fid in causal_feature_ids[:5]:
        W_steer += sae.W_dec[fid].to(device=device, dtype=dtype)
    W_steer = W_steer / (W_steer.norm() + 1e-10)
    
    def steer_hook(value, hook):
        return value + alpha * W_steer.unsqueeze(0).unsqueeze(0)
    
    COT_TEMPLATE = "Answer the question using step-by-step reasoning.\nQuestion: {q}\n{steps}Step {n}: "
    steps_so_far = ""
    entropies = []
    
    for step in range(1, n_steps + 1):
        prompt = COT_TEMPLATE.format(q=question, steps=steps_so_far, n=step)
        tokens = model.to_tokens(prompt).to(device)
        
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                logits, cache = model.run_with_cache(tokens, names_filter=[])
        
        entropy_nats = compute_entropy(logits[0, -1]) / 10.0
        entropies.append(min(float(entropy_nats), 1.0))
        
        # Generate step output with steering
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                out = model.generate(tokens, max_new_tokens=60, do_sample=False, verbose=False)
        step_out = model.to_string(out[0, tokens.shape[1]:]).strip()
        steps_so_far += f"Step {step}: {step_out}\n"
    
    return {"entropies": entropies, "final_output": steps_so_far.split("Step 5:")[-1].strip()[:200]}


def run_steering_experiment(
    jsonl_path: str,
    circuit_path: str,
    model: HookedTransformer,
    sae: SAE,
    output_dir: str = "./results/",
    alpha_values: list[float] = [5.0, 10.0, 20.0],
    n_test: int = 50,
    sae_layer: int = 12,
    device: str = None,
):
    """Main experiment: compare unsteered vs steered entropy trajectories."""
    if device is None:
        device = get_default_device()
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    
    with open(circuit_path) as f:
        circuit = json.load(f)
    
    causal_feature_ids = [f["feature_id"] for f in circuit]
    print(f"Using {len(causal_feature_ids)} causal features for steering")
    
    test_runs = [
        r for r in results
        if not r["final_correct"] and any(s["entropy"] > 0.7 for s in r["steps"][:3])
    ][:n_test]
    
    print(f"Test runs: {len(test_runs)}")
    
    all_results = []
    spike_threshold = 0.7
    
    for i, run in enumerate(test_runs):
        print(f"[{i+1}/{len(test_runs)}] {run['question'][:50]}")
        
        unsteered_entropies = [s["entropy"] for s in run["steps"]]
        unsteered_spikes = sum(1 for e in unsteered_entropies if e > spike_threshold)
        
        row = {
            "question": run["question"][:100],
            "unsteered_spike_rate": unsteered_spikes / len(unsteered_entropies),
            "unsteered_entropies": unsteered_entropies,
        }
        
        for alpha in alpha_values:
            try:
                steered = run_steered_trajectory(
                    model, sae, run["question"], causal_feature_ids, alpha, sae_layer, 5, device
                )
                steered_spikes = sum(1 for e in steered["entropies"] if e > spike_threshold)
                row[f"steered_alpha{alpha}_spike_rate"] = steered_spikes / len(steered["entropies"])
                row[f"steered_alpha{alpha}_entropies"] = steered["entropies"]
            except Exception as e:
                print(f"  Steering failed at alpha={alpha}: {e}")
        
        all_results.append(row)
    
    print("\n=== STEERING EXPERIMENT RESULTS ===")
    print(f"{'Condition':<30} {'Spike Rate':>12} {'vs Unsteered':>15}")
    print("-" * 60)
    
    base_spike_rate = np.mean([r["unsteered_spike_rate"] for r in all_results]) if all_results else 0.0
    print(f"{'Unsteered (baseline)':<30} {base_spike_rate:>11.2%} {'—':>15}")
    
    for alpha in alpha_values:
        key = f"steered_alpha{alpha}_spike_rate"
        if any(key in r for r in all_results):
            rates = [r[key] for r in all_results if key in r]
            steered_rate = np.mean(rates)
            delta = (steered_rate - base_spike_rate) / (base_spike_rate + 1e-10) * 100
            direction = "↓" if delta < 0 else "↑"
            print(f"{'Steered α='+str(alpha):<30} {steered_rate:>11.2%} {direction}{abs(delta):>13.1f}%")
    
    out_path = Path(output_dir) / "steering_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope v2 Steering Experiment")
    parser.add_argument("jsonl_path", help="Path to trajectories .jsonl file")
    parser.add_argument("circuit_path", help="Path to circuit_features.json")
    parser.add_argument("--model", default="gemma-2-2b-it", help="Model name")
    parser.add_argument("--sae-layer", type=int, default=12, help="SAE layer")
    parser.add_argument("--output-dir", default="./results/", help="Output directory")
    args = parser.parse_args()
    
    from neuroscope_standalone.batch_runner import load_model_and_sae
    model, sae = load_model_and_sae(args.model, args.sae_layer)
    run_steering_experiment(args.jsonl_path, args.circuit_path, model, sae, args.output_dir, sae_layer=args.sae_layer)
