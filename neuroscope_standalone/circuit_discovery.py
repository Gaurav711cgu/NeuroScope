"""
NeuroScope v2 — Causal Circuit Discovery
Identifies features active at entropy spikes, measures causal contribution via ablation/path patching,
labels features via Neuronpedia API, and plots the causal circuit graph.
"""
from __future__ import annotations
import argparse
import json
import time
import requests
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from transformer_lens import HookedTransformer
from sae_lens import SAE

ENTROPY_SPIKE_THRESHOLD = 0.70
CAUSAL_EFFECT_THRESHOLD = 0.10  # nats — ablation must reduce entropy by this much
CANDIDATE_FREQ_THRESHOLD = 0.30  # feature must appear in ≥30% of spike steps


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_entropy(logits_tensor: torch.Tensor) -> float:
    probs = F.softmax(logits_tensor.float(), dim=-1)
    return float(-(probs * (probs + 1e-10).log()).sum().item())


def identify_spike_features(results: list[dict], spike_threshold: float = ENTROPY_SPIKE_THRESHOLD) -> list[int]:
    """Step 1: Which features are active at entropy-spike steps in failing runs?"""
    feature_counter = Counter()
    total_spike_steps = 0
    
    for r in results:
        if r["final_correct"]:
            continue
        for s in r["steps"]:
            if s["entropy"] > spike_threshold:
                total_spike_steps += 1
                for f in s.get("top_features", []):
                    feature_counter[f["feature_id"]] += 1
    
    if total_spike_steps == 0:
        print(f"WARNING: No spike steps found at threshold {spike_threshold}")
        return []
    
    candidates = [
        fid for fid, count in feature_counter.items()
        if count / total_spike_steps >= CANDIDATE_FREQ_THRESHOLD
    ]
    print(f"Spike steps: {total_spike_steps} | Candidate features: {len(candidates)}")
    return candidates


def measure_causal_effect(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    feature_id: int,
    sae_layer: int = 12,
    device: str = "cuda",
) -> float:
    """
    Ablate feature_id at layer sae_layer and measure entropy change at final token.
    Returns entropy_delta = entropy_baseline - entropy_ablated
    Positive delta = ablation REDUCED entropy = feature was upstream of high entropy.
    """
    hook_name = f"blocks.{sae_layer}.hook_resid_post"
    tokens = model.to_tokens(prompt).to(device)
    
    with torch.no_grad():
        baseline_logits, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    entropy_baseline = compute_entropy(baseline_logits[0, -1])
    
    dtype = model.cfg.dtype
    W_dec_feature = sae.W_dec[feature_id].to(device=device, dtype=dtype)
    
    resid = cache[hook_name][0].float()
    with torch.no_grad():
        features = sae.encode(resid.unsqueeze(0))
    act_A = features[0, -1, feature_id].item()
    
    def ablate_hook(value, hook):
        value[:, -1, :] = value[:, -1, :] - act_A * W_dec_feature.unsqueeze(0)
        return value
    
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, ablate_hook)]):
            ablated_logits = model(tokens, return_type="logits")
    entropy_ablated = compute_entropy(ablated_logits[0, -1])
    
    return entropy_baseline - entropy_ablated


def label_features_neuronpedia(feature_ids: list[int], layer: int = 12, model_slug: str = "gemma-2-2b") -> dict[int, str]:
    """Call Neuronpedia API for each feature. Free, no auth."""
    labels = {}
    for fid in feature_ids:
        url = f"https://www.neuronpedia.org/api/feature/{model_slug}/{layer}/{fid}"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                desc = data.get("explanations", [{}])[0].get("description", "No label")
                labels[fid] = desc[:80]
            else:
                labels[fid] = f"Feature {fid}"
        except Exception:
            labels[fid] = f"Feature {fid}"
        time.sleep(0.2)
    return labels


def plot_circuit_diagram(causal_features: list[dict], output_path: str):
    """Create circuit diagram: nodes=features, edges=causal contribution."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if not causal_features:
        ax.text(0.5, 0.5, "No causal features found", ha="center", va="center", fontsize=14)
        plt.savefig(output_path, dpi=150)
        return
    
    n = len(causal_features)
    xs = np.linspace(0.1, 0.9, min(n, 10))
    rng = np.random.RandomState(42)
    ys = rng.uniform(0.2, 0.8, len(xs))

    
    max_delta = max(f["entropy_delta"] for f in causal_features[:len(xs)]) or 1.0
    
    for i, (feat, x, y) in enumerate(zip(causal_features[:len(xs)], xs, ys)):
        size = 500 + 2000 * (feat["entropy_delta"] / max_delta)
        ax.scatter([x], [y], s=size, c="#E91E63", alpha=0.7, zorder=5)
        label = feat.get("label", f"F{feat['feature_id']}")[:30]
        ax.annotate(f"#{feat['feature_id']}\n{label}\nΔH={feat['entropy_delta']:.2f}",
                    (x, y), ha="center", va="center", fontsize=7, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    ax.scatter([0.5], [0.05], s=800, c="#FF5722", marker="D", zorder=5)
    ax.annotate("Entropy\nSpike", (0.5, 0.05), ha="center", va="center", fontsize=9, zorder=6)
    
    for feat, x, y in zip(causal_features[:len(xs)], xs, ys):
        weight = feat["entropy_delta"] / max_delta
        ax.annotate("", xy=(0.5, 0.08), xytext=(x, y - 0.05),
                    arrowprops=dict(arrowstyle="->", color="#9C27B0",
                                   lw=0.5 + 2 * weight, alpha=0.5 + 0.5 * weight))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"NeuroScope v2: Causal Circuit — {n} GemmaScope Layer-12 Features\n"
                 f"Node size = causal contribution (entropy delta in nats)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Circuit diagram saved: {output_path}")


def run_circuit_discovery(
    jsonl_path: str,
    model: HookedTransformer,
    sae: SAE,
    output_dir: str = "./results/",
    sae_layer: int = 12,
    device: str = None,
    max_trajectories_for_patching: int = 50,
):
    """Main circuit discovery pipeline."""
    if device is None:
        device = get_default_device()
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    
    candidates = identify_spike_features(results)
    if not candidates:
        print("No candidates found. Lower CANDIDATE_FREQ_THRESHOLD or check data.")
        return []
    
    print(f"\nRunning path patching on {len(candidates)} candidate features...")
    
    patching_data = []
    for r in results:
        if r["final_correct"] or len(patching_data) >= max_trajectories_for_patching:
            continue
        for s in r["steps"]:
            if s["entropy"] > ENTROPY_SPIKE_THRESHOLD and s["step_n"] < len(r["steps"]):
                prompt = f"Question: {r['question']}\nStep {s['step_n']}: {s['output']}\nStep {s['step_n']+1}:"
                patching_data.append({"prompt": prompt, "step_n": s["step_n"]})
                break
    
    causal_results = []
    for i, fid in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] Feature {fid}...", end=" ")
        deltas = []
        for pd in patching_data[:20]:
            try:
                delta = measure_causal_effect(model, sae, pd["prompt"], fid, sae_layer, device)
                deltas.append(delta)
            except Exception as e:
                pass
        
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        print(f"ΔH = {mean_delta:.3f}")
        
        if mean_delta > CAUSAL_EFFECT_THRESHOLD:
            causal_results.append({
                "feature_id": fid,
                "entropy_delta": round(mean_delta, 4),
                "n_patching_runs": len(deltas),
            })
    
    causal_results.sort(key=lambda x: x["entropy_delta"], reverse=True)
    print(f"\nCausal features confirmed: {len(causal_results)}")
    
    if not causal_results:
        print("No causal features found above threshold.")
        return []
    
    print("\nLabeling via Neuronpedia API...")
    feature_ids = [f["feature_id"] for f in causal_results]
    labels = label_features_neuronpedia(feature_ids, layer=sae_layer)
    for f in causal_results:
        f["label"] = labels.get(f["feature_id"], "Unknown")
    
    out_path = Path(output_dir) / "circuit_features.json"
    with open(out_path, "w") as f:
        json.dump(causal_results, f, indent=2)
    print(f"Circuit features saved: {out_path}")
    
    plot_circuit_diagram(causal_results, str(Path(output_dir) / "fig2_circuit_diagram.png"))
    return causal_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope v2 Circuit Discovery")
    parser.add_argument("jsonl_path", help="Path to trajectories .jsonl file")
    parser.add_argument("--model", default="gemma-2-2b-it", help="Model name")
    parser.add_argument("--sae-layer", type=int, default=12, help="SAE layer")
    parser.add_argument("--output-dir", default="./results/", help="Output directory")
    args = parser.parse_args()
    
    from neuroscope_standalone.batch_runner import load_model_and_sae
    model, sae = load_model_and_sae(args.model, args.sae_layer)
    run_circuit_discovery(args.jsonl_path, model, sae, args.output_dir, args.sae_layer)
