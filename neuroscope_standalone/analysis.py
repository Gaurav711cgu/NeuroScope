"""
NeuroScope v2 — Statistical Analysis + Figure Generation
Reads .jsonl results from batch_runner.py.
Outputs: CI table, KM survival curve, entropy trajectory figure.
Dependencies: scipy, numpy, matplotlib only.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats


def load_results(jsonl_path: str) -> list[dict]:
    results = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def spearman_bootstrap(x, y, n_boot: int = 2000, seed: int = 42) -> dict:
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    rng = np.random.default_rng(seed)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"rho": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_width": 0.0}
    rho, pval = stats.spearmanr(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        if len(np.unique(x[idx])) < 2:
            boots.append(0.0)
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        boots.append(float(r) if not np.isnan(r) else 0.0)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        "rho": round(float(rho), 3),
        "pval": round(float(pval), 5),
        "ci_low": round(float(ci_low), 3),
        "ci_high": round(float(ci_high), 3),
        "ci_width": round(float(ci_high - ci_low), 3),
    }


def compute_warning_horizon(results: list[dict]) -> dict[str, float]:
    """For failing trajectories, at which step does each signal first exceed threshold?"""
    failing = [r for r in results if not r["final_correct"]]
    horizons = {"entropy": [], "attention_diffusion": [], "drift_proxy": []}
    
    for r in failing:
        n = len(r["steps"])
        for signal, threshold in [("entropy", 0.7), ("attention_diffusion", 0.6), ("drift_proxy", 0.5)]:
            first_spike = None
            for s in r["steps"]:
                if s.get(signal, 0.0) > threshold:
                    first_spike = s["step_n"]
                    break
            if first_spike is not None:
                steps_before_end = n - first_spike
                horizons[signal].append(steps_before_end)
    
    return {sig: round(float(np.mean(v)), 2) if v else 0.0 for sig, v in horizons.items()}


def kaplan_meier(T, E):
    """KM estimator. T = step of first spike. E = event occurred (1) or censored (0)."""
    times = np.sort(np.unique(T))
    S = 1.0
    km_times, km_probs = [0], [1.0]
    for t in times:
        n_risk = np.sum(T >= t)
        n_events = np.sum((T == t) & (E == 1))
        if n_risk > 0:
            S *= (1 - n_events / n_risk)
        km_times.append(int(t))
        km_probs.append(S)
    return km_times, km_probs


def run_full_analysis(jsonl_path: str, output_dir: str = "./results/"):
    """Main analysis function. Produces all tables and figures."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = load_results(jsonl_path)
    print(f"Loaded {len(results)} trajectories")
    
    if not results:
        print("No trajectories found in jsonl file.")
        return {}
    
    # Flatten step-level data
    all_entropy, all_attn, all_drift, all_correct = [], [], [], []
    for r in results:
        label = 1.0 if r["final_correct"] else 0.0
        for s in r["steps"]:
            all_entropy.append(s["entropy"])
            all_attn.append(s["attention_diffusion"])
            all_drift.append(s["drift_proxy"])
            all_correct.append(label)
    
    # Bootstrap CIs
    entropy_ci = spearman_bootstrap(all_entropy, all_correct)
    attn_ci = spearman_bootstrap(all_attn, all_correct)
    drift_ci = spearman_bootstrap(all_drift, all_correct)
    horizon = compute_warning_horizon(results)
    
    n_correct = sum(1 for r in results if r["final_correct"])
    
    print("\n=== RESULTS TABLE ===")
    print(f"N trajectories: {len(results)} ({n_correct} correct, {len(results)-n_correct} incorrect)")
    print(f"Accuracy: {n_correct/len(results):.1%}")
    print(f"\n{'Signal':<25} {'ρ':>6} {'CI [2.5%, 97.5%]':>20} {'Width':>8} {'Horizon':>8}")
    print("-" * 75)
    for label, ci, h_key in [
        ("Next-Token Entropy", entropy_ci, "entropy"),
        ("Attention Diffusion", attn_ci, "attention_diffusion"),
        ("SAE Feature Drift", drift_ci, "drift_proxy"),
    ]:
        print(f"{label:<25} {ci['rho']:>6.3f} [{ci['ci_low']:>6.3f}, {ci['ci_high']:>6.3f}]  "
              f"{ci['ci_width']:>7.3f} {horizon[h_key]:>7.1f}s")
    
    # Figure 1: Entropy trajectory — correct vs incorrect
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    correct_runs = [r for r in results if r["final_correct"]]
    failing_runs = [r for r in results if not r["final_correct"]]
    n_steps = max(len(r["steps"]) for r in results) if results else 5
    
    for ax, (signal, label) in zip(axes, [
        ("entropy", "Next-Token Entropy"),
        ("attention_diffusion", "Attention Diffusion"),
        ("drift_proxy", "Feature Drift"),
    ]):
        ci_map = {"entropy": entropy_ci, "attention_diffusion": attn_ci, "drift_proxy": drift_ci}
        for runs, color, lbl in [(correct_runs, "#2196F3", "Correct"), (failing_runs, "#F44336", "Incorrect")]:
            step_vals = [[] for _ in range(n_steps)]
            for r in runs:
                for s in r["steps"]:
                    step_vals[s["step_n"] - 1].append(s[signal])
            means = [np.mean(v) if v else 0.0 for v in step_vals]
            stds = [np.std(v) if v else 0.0 for v in step_vals]
            xs = list(range(1, n_steps + 1))
            ax.plot(xs, means, color=color, label=lbl, linewidth=2)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)
        ax.set_xlabel("Reasoning Step", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label}\nρ={ci_map[signal]['rho']:.3f}", fontsize=11)
        ax.set_xticks(range(1, n_steps + 1))
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    
    plt.suptitle("NeuroScope v2: Hallucination Early-Warning Signals", fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = Path(output_dir) / "fig1_signal_trajectories.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    
    summary = {
        "n_trajectories": len(results),
        "accuracy": round(n_correct / len(results), 4) if results else 0.0,
        "entropy": entropy_ci,
        "attention_diffusion": attn_ci,
        "feature_drift": drift_ci,
        "warning_horizon": horizon,
    }
    with open(Path(output_dir) / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope v2 Analysis & Visualization")
    parser.add_argument("jsonl_path", help="Path to trajectories .jsonl file")
    parser.add_argument("--output-dir", default="./results/", help="Output directory")
    args = parser.parse_args()
    
    run_full_analysis(args.jsonl_path, args.output_dir)
