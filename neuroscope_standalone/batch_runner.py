"""
NeuroScope v2 — Standalone Batch Runner
Zero PostgreSQL, zero FastAPI, zero Docker.
Runs on Google Colab T4, Kaggle, or local machine.

Usage:
    python -m neuroscope_standalone.batch_runner --n-triviaqa 200 --n-hotpotqa 100 --model gemma-2-2b-it
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_sae(model_name: str, sae_layer: int = 12, device: str = None):
    """Load Gemma-2-2b-it or Gemma-2-9b-it with GemmaScope SAE."""
    if device is None:
        device = get_default_device()
    print(f"Loading model: {model_name} on device: {device}")
    
    tl_name_map = {
        "gemma-2-2b-it": "gemma-2-2b-it",
        "gemma-2-9b-it": "gemma-2-9b-it",
    }
    
    dtype = torch.float16 if device != "cpu" else torch.float32
    
    model = HookedTransformer.from_pretrained(
        tl_name_map[model_name],
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    
    sae_release_map = {
        "gemma-2-2b-it": "gemma-scope-2b-pt-res",
        "gemma-2-9b-it": "gemma-scope-9b-pt-res",
    }
    sae_id = f"layer_{sae_layer}/width_16k/average_l0_71"
    
    print(f"Loading SAE: {sae_release_map[model_name]} / {sae_id}")
    sae, _, _ = SAE.from_pretrained(
        release=sae_release_map[model_name],
        sae_id=sae_id,
    )
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    
    return model, sae


def run_single_trajectory(
    model: HookedTransformer,
    sae: SAE,
    question: str,
    answers: list[str],
    sae_layer: int = 12,
    n_steps: int = 5,
    device: str = None,
) -> dict:
    """
    Run one N-step CoT trajectory. Returns metrics dict.
    Does NOT save raw activations to disk — computes all metrics on-the-fly.
    """
    if device is None:
        device = get_default_device()
        
    hook_name = f"blocks.{sae_layer}.hook_resid_post"
    last_attn_name = f"blocks.{model.cfg.n_layers - 1}.attn.hook_pattern"
    
    COT_TEMPLATE = (
        "Answer the question using step-by-step reasoning.\n"
        "Question: {q}\n"
        "{steps}Step {n}: "
    )
    
    steps_so_far = ""
    step_metrics = []
    
    for step in range(1, n_steps + 1):
        prompt = COT_TEMPLATE.format(q=question, steps=steps_so_far, n=step)
        tokens = model.to_tokens(prompt).to(device)
        
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name, last_attn_name],
                return_type="logits",
            )
        
        # Signal 1: Next-token entropy
        last_logits = logits[0, -1].float()
        probs = F.softmax(last_logits, dim=-1)
        entropy_nats = float(-(probs * (probs + 1e-10).log()).sum().item())
        entropy_norm = min(entropy_nats / 10.0, 1.0)
        
        # Signal 2: Attention diffusion (last layer)
        attn = cache[last_attn_name][0].float()  # [heads, q, k]
        a_safe = attn + 1e-10
        attn_ent = float(-(a_safe * a_safe.log()).sum(-1).mean().item())
        attn_norm = min(attn_ent / 6.0, 1.0)
        
        # Signal 3: GemmaScope SAE features
        resid = cache[hook_name][0].float()  # [seq_len, d_model]
        with torch.no_grad():
            features = sae.encode(resid.unsqueeze(0))  # [1, seq_len, 16384]
        last_features = features[0, -1]  # [16384]
        n_active = int((last_features > 0).sum().item())
        top_k = last_features.topk(min(25, n_active or 25))
        top_features = [
            {"feature_id": int(i), "activation": round(float(v), 4)}
            for i, v in zip(top_k.indices.tolist(), top_k.values.tolist())
            if v > 0
        ]
        
        # Feature drift proxy (variance across steps)
        drift_proxy = float(np.var([f["activation"] for f in top_features])) if top_features else 0.0
        drift_norm = min(drift_proxy / 50.0, 1.0)
        
        # Generate next step output (greedy decode, max 60 tokens)
        with torch.no_grad():
            out_tokens = model.generate(
                tokens,
                max_new_tokens=60,
                do_sample=False,
                verbose=False,
            )
        step_output = model.to_string(out_tokens[0, tokens.shape[1]:])
        
        # Append to CoT history
        steps_so_far += f"Step {step}: {step_output.strip()}\n"
        
        step_metrics.append({
            "step_n": step,
            "entropy": round(entropy_norm, 4),
            "attention_diffusion": round(attn_norm, 4),
            "drift_proxy": round(drift_norm, 4),
            "n_active_features": n_active,
            "top_features": top_features,
            "output": step_output.strip()[:300],
        })
    
    # Grade the final output
    final_output = step_metrics[-1]["output"].lower()
    correct = any(str(ans).lower() in final_output for ans in answers)
    
    return {
        "question": question[:200],
        "answers": [str(a) for a in answers[:3]],
        "final_correct": correct,
        "steps": step_metrics,
    }


def run_batch_experiment(
    model_name: str = "gemma-2-2b-it",
    n_triviaqa: int = 200,
    n_hotpotqa: int = 100,
    sae_layer: int = 12,
    n_steps: int = 5,
    output_dir: str = "./results/",
    device: str = None,
    seed: int = 42,
):
    """Main entry point. Runs batch experiment and saves .jsonl results."""
    if device is None:
        device = get_default_device()
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model, sae = load_model_and_sae(model_name, sae_layer, device)
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path(output_dir) / f"trajectories_{model_name}_{timestamp}.jsonl"
    
    total = n_triviaqa + n_hotpotqa
    
    with open(outfile, "w") as f:
        # TriviaQA
        if n_triviaqa > 0:
            print(f"Loading TriviaQA (N={n_triviaqa})...")
            tqa = load_dataset("trivia_qa", "rc", split=f"validation[:{n_triviaqa}]")
            for i, item in enumerate(tqa):
                print(f"[{i+1}/{total}] TriviaQA: {item['question'][:50]}")
                t0 = time.time()
                answers = item["answer"]["aliases"] + [item["answer"]["value"]]
                try:
                    result = run_single_trajectory(
                        model, sae, item["question"], answers, sae_layer, n_steps, device
                    )
                    result["source"] = "triviaqa"
                    result["id"] = item.get("question_id", str(i))
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    results.append(result)
                    print(f"   Correct: {result['final_correct']} | {time.time()-t0:.1f}s")
                except Exception as e:
                    print(f"   ERROR: {e}")
        
        # HotpotQA
        if n_hotpotqa > 0:
            print(f"Loading HotpotQA (N={n_hotpotqa})...")
            hpqa = load_dataset("hotpot_qa", "fullwiki", split=f"validation[:{n_hotpotqa}]")
            for j, item in enumerate(hpqa):
                print(f"[{n_triviaqa+j+1}/{total}] HotpotQA: {item['question'][:50]}")
                t0 = time.time()
                answers = [item["answer"]]
                try:
                    result = run_single_trajectory(
                        model, sae, item["question"], answers, sae_layer, n_steps, device
                    )
                    result["source"] = "hotpotqa"
                    result["id"] = item.get("id", str(j))
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    results.append(result)
                    print(f"   Correct: {result['final_correct']} | {time.time()-t0:.1f}s")
                except Exception as e:
                    print(f"   ERROR: {e}")
    
    print(f"\nDone. Results saved to: {outfile}")
    print(f"N trajectories: {len(results)}")
    if results:
        print(f"Accuracy: {np.mean([r['final_correct'] for r in results]):.2%}")
    return str(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScope v2 Batch Trajectory Runner")
    parser.add_argument("--model", default="gemma-2-2b-it", help="Model name (e.g. gemma-2-2b-it)")
    parser.add_argument("--n-triviaqa", type=int, default=200, help="Number of TriviaQA samples")
    parser.add_argument("--n-hotpotqa", type=int, default=100, help="Number of HotpotQA samples")
    parser.add_argument("--sae-layer", type=int, default=12, help="SAE layer")
    parser.add_argument("--output-dir", default="./results/", help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()
    
    run_batch_experiment(
        model_name=args.model,
        n_triviaqa=args.n_triviaqa,
        n_hotpotqa=args.n_hotpotqa,
        sae_layer=args.sae_layer,
        output_dir=args.output_dir,
        device=args.device,
    )
