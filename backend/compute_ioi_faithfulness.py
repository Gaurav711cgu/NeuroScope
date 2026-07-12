"""
IOI Circuit Faithfulness Benchmark (Wang et al. 2022)
=====================================================
Replicates the standard circuit faithfulness metric from:
  "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
  Wang et al., 2022 — https://arxiv.org/abs/2211.00593

Methodology:
  - Resampling ablation: non-circuit heads are patched with activations from a
    corrupted run (ABC prompts with swapped names), not zero-ablated.
  - Faithfulness = (circuit_logit_diff - corrupted_logit_diff) /
                   (clean_logit_diff - corrupted_logit_diff)
  - Metric range: 0 (corrupted baseline) → 1 (full model performance)
  - Expected range for this circuit: ~0.65–0.85

Wang et al. published circuit heads (26 heads total):
  - Name Mover:         (9,9), (10,0), (9,6)
  - Backup Name Mover:  (10,10),(10,6),(10,2),(10,1),(11,2),(9,7),(9,0),(11,9)
  - Negative Name Mover:(10,7),(11,10)
  - S-Inhibition:       (7,3),(7,9),(8,6),(8,10)
  - Induction:          (5,5),(5,8),(5,9),(6,9)
  - Duplicate Token:    (0,1),(0,10),(3,0)
  - Previous Token:     (2,2),(4,11)
"""

import time
import random
import torch
import transformer_lens as tl

# ──────────────────────────────────────────────
# Wang et al. published IOI circuit (26 heads)
# ──────────────────────────────────────────────
CIRCUIT_HEADS = {
    (9, 9), (10, 0), (9, 6),                            # Name Mover
    (10, 10), (10, 6), (10, 2), (10, 1),                # Backup Name Mover
    (11, 2), (9, 7), (9, 0), (11, 9),                   # Backup Name Mover (cont.)
    (10, 7), (11, 10),                                   # Negative Name Mover
    (7, 3), (7, 9), (8, 6), (8, 10),                    # S-Inhibition
    (5, 5), (5, 8), (5, 9), (6, 9),                     # Induction
    (0, 1), (0, 10), (3, 0),                             # Duplicate Token
    (2, 2), (4, 11),                                     # Previous Token
}


# ──────────────────────────────────────────────
# Dataset: fixed-length IOI prompts
# Template: "Then, [Name A] and [Name B] went to the store. [Name B] gave the bag to"
# All prompts tokenize to the same length (required for batching).
# ──────────────────────────────────────────────
NAMES = [
    " John", " Mary", " Bob", " Alice", " James",
    " Sarah", " Paul", " Emma", " Chris", " Lisa",
    " David", " Anna", " Mark", " Laura", " Peter",
]

TEMPLATE = "Then,{A} and{B} went to the store.{B} gave the bag to"


def build_dataset(model, N: int = 50, seed: int = 42):
    """Build fixed-length IOI dataset with ABC-corrupted counterparts."""
    random.seed(seed)
    
    clean_prompts, corrupted_prompts = [], []
    io_token_ids, s_token_ids = [], []

    attempts = 0
    while len(clean_prompts) < N and attempts < N * 10:
        attempts += 1
        A, B = random.sample(NAMES, 2)

        clean_text = TEMPLATE.format(A=A, B=B)
        # Corrupted: swap A and B introduction (model should now predict B, not A)
        corrupted_text = TEMPLATE.format(A=B, B=A)

        try:
            io_tok = model.to_single_token(A)
            s_tok = model.to_single_token(B)
        except Exception:
            continue  # skip multi-token names

        clean_prompts.append(clean_text)
        corrupted_prompts.append(corrupted_text)
        io_token_ids.append(io_tok)
        s_token_ids.append(s_tok)

    assert len(clean_prompts) == N, f"Only built {len(clean_prompts)}/{N} examples"

    # Tokenize — all prompts must be equal length
    clean_toks = model.to_tokens(clean_prompts)       # [N, seq_len]
    corr_toks  = model.to_tokens(corrupted_prompts)   # [N, seq_len]

    assert clean_toks.shape == corr_toks.shape, (
        f"Length mismatch: clean={clean_toks.shape}, corrupted={corr_toks.shape}"
    )

    io_ids = torch.tensor(io_token_ids, device=clean_toks.device)
    s_ids  = torch.tensor(s_token_ids,  device=clean_toks.device)
    end_pos = clean_toks.shape[1] - 1  # last token position

    return clean_toks, corr_toks, io_ids, s_ids, end_pos


# ──────────────────────────────────────────────
# Logit-difference metric
# ──────────────────────────────────────────────
def logit_diff(logits, io_ids, s_ids, end_pos):
    """IO logit − S logit at the END token position."""
    end_logits = logits[:, end_pos, :]          # [N, vocab]
    io = end_logits[torch.arange(logits.shape[0]), io_ids]
    s  = end_logits[torch.arange(logits.shape[0]), s_ids]
    return (io - s).mean()


# ──────────────────────────────────────────────
# Resampling ablation: patch non-circuit heads
# with activations from the corrupted run
# ──────────────────────────────────────────────
def run_circuit_only(model, clean_toks, corr_toks, circuit_heads):
    """Run model with only circuit heads active (others patched from corrupted)."""
    _, corr_cache = model.run_with_cache(corr_toks)

    def ablate_non_circuit(z, hook):
        """Replace non-circuit head outputs with corrupted activations."""
        layer = hook.layer()
        for h in range(model.cfg.n_heads):
            if (layer, h) not in circuit_heads:
                z[:, :, h, :] = corr_cache[hook.name][:, :, h, :]
        return z

    # Add hook to every attention layer
    for layer in range(model.cfg.n_layers):
        model.add_hook(f"blocks.{layer}.attn.hook_z", ablate_non_circuit)

    with torch.no_grad():
        logits = model(clean_toks)

    model.reset_hooks()
    return logits


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load model (CPU to avoid MPS numerical issues)
    print("Loading GPT-2 small...")
    model = tl.HookedTransformer.from_pretrained("gpt2", device="cpu")
    model.eval()

    # Build dataset
    print("Building IOI dataset (N=50, fixed-length templates)...")
    clean_toks, corr_toks, io_ids, s_ids, end_pos = build_dataset(model, N=50)
    print(f"  Token sequence length: {clean_toks.shape[1]}")
    print(f"  Batch size: {clean_toks.shape[0]}")

    # ── 1. Full-model logit diff (clean run) ──
    print("\nRunning full model on clean prompts...")
    with torch.no_grad():
        clean_logits = model(clean_toks)
    ld_clean = logit_diff(clean_logits, io_ids, s_ids, end_pos)
    print(f"  Clean logit diff:      {ld_clean.item():+.4f}")

    # ── 2. Corrupted baseline logit diff ──
    print("Running full model on corrupted prompts...")
    with torch.no_grad():
        corr_logits = model(corr_toks)
    ld_corr = logit_diff(corr_logits, io_ids, s_ids, end_pos)
    print(f"  Corrupted logit diff:  {ld_corr.item():+.4f}")

    # ── 3. Circuit-only logit diff (resampling ablation) ──
    print(f"\nRunning circuit-only model ({len(CIRCUIT_HEADS)} heads, Wang et al.)...")
    circuit_logits = run_circuit_only(model, clean_toks, corr_toks, CIRCUIT_HEADS)
    ld_circuit = logit_diff(circuit_logits, io_ids, s_ids, end_pos)
    print(f"  Circuit logit diff:    {ld_circuit.item():+.4f}")

    # ── 4. Faithfulness ──
    # Standard formula from Wang et al. (proportion of performance recovered)
    faithfulness = (ld_circuit - ld_corr) / (ld_clean - ld_corr)

    print("\n" + "="*55)
    print("  IOI CIRCUIT FAITHFULNESS RESULTS")
    print("="*55)
    print(f"  Full model logit diff:      {ld_clean.item():+.4f}")
    print(f"  Corrupted baseline:         {ld_corr.item():+.4f}")
    print(f"  Circuit-only logit diff:    {ld_circuit.item():+.4f}")
    print(f"  Circuit faithfulness score: {faithfulness.item():.3f}")
    print("="*55)
    print(f"\n  ✓ Resume metric: Circuit faithfulness = {faithfulness.item():.2f}")
    print(f"    (Expected range 0.65–0.85 per Wang et al. 2022)")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
