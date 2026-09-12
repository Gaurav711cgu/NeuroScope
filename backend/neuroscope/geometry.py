"""
OpenAI-Style Feature Geometry & Sparse Dictionary Orthogonality Metrics
========================================================================
Analyzes SAE feature space geometry, superposition density, and dictionary orthogonality:

1. Feature Orthogonality: ||W_dec^T W_dec - I||_F
2. Cosine Similarity Distribution: Pairwise feature direction alignment
3. Feature Superposition Dimensionality: Active feature capacity per residual dimension
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_dictionary_orthogonality(W_dec: torch.Tensor) -> dict:
    """Compute Frobenius norm of off-diagonal elements in W_dec^T W_dec matrix.
    
    W_dec shape: [n_features, d_model]
    Ideal orthogonal dictionary: off-diagonal elements near 0.0.
    """
    # Normalize rows to unit vectors
    W_norm = F.normalize(W_dec.float(), p=2, dim=-1)
    
    # Subsample to avoid OOM for large dictionaries (e.g., 16K+ features)
    n_features = W_dec.shape[0]
    if n_features > 2048:
        indices = torch.randperm(n_features, device=W_dec.device)[:2048]
        W_norm = W_norm[indices]
        n_features = 2048
        
    # Gram matrix: pairwise cosine similarities [n_features, n_features]
    gram = torch.matmul(W_norm, W_norm.T)
    
    diag = torch.eye(n_features, device=W_dec.device)
    off_diag = gram - diag
    
    frobenius_norm = float(torch.norm(off_diag, p="fro").item())
    max_cosine_sim = float(torch.max(torch.abs(off_diag)).item())
    mean_abs_cosine_sim = float(torch.mean(torch.abs(off_diag)).item())
    
    return {
        "n_features_sampled": n_features,
        "d_model": W_dec.shape[1],
        "frobenius_orthogonality_norm": round(frobenius_norm, 4),
        "max_pairwise_cosine_sim": round(max_cosine_sim, 4),
        "mean_abs_cosine_sim": round(mean_abs_cosine_sim, 6)
    }


def compute_feature_cosine_similarity(W_dec: torch.Tensor, feature_id_a: int, feature_id_b: int) -> float:
    """Compute exact cosine similarity between two SAE feature decoder directions."""
    v_a = W_dec[feature_id_a].float()
    v_b = W_dec[feature_id_b].float()
    cos_sim = F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
    return float(cos_sim)


def compute_sae_quality(model, sae, tokens: torch.Tensor, mask_special: bool = True) -> dict:
    """Compute SAE quality metrics from Gao et al. (2024) and Lieberum et al. (2024).
    
    Returns: {l0, delta_lm_loss, fvu}
    delta_lm_loss = ce_with_sae - ce_baseline
    fvu = mean_reconstruction_mse / mean_prediction_mse
    """
    hook_point = sae.cfg.hook_name
    
    with torch.no_grad():
        logits_baseline, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        shift_logits = logits_baseline[:, :-1, :].contiguous().view(-1, model.cfg.d_vocab)
        shift_labels = tokens[:, 1:].contiguous().view(-1)
        ce_baseline = loss_fn(shift_logits, shift_labels).view(tokens.shape[0], -1)
        
        acts = cache[hook_point]
        feature_acts = sae.encode(acts)
        reconstructed = sae.decode(feature_acts)
        
        l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
        
        mse = ((acts - reconstructed) ** 2).mean()
        var = ((acts - acts.mean(dim=0, keepdim=True)) ** 2).mean()
        fvu = (mse / var).item() if var.item() > 0 else 0.0

    def patch_hook(act, hook):
        return sae.decode(sae.encode(act))

    with torch.no_grad():
        logits_patched = model.run_with_hooks(
            tokens, 
            return_type="logits",
            fwd_hooks=[(hook_point, patch_hook)]
        )
        
        shift_logits_patched = logits_patched[:, :-1, :].contiguous().view(-1, model.cfg.d_vocab)
        ce_patched = loss_fn(shift_logits_patched, shift_labels).view(tokens.shape[0], -1)
        
        if mask_special and hasattr(model, 'tokenizer') and model.tokenizer is not None:
            bos = model.tokenizer.bos_token_id
            eos = model.tokenizer.eos_token_id
            pad = model.tokenizer.pad_token_id
            mask = (shift_labels != bos) & (shift_labels != eos) & (shift_labels != pad)
            mask = mask.view(tokens.shape[0], -1)
            
            ce_baseline_mean = ce_baseline[mask].mean().item()
            ce_patched_mean = ce_patched[mask].mean().item()
        else:
            ce_baseline_mean = ce_baseline.mean().item()
            ce_patched_mean = ce_patched.mean().item()
            
        delta_lm_loss = ce_patched_mean - ce_baseline_mean

    return {
        "l0": round(l0, 2),
        "delta_lm_loss": round(delta_lm_loss, 4),
        "fvu": round(fvu, 4)
    }


def compute_ablation_sparsity(model, sae, tokens: torch.Tensor, feature_ids: list[int], T: int = 16) -> dict:
    """Compute ablation sparsity metric from Gao et al. (2024) §4.5.
    
    (L1/L2)^2 / (V*T) of downstream logit changes when each latent is ablated.
    Subtracts median logit difference per token.
    """
    hook_point = sae.cfg.hook_name
    
    with torch.no_grad():
        logits_baseline = model(tokens, return_type="logits")
    
    results = {}
    for fid in feature_ids:
        def ablate_hook(act, hook):
            # Compute activation and direction to subtract
            feature_acts = sae.encode(act)
            activation = feature_acts[..., fid].unsqueeze(-1)  # [batch, seq, 1]
            direction = sae.W_dec[fid].view(1, 1, -1)          # [1, 1, d_model]
            return act - (activation * direction)

        with torch.no_grad():
            logits_ablated = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(hook_point, ablate_hook)]
            )
            
        diff = logits_ablated - logits_baseline
        median_diff = diff.median(dim=-1, keepdim=True).values
        diff = diff - median_diff
        
        l1 = diff.abs().sum(dim=-1)
        l2 = (diff ** 2).sum(dim=-1).sqrt()
        
        sparsity = ((l1 / (l2 + 1e-8)) ** 2) / model.cfg.d_vocab
        results[fid] = float(sparsity.mean().item())
        
    return results
