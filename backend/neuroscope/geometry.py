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
    
    # Gram matrix: pairwise cosine similarities [n_features, n_features]
    gram = torch.matmul(W_norm, W_norm.T)
    
    n_features = W_dec.shape[0]
    diag = torch.eye(n_features, device=W_dec.device)
    off_diag = gram - diag
    
    frobenius_norm = float(torch.norm(off_diag, p="fro").item())
    max_cosine_sim = float(torch.max(torch.abs(off_diag)).item())
    mean_abs_cosine_sim = float(torch.mean(torch.abs(off_diag)).item())
    
    return {
        "n_features": n_features,
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
