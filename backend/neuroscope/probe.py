"""Linear probing and survival hazard modeling on step residual activations.
"""
from __future__ import annotations

import logging
import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .db import get_run
from .storage import load_step_activations
from .ml_ops import semantic_clip

logger = logging.getLogger("neuroscope.probe")

def eval_probe_loss(sae_latents: np.ndarray, binary_labels: np.ndarray) -> float:
    """1D logistic probe trained with Newton-Raphson on each latent (Gao et al. §4.2).
    
    Records best cross-entropy across all latents.
    """
    from sklearn.metrics import log_loss
    best_ce = float('inf')
    n_latents = sae_latents.shape[1]
    
    lr = LogisticRegression(solver='newton-cg', max_iter=100)
    
    for i in range(n_latents):
        X_1d = sae_latents[:, i:i+1]
        if np.all(X_1d == 0):
            continue
            
        try:
            lr.fit(X_1d, binary_labels)
            probs = lr.predict_proba(X_1d)
            ce = log_loss(binary_labels, probs)
            if ce < best_ce:
                best_ce = ce
        except Exception:
            pass
            
    return float(best_ce) if best_ce != float('inf') else None


def cox_partial_log_likelihood(beta, X, T, E, alpha_l2=0.1):
    """Compute L2-penalized negative log partial likelihood for Cox Proportional Hazards model."""
    n_samples = len(T)
    if n_samples == 0:
        return 0.0
    
    # Calculate dot products
    scores = np.dot(X, beta)
    
    neg_log_lik = 0.0
    for i in range(n_samples):
        if E[i] == 1:
            # Risk set: all items with time >= T[i]
            risk_indices = np.where(T >= T[i])[0]
            if len(risk_indices) > 0:
                # Log-sum-exp trick for numerical stability
                max_score = np.max(scores[risk_indices])
                sum_exp = np.sum(np.exp(scores[risk_indices] - max_score))
                neg_log_lik -= (scores[i] - (max_score + np.log(sum_exp)))
                
    # Add L2 penalty
    neg_log_lik += alpha_l2 * np.sum(beta ** 2)
    return neg_log_lik


def fit_cox_survival_model(X, T, E, alpha_l2=0.5) -> tuple[np.ndarray, bool]:
    """Fit a Cox Proportional Hazards model using L2-regularized MLE."""
    n_features = X.shape[1]
    initial_beta = np.zeros(n_features)
    res = opt.minimize(
        cox_partial_log_likelihood,
        initial_beta,
        args=(X, T, E, alpha_l2),
        method="L-BFGS-B"
    )
    return res.x, res.success


def calculate_kaplan_meier(T, E) -> tuple[list[int], list[float]]:
    """Calculate Kaplan-Meier survival curve probabilities for trajectory steps."""
    unique_times = np.sort(np.unique(T))
    survival_prob = 1.0
    times = [0]
    probs = [1.0]
    
    for t in unique_times:
        # At risk: T_j >= t
        n_at_risk = np.sum(T >= t)
        # Deaths/Events: T_j == t and E_j == 1
        n_events = np.sum((T == t) & (E == 1))
        
        if n_at_risk > 0:
            survival_prob *= (1.0 - (n_events / n_at_risk))
            
        times.append(int(t))
        probs.append(float(survival_prob))
        
    return times, probs


async def train_hallucination_probe(
    run_ids: list[str],
    layer: int = 12,
) -> dict:
    """Train a linear probe and a Cox survival hazard model on intermediate step activations.

    Linear Probe: Predicts final outcome correctness (0 = factual, 1 = hallucinated).
    Survival Model: Predicts the step-by-step hazard rate of hallucination decay.
    """
    X, y = [], []
    T, E = [], []  # For survival analysis

    for run_id in run_ids:
        run = await get_run(run_id)
        if not run or not run.get("steps"):
            continue
        
        is_correct = run.get("correct")
        if is_correct is None:
            continue
        label = 0 if is_correct else 1

        # Locate the step at which hallucination first occurred (threshold > 0.5)
        hallucination_step = None
        for step in run["steps"]:
            h_score = step.get("hallucination", {}).get("composite", 0.0)
            if h_score >= 0.5:
                hallucination_step = step["step_n"]
                break

        # Time-to-event (T) and event occurred (E)
        if hallucination_step is not None:
            event_time = hallucination_step
            event_occurred = 1
        else:
            event_time = len(run["steps"])  # Right-censored at the end
            event_occurred = 0

        # Process step-by-step activations
        steps_to_train = run["steps"][:-1] if len(run["steps"]) > 1 else run["steps"]
        for step in steps_to_train:
            act_path = step["activation_path"]
            if not act_path:
                continue
            try:
                acts = load_step_activations(act_path)
                key = f"blocks.{layer}.hook_resid_post"
                if key in acts:
                    # Retrieve last-token activations
                    last_token_resid = acts[key][0, -1, :].astype(np.float32)
                    # Apply defensive MLOps semantic clipping
                    clipped_resid = semantic_clip(last_token_resid)
                    
                    X.append(clipped_resid)
                    y.append(label)
                    T.append(event_time)
                    E.append(event_occurred)
            except Exception as e:
                logger.error("Failed to load activations for step in run %s: %s", run_id, e)

    if len(X) < 5:
        logger.warning("Not enough samples (%d) to train a probe (minimum 5 required).", len(X))
        return {
            "layer": layer,
            "n_samples": len(X),
            "cv_auc_mean": None,
            "cv_auc_std": None,
            "probe_accuracy": None,
            "top_predictive_dims": [],
            "features": [],
            "survival_analysis": {
                "times": [],
                "survival_probabilities": []
            },
            "real": False,
            "error": f"Insufficient activation samples ({len(X)}/5 required) to train probe."
        }


    X = np.array(X)
    y = np.array(y)
    T = np.array(T)
    E = np.array(E)

    # 1. Fit Logistic Regression Classifier
    if len(np.unique(y)) < 2:
        return {
            "real": False,
            "error": "single_class"
        }

    probe = LogisticRegression(max_iter=1000, C=0.1, solver="liblinear", random_state=42)
    n_splits = min(5, len(X))
    cv_scores = cross_val_score(probe, X, y, cv=n_splits, scoring="roc_auc")
    probe.fit(X, y)
    probe_accuracy = float((probe.predict(X) == y).mean())
    cv_auc_mean = float(np.mean(cv_scores))
    cv_auc_std = float(np.std(cv_scores))
    coefs = probe.coef_[0]

    # 2. Fit Cox Proportional Hazards Model
    cox_betas = np.zeros(X.shape[1])
    cox_fit_success = False
    log_rank_p_value = 1.0
    n_events = int(np.sum(E))

    if len(np.unique(E)) >= 2:
        try:
            cox_betas, cox_fit_success = fit_cox_survival_model(X, T, E, alpha_l2=0.5)
            
            import scipy.stats as stats
            mask_h = (y == 1)
            mask_f = (y == 0)
            
            if np.sum(mask_h) > 0 and np.sum(mask_f) > 0:
                x_data = stats.CensoredData(
                    uncensored=T[mask_h & (E == 1)],
                    right=T[mask_h & (E == 0)]
                )
                y_data = stats.CensoredData(
                    uncensored=T[mask_f & (E == 1)],
                    right=T[mask_f & (E == 0)]
                )
                res_lr = stats.logrank(x=x_data, y=y_data)
                log_rank_p_value = float(res_lr.pvalue)
                
        except Exception as e:
            logger.error("Failed to fit Cox proportional hazards model: %s", e)

    # Calculate Kaplan-Meier survival curves
    km_times, km_probs = calculate_kaplan_meier(T, E)

    # Sort dimensions by combination of Logistic Regression and Cox hazard weights
    abs_coefs = np.abs(coefs)
    top_dims = np.argsort(abs_coefs)[-20:][::-1].tolist()

    features_weight = []
    for dim in top_dims:
        w = float(coefs[dim])
        beta_cox = float(cox_betas[dim])
        hazard_ratio = float(np.exp(beta_cox))
        features_weight.append({
            "dimension": dim,
            "weight": round(w, 4),
            "direction": "hallucination" if w > 0 else "factual",
            "hazard_ratio": round(hazard_ratio, 4)
        })

    return {
        "layer": layer,
        "n_samples": len(X),
        "cv_auc_mean": round(cv_auc_mean, 4),
        "cv_auc_std": round(cv_auc_std, 4),
        "probe_accuracy": round(probe_accuracy, 4),
        "top_predictive_dims": top_dims,
        "features": features_weight,
        "survival_analysis": {
            "times": km_times,
            "survival_probabilities": km_probs,
            "cox_fit_success": cox_fit_success,
            "log_rank_p_value": log_rank_p_value,
            "n_events": n_events
        },
        "real": True
    }
