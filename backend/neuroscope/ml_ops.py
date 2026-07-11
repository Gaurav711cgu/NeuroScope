"""Defensive MLOps Guardrails for NeuroScope: semantic feature clipping and PSI drift monitoring."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger("neuroscope.ml_ops")

# Default boundary clips representing 1st and 99th percentiles of typical residual feature activations
DEFAULT_MIN_CLIP = 0.0
DEFAULT_MAX_CLIP = 12.0


def semantic_clip(activations: np.ndarray, min_val: float = DEFAULT_MIN_CLIP, max_val: float = DEFAULT_MAX_CLIP) -> np.ndarray:
    """Cap incoming activation features to historical bounds to prevent wild inference extrapolation."""
    return np.clip(activations, min_val, max_val)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) between two numeric distributions.
    
    Interpretations:
    - PSI < 0.10: No significant distribution drift.
    - 0.10 <= PSI < 0.25: Moderate distribution drift detected.
    - PSI >= 0.25: Significant distribution drift (action/alert required).
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges using percentiles of the reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)
    
    if len(bins) < 2:
        return 0.0  # Constant distribution

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Add epsilon smoothing to prevent divide-by-zero or undefined log math
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    # Re-normalize proportions
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    if psi_value >= 0.25:
        logger.warning(
            "[MLOps Alert] Significant feature representation drift detected! PSI = %.4f (threshold = 0.25)",
            psi_value
        )
    elif psi_value >= 0.10:
        logger.info(
            "[MLOps Notice] Moderate feature representation drift detected. PSI = %.4f",
            psi_value
        )
    else:
        logger.debug("Feature distribution stable. PSI = %.4f", psi_value)

    return float(psi_value)
