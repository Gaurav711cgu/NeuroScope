"""Local activation artifact storage using local disk.

Saves .npz files locally under backend/data/activations.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Local directory: backend/data/activations
LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "activations"


def save_step_activations(
    run_id: str,
    step_n: int,
    captured: dict,
) -> str:
    """Serialize float16 .npz to local disk. Returns local:// path."""
    local_path = LOCAL_DATA_DIR / run_id / f"step_{step_n}.npz"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        local_path,
        **{k: v.astype(np.float16) for k, v in captured.items()},
    )
    logger.info("Saved activations to local disk → %s", local_path)
    return f"local://{run_id}/step_{step_n}.npz"


def load_step_activations(path: str) -> dict:
    """Read .npz from local disk."""
    clean_path = path.replace("local://", "")
    local_path = LOCAL_DATA_DIR / clean_path
    
    if not local_path.exists():
        # Handle fallback for raw filenames if they don't have local:// prefix
        local_path = LOCAL_DATA_DIR / path
        if not local_path.exists():
            raise FileNotFoundError(f"Activations not found locally: {path}")
            
    npz = np.load(local_path, allow_pickle=False)
    return {k: npz[k].astype(np.float16) for k in npz.files}
