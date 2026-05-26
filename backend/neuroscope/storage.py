"""Activation artifact storage using Google Firebase Storage with Local Disk fallback.

If Firebase Storage is not configured (or fails), saves .npz files locally
in backend/data/activations to keep the app 100% free and zero-setup.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Local fallback directory: backend/data/activations
LOCAL_DATA_DIR = Path(__file__).parent.parent / "data" / "activations"


def save_step_activations(
    run_id: str,
    step_n: int,
    captured: dict,
) -> str:
    """Serialize float16 .npz. Returns the storage path (either remote or local://)."""
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        **{k: v.astype(np.float16) for k, v in captured.items()},
    )
    buf.seek(0)
    path = f"runs/{run_id}/step_{step_n}.npz"

    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if bucket_name and bucket_name != "your-project-id.appspot.com":
        try:
            from neuroscope.firebase_init import get_bucket
            bucket = get_bucket()
            blob = bucket.blob(path)
            blob.upload_from_string(
                buf.getvalue(),
                content_type="application/octet-stream"
            )
            logger.info("Saved activations to Firebase Storage → %s", path)
            return path
        except Exception as e:
            logger.warning("Firebase Storage upload failed; falling back to local disk: %s", e)

    # Local Fallback
    local_path = LOCAL_DATA_DIR / "runs" / run_id / f"step_{step_n}.npz"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(buf.getvalue())
    logger.info("Saved activations to local disk → %s", local_path)
    return f"local://{path}"


def load_step_activations(path: str) -> dict:
    """Download .npz from Firebase Storage or read from local disk."""
    if path.startswith("local://"):
        clean_path = path.replace("local://", "")
        local_path = LOCAL_DATA_DIR / clean_path
        npz = np.load(local_path, allow_pickle=False)
        return {k: npz[k].astype(np.float16) for k in npz.files}

    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET")
    if bucket_name and bucket_name != "your-project-id.appspot.com":
        try:
            from neuroscope.firebase_init import get_bucket
            bucket = get_bucket()
            blob = bucket.blob(path)
            data = blob.download_as_bytes()
            buf = io.BytesIO(data)
            npz = np.load(buf, allow_pickle=False)
            return {k: npz[k].astype(np.float16) for k in npz.files}
        except Exception as e:
            logger.error("Failed to download from Firebase Storage, checking local disk: %s", e)

    # Fallback to checking local path directly as a last resort
    local_path = LOCAL_DATA_DIR / path
    if local_path.exists():
        npz = np.load(local_path, allow_pickle=False)
        return {k: npz[k].astype(np.float16) for k in npz.files}

    raise FileNotFoundError(f"Activations not found in Firebase Storage or local path: {path}")
