"""Global in-memory singletons and state management for NeuroScope."""
from __future__ import annotations

import logging
import os
import threading
import torch

logger = logging.getLogger("neuroscope.core.state")

# Thread safety lock for model/SAE loading
state_lock = threading.Lock()

# Singletons (initialized lazily)
model = None
sae_cache = {}

MODEL_NAME = os.environ.get("NEUROSCOPE_MODEL", "google/gemma-2-2b-it")
SAE_RELEASE = os.environ.get("NEUROSCOPE_SAE_RELEASE", "gemma-scope-2b-pt-res")
DEFAULT_SAE_LAYER = int(os.environ.get("NEUROSCOPE_SAE_LAYER", "12"))

is_gemma = "gemma" in MODEL_NAME.lower()
CAPTURE_LAYERS = [6, 12, 18, 24] if is_gemma else [3, 6, 7, 9, 10]

# Configure PyTorch thread limits
torch.set_num_threads(int(os.environ.get("NEUROSCOPE_THREADS", "4")))
