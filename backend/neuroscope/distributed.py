"""
Multi-GPU Distributed Activation Ingestion Pipeline (PyTorch DDP / Ray)
======================================================================
Enables high-throughput multi-GPU activation extraction & SAE encoding
across distributed compute clusters (OpenAI / DeepMind style scaling).
"""
from __future__ import annotations

import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger("neuroscope.distributed")


def init_distributed_environment() -> dict:
    """Initialize PyTorch DDP process group for multi-GPU activation extractions."""
    if not dist.is_available():
        return {"distributed": False, "rank": 0, "world_size": 1, "device": "cpu"}
        
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        logger.info("Initialized PyTorch DDP rank %d/%d (device=cuda:%d)", rank, world_size, local_rank)
        
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return {
        "distributed": world_size > 1,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device
    }
