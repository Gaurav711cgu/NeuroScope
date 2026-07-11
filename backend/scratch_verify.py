"""Verification script for NeuroScope v3.

Validates PostgreSQL schema, local activation saving/loading,
mock path-patching guards, and module loading.
"""
from __future__ import annotations

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

# Add ROOT_DIR to Python path
sys.path.append(str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify")

async def verify_all():
    print("=== NeuroScope v3 Configuration Verification ===\n")
    
    # 1. Verify PostgreSQL setup
    print("1. Testing PostgreSQL Connection & Schema...")
    try:
        from neuroscope import db
        await db.init_db()
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            # Query public tables
            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )
            table_names = [t["table_name"] for t in tables]
            print(f"   Found tables: {table_names}")
            
            required_tables = ["runs", "steps", "step_features", "patch_results", "queries", "experiments", "feature_labels", "attribution_graphs", "findings_runs", "outbox"]
            missing_tables = [t for t in required_tables if t not in table_names]
            assert not missing_tables, f"Missing tables: {missing_tables}"
            
        print("   ✅ PostgreSQL database connection & tables verified!\n")
    except Exception as e:
        print(f"   ❌ PostgreSQL test failed: {e}\n")
        
    # 2. Verify Local Activation Storage
    print("2. Testing Local Activation Storage...")
    try:
        from neuroscope.storage import save_step_activations, load_step_activations
        import numpy as np
        
        run_id = "test-verify-run"
        step_n = 99
        captured = {"blocks.7.hook_resid_post": np.array([[[1.0, 2.0, 3.0]]], dtype=np.float16)}
        
        path = save_step_activations(run_id, step_n, captured)
        print(f"   Saved path: {path}")
        assert path == f"local://{run_id}/step_{step_n}.npz", f"Unexpected path format: {path}"
        
        loaded = load_step_activations(path)
        assert np.allclose(loaded["blocks.7.hook_resid_post"], captured["blocks.7.hook_resid_post"]), "Loaded activations do not match original"
        print("   ✅ Local activation storage save and load verified successfully!\n")
        
        # Cleanup
        local_file = Path(ROOT_DIR) / "data" / "activations" / run_id / f"step_{step_n}.npz"
        if local_file.exists():
            local_file.unlink()
            local_file.parent.rmdir()
    except Exception as e:
        print(f"   ❌ Local activation storage test failed: {e}\n")
        
    # 3. Verify Causal Patching Guard
    print("3. Testing Causal Patching Mock Guard...")
    try:
        from neuroscope.patching import feature_path_patch
        try:
            # Running with real=False should raise ValueError
            feature_path_patch(None, "test prompt", 12, 1402, [], real=False)
            print("   ❌ Causal patching mock mode was NOT blocked!")
        except ValueError as ve:
            print(f"   ✅ Causal patching mock mode successfully blocked with ValueError: '{ve}'\n")
    except Exception as e:
        print(f"   ❌ Causal patching guard test failed: {e}\n")
        
    # 4. Verify Probing & Steering Modules
    print("4. Testing Probing & Steering Module Loading...")
    try:
        from neuroscope.probe import train_hallucination_probe
        from neuroscope.steering import steer_and_regenerate
        print("   ✅ Probing and Steering modules loaded successfully!\n")
    except Exception as e:
        print(f"   ❌ Probing or Steering module loading failed: {e}\n")

if __name__ == "__main__":
    asyncio.run(verify_all())
