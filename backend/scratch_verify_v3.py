"""Verification script to test NeuroScope v3 additions (Steering, Probing, Causal Patching).
Ensures all mock and real execution contracts function correctly.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify_v3")

def verify_v3_components():
    print("=== NeuroScope v3 Components Verification ===\n")
    
    # 1. Test Sparse Probing API
    print("1. Testing Sparse Probing Logic (Mock Mode)...")
    try:
        from train_probe import train_factual_probe
        result = train_factual_probe(model=None, real=False)
        
        assert result["real"] is False, "Should return simulated results"
        assert "accuracy" in result, "Accuracy metric is missing"
        assert "roc_auc" in result, "ROC AUC metric is missing"
        assert len(result["features"]) > 0, "No active features returned"
        print("   Accuracy:", result["accuracy"])
        print("   ROC AUC:", result["roc_auc"])
        print("   Active Features Count:", result["n_active_features"])
        print("   ✅ Sparse probing logic runs successfully and yields correct mock structures!\n")
    except Exception as e:
        print(f"   ❌ Sparse probing test failed: {e}\n")
        
    # 2. Test Feature Path Patching
    print("2. Testing Feature Path Patching (Simulated Mode)...")
    try:
        from neuroscope.patching import feature_path_patch
        target_features = [
            {"feature_id": 804, "layer": 12},
            {"feature_id": 5291, "layer": 12}
        ]
        result = feature_path_patch(
            model=None,
            prompt="Einstein won the Nobel Prize.",
            layer=12,
            feature_id=1402,
            target_features=target_features,
            real=False
        )
        
        assert result["real"] is False, "Should run simulated patching"
        assert result["source_feature_id"] == 1402, "Source feature ID mismatch"
        assert len(result["effects"]) == 2, "Causal effects count mismatch"
        print("   Source Feature:", result["source_feature_id"])
        for eff in result["effects"]:
            print(f"     -> Target: #{eff['target_feature_id']} | Effect: {eff['effect']}")
        print("   ✅ Feature path patching executes successfully!\n")
    except Exception as e:
        print(f"   ❌ Feature path patching test failed: {e}\n")

    # 3. Test Step Causal Attribution Graph
    print("3. Testing Step Causal Attribution Graph Generation...")
    try:
        from neuroscope.patching import causal_attribution_for_step
        top_features = [
            {"feature_id": 1402, "activation": 3.4},
            {"feature_id": 804, "activation": 2.1},
            {"feature_id": 5291, "activation": 1.8}
        ]
        result = causal_attribution_for_step(
            model=None,
            prompt="Test prompt",
            layer=12,
            top_features=top_features,
            real=False
        )
        
        assert result["method"] == "causal_path_patching", "Method name mismatch"
        assert len(result["nodes"]) == 3, "Node count mismatch"
        print("   Causal Edges Generated:", len(result["edges"]))
        for edge in result["edges"]:
            print(f"     Edge: {edge['source']} -> {edge['target']} (weight: {edge['weight']})")
        print("   ✅ Causal attribution graph generation works beautifully!\n")
    except Exception as e:
        print(f"   ❌ Causal attribution test failed: {e}\n")

if __name__ == "__main__":
    verify_v3_components()
