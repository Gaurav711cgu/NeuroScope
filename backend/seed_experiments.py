"""Seed pre-built experiments into Supabase. Run once after install.

    cd backend && python seed_experiments.py

Each experiment is executed end-to-end with Gemma-2-2b-it + GemmaScope,
including feature timelines, patch matrix, and a finding written by Claude.

Updated for v2:
  - Supabase replaces MongoDB (synchronous supabase-py client)
  - sae_layer defaults to 12 (Gemma-2-2b-it mid-model)
  - patch matrix layers: [6, 12, 18]
  - model field: 'gemma-2-2b-it'
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from neuroscope.firebase_init import get_db

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)

from neuroscope import llm as ns_llm  # noqa: E402
from neuroscope.runner import patch_matrix, run_trajectory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("seed")

# Firebase DB client
db = get_db()

EXPERIMENTS = [
    {
        "slug": "hallucination-propagation",
        "title": "Hallucination Propagation",
        "category": "Hallucination",
        "hypothesis": (
            "H1: Hallucination-associated GemmaScope features activate 1-2 steps before\n"
            "the final hallucinated output, not at the output step itself."
        ),
        "task": "What year did Albert Einstein win the Nobel Prize, and for what discovery?",
        "n_steps": 4,
        "sae_layer": 12,
        "inject_observation": {2: "Note: Einstein won the Nobel Prize in 1933 for relativity."},
        "finding_seed": (
            "This experiment injects a false context at step 2 (claiming Einstein won the Nobel for "
            "relativity in 1933, both false) and watches whether the hallucination risk signal "
            "spikes BEFORE the final output is generated. If H1 holds, we expect GemmaScope feature "
            "drift and risk elevation at steps 2-3, with the bad output crystallizing at step 4."
        ),
    },
    {
        "slug": "tool-call-prediction",
        "title": "Tool-Call Prediction",
        "category": "Tool-use",
        "hypothesis": (
            "H2: Layer 12 residual stream activations can predict which tool the agent will call\n"
            "BEFORE the action token is emitted."
        ),
        "task": "You need the current weather in Paris. Choose between search, lookup, or calc, then provide the input.",
        "n_steps": 3,
        "sae_layer": 12,
        "inject_observation": None,
        "finding_seed": (
            "Examines tool-selection circuits across steps. If H2 holds, the GemmaScope features "
            "active at the THOUGHT token already encode the upcoming ACTION choice, visible as a "
            "stable set of co-active features 1-2 tokens before the action surfaces."
        ),
    },
    {
        "slug": "reasoning-collapse",
        "title": "Reasoning Collapse",
        "category": "Multi-hop",
        "hypothesis": (
            "H3: Cross-step causal patching of layer 12 activations from an earlier 'still on-track'\n"
            "step can restore a collapsed reasoning chain."
        ),
        "task": "If a train leaves at 14:30 and travels for 2 hours 45 minutes, what time does it arrive? Show work.",
        "n_steps": 4,
        "sae_layer": 12,
        "inject_observation": None,
        "finding_seed": (
            "Long arithmetic chains where Gemma-2-2b-it may collapse. Cross-step patches identify "
            "the step+layer where the chain breaks; significant KL on (source=1, target=4) at layer 12 "
            "would suggest the model carried early-step structure but lost it mid-chain."
        ),
    },
    {
        "slug": "ioi-persistence",
        "title": "IOI Persistence Across Steps",
        "category": "Circuit universality",
        "hypothesis": (
            "H4: The IOI (Indirect Object Identification) circuit discovered in single-prompt\n"
            "settings persists and re-activates across multi-step agent reasoning."
        ),
        "task": "When Sarah and Tom went to the store, Sarah gave a book to whom?",
        "n_steps": 3,
        "sae_layer": 12,
        "inject_observation": None,
        "finding_seed": (
            "Probes whether the IOI circuit — a canonical interpretability result from "
            "Wang et al. — remains visible at GemmaScope-feature granularity across three "
            "reasoning steps. Bridges single-prompt mechanistic interpretability to trajectory-level analysis."
        ),
    },
    {
        "slug": "self-correction",
        "title": "Self-Correction Mechanism",
        "category": "Self-correction",
        "hypothesis": (
            "H5: When the model catches its own error mid-trajectory, a distinct set of GemmaScope\n"
            "features activates that does not appear in failure trajectories."
        ),
        "task": "Compute 23 * 17. Then double-check by computing 17 * 23 and verify they match.",
        "n_steps": 4,
        "sae_layer": 12,
        "inject_observation": None,
        "finding_seed": (
            "Tests whether 'self-correction' is mechanistically distinguishable from 'self-doubt'. "
            "We expect a particular subset of mid-layer GemmaScope features to spike specifically "
            "when the model verifies a prior step."
        ),
    },
]


async def seed():
    from neuroscope.loader import MODEL_NAME, SAE_RELEASE
    is_gemma = "gemma" in MODEL_NAME.lower()
    patch_layers = [6, 12, 18] if is_gemma else [3, 7, 10]

    for spec in EXPERIMENTS:
        # Check if already seeded
        existing = db.collection("experiments").where("slug", "==", spec["slug"]).limit(1).get()
        if existing and not os.environ.get("FORCE_RESEED"):
            log.info("%s already exists; skipping (set FORCE_RESEED=1 to overwrite)", spec["slug"])
            continue

        log.info("== Seeding experiment: %s ==", spec["slug"])
        t0 = time.time()
        run_id = f"exp-{spec['slug']}"

        adjusted_sae_layer = spec["sae_layer"] if is_gemma else 7

        result = run_trajectory(
            run_id=run_id,
            task=spec["task"],
            n_steps=spec["n_steps"],
            sae_layer=adjusted_sae_layer,
            inject_context_at_step=spec["inject_observation"],
        )

        def target_prompt_fn(step_n: int) -> str:
            for s in result["steps"]:
                if s["step_n"] == step_n:
                    return s["prompt"]
            return ""

        pm = patch_matrix(result["steps"], target_prompt_fn, layers=patch_layers)

        ctx = {
            "task": spec["task"],
            "hypothesis": spec["hypothesis"],
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "steps": [
                {
                    "step_n": s["step_n"],
                    "output": s["output"][:160],
                    "hallucination": s["hallucination"],
                    "top_features": s["top_features"][:5],
                }
                for s in result["steps"]
            ],
            "feature_timelines": result["feature_timelines"][:6],
            "patch_summary": {
                "max_kl": max((r["kl"] for r in pm), default=0.0),
                "significant_count": sum(1 for r in pm if r["significant"]),
                "n": len(pm),
            },
        }
        finding = await ns_llm.report(ctx, session_id=f"seed-{spec['slug']}")

        doc = {
            "id": str(uuid.uuid4()),
            "slug": spec["slug"],
            "title": spec["title"],
            "category": spec["category"],
            "hypothesis": spec["hypothesis"],
            "task": spec["task"],
            "n_steps": spec["n_steps"],
            "sae_layer": adjusted_sae_layer,
            "model": MODEL_NAME,
            "steps": result["steps"],
            "feature_timelines": result["feature_timelines"],
            "patch_matrix": pm,
            "patch_matrix_summary": {
                "layers": patch_layers,
                "n_results": len(pm),
                "max_kl": max((r["kl"] for r in pm), default=0.0),
                "significant_count": sum(1 for r in pm if r["significant"]),
            },
            "finding_seed": spec["finding_seed"],
            "finding": finding,
            "total_elapsed_ms": result["total_elapsed_ms"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Upsert on slug (use slug as document ID)
        db.collection("experiments").document(spec["slug"]).set(doc)
        log.info("== Seeded %s in %.1fs ==", spec["slug"], time.time() - t0)


if __name__ == "__main__":
    asyncio.run(seed())

