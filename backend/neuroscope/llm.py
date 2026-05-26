"""Google Gemini client for NL circuit query explanations.

Uses the official google-generativeai Python SDK with gemini-1.5-flash.
100% free under Google AI Studio free tier limits.
"""
from __future__ import annotations

import json
import logging
import os

import google.generativeai as genai

logger = logging.getLogger(__name__)

_initialized = False


def _init_client():
    global _initialized
    if not _initialized:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Add it to backend/.env."
            )
        # Configure the Google GenAI SDK
        genai.configure(api_key=api_key)
        _initialized = True


SYSTEM_QUERY = """You are a mechanistic interpretability assistant for NeuroScope.
You have access to a multi-step agent trajectory run through Gemma-2-2b-it (26-layer transformer) with:
- Per-step residual stream activation L2-norms at captured layers 6, 12, 18, 24
- Top GemmaScope features (16k-width canonical SAEs, trained on resid_POST) with per-step activations and drift scores
- Three-signal hallucination risk per step: entropy + attention diffusion + drift proxy (no hardcoded feature IDs)
- Cross-step causal patching results: KL(patched||baseline) and top token shifts

Answer questions about the model's internal behavior in 3-6 sentences.
Cite specific steps, layers, and feature IDs (e.g. 'GemmaScope feature #8421 at layer 12').
Be technically precise and acknowledge uncertainty.
Do not overstate what interpretability can show.
Never claim features are 'the circuit for X' — claim they 'co-activate with X' or 'are associated with X'.
"""

SYSTEM_REPORT = """You are a mechanistic interpretability research-writer for NeuroScope.
Given a trajectory summary from Gemma-2-2b-it analysis with GemmaScope SAEs, write a 4-6 sentence research finding.
Be technically precise. Cite step numbers, layer numbers, feature IDs, and KL values.
Acknowledge what the data cannot show (e.g. correlation vs causation).
"""


async def ask(query: str, context: dict, session_id: str) -> str:
    """Answer a natural language question about a trajectory using Gemini."""
    _init_client()
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_QUERY
        )
        prompt = (
            f"Trajectory data (JSON):\n{json.dumps(context, indent=2)}"
            f"\n\nQuestion: {query}"
        )
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
            )
        )
        return response.text
    except Exception as e:
        logger.exception("Gemini API error for query (session=%s)", session_id)
        return f"[Query error: {type(e).__name__}: {e}]"


async def report(context: dict, session_id: str) -> str:
    """Generate a research finding paragraph for a trajectory using Gemini."""
    _init_client()
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_REPORT
        )
        prompt = (
            f"Trajectory data (JSON):\n{json.dumps(context, indent=2)}"
            f"\n\nWrite the finding paragraph."
        )
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=256,
            )
        )
        return response.text
    except Exception as e:
        logger.exception("Gemini API error for report (session=%s)", session_id)
        return f"[Report error: {type(e).__name__}: {e}]"
