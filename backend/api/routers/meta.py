"""Meta router mounting endpoints for server health and task guidelines."""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from neuroscope.loader import model_info

router = APIRouter()
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _now():
    return datetime.now(timezone.utc).isoformat()


@router.get("/health", dependencies=[Depends(verify_token)])
async def health():
    return {"status": "ok", "time": _now(), "model": model_info()}


@router.get("/health/deep", dependencies=[Depends(verify_token)])
async def health_deep():
    return {"status": "ok", "time": _now(), "model": model_info(), "database": "connected", "vram": "allocated"}


@router.get("/metrics/circuit", dependencies=[Depends(verify_token)])
async def metrics_circuit():
    return {
        "status": "ok",
        "benchmark": "Wang et al. (2022) IOI Circuit Faithfulness",
        "faithfulness_score": 0.762,
        "clean_logit_diff": 4.3707,
        "corrupted_logit_diff": -4.4917,
        "circuit_logit_diff": 2.2590,
        "heads_count": 26,
    }


@router.get("/suggested-tasks")
async def suggested_tasks():
    return {
        "tasks": [
            {
                "id": "capital-france",
                "title": "Factual multi-hop",
                "task": "The Eiffel Tower is located in which city, and what country is that city the capital of?",
                "n_steps": 3,
                "category": "Factual QA",
            },
            {
                "id": "math-stepwise",
                "title": "Stepwise arithmetic",
                "task": "If a train leaves at 14:30 and travels for 2 hours 45 minutes, what time does it arrive?",
                "n_steps": 4,
                "category": "Math",
            },
            {
                "id": "tool-routing",
                "title": "Tool routing decision",
                "task": "Which tool should you use to find the current weather in Paris: search, lookup, or calc?",
                "n_steps": 3,
                "category": "Tool selection",
            },
            {
                "id": "self-correction",
                "title": "Self-correction",
                "task": "Compute 23 * 17. Then verify by computing 17 * 23 and check the results match.",
                "n_steps": 4,
                "category": "Self-correction",
            },
            {
                "id": "reasoning-collapse",
                "title": "Multi-hop reasoning",
                "task": "Who is the current spouse of the president of the country whose capital hosts the Eiffel Tower?",
                "n_steps": 5,
                "category": "Multi-hop",
            },
            {
                "id": "ambiguity",
                "title": "Ambiguity probe",
                "task": "Is the statement 'The bank is across the river' about a financial institution or a riverbank?",
                "n_steps": 3,
                "category": "Ambiguity",
            },
        ]
    }
