"""AEGIS FastAPI inference service (spec §8.2).

Thin model server: loads the trained checkpoint at startup and exposes scoring, flags, faithful
explanations, the adversarial before/after, and metrics. The Spring Boot BFF calls these and owns
graph-capping + product logic.

    AEGIS_CHECKPOINT=outputs/<run>/best.pt AEGIS_FEATURE_CACHE=cache/features \
        uvicorn inference.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from inference.schemas import (
    AdversarialRequest, ExplainRequest, HealthResponse, NodeScore, ScoreRequest,
)
from inference.service import get_service

app = FastAPI(title="AEGIS Inference Service", version="1.0",
              description="GNN money-laundering detection: score, explain, adversarial.")


@app.get("/health", response_model=HealthResponse)
def health():
    s = get_service()
    return {"status": "ok", **{k: v for k, v in s.info().items()
                               if k in {"model", "num_nodes", "num_edges", "num_illicit",
                                        "feature_dim", "device", "dataset"}}}


@app.get("/info")
def info():
    return get_service().info()


@app.post("/score", response_model=list[NodeScore])
def score(req: ScoreRequest):
    return get_service().scores_for(req.node_ids)


@app.get("/flags", response_model=list[NodeScore])
def flags(threshold: float = Query(0.5, ge=0.0, le=1.0), limit: int = Query(100, ge=1, le=5000)):
    return get_service().flags(threshold, limit)


@app.post("/explain")
def explain(req: ExplainRequest):
    try:
        return get_service().explain(req.node_id, req.method, req.num_hops, req.max_nodes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/summary/{node_id}")
def summary(node_id: int):
    """Poll for the async, grounded LLM narration of a node's explanation.
    Returns {ready: bool, summary: str|null}; the UI shows the instant template until ready."""
    try:
        return get_service().get_summary(node_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/metrics")
def metrics(split: str = Query("test", pattern="^(val|test)$")):
    return get_service().metrics(split)


@app.post("/adversarial")
def adversarial(req: AdversarialRequest):
    try:
        return get_service().adversarial(req.artifact_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
