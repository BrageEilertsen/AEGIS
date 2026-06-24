"""Pydantic request/response schemas for the inference service.

Response payloads for /explain and /adversarial are passed through as dicts (the ml/ layer already
produces versioned, JSON-native ExplanationContract / AdversarialArtifactContract objects), so the
schemas here cover the request bodies and the simple score/flag/metric responses.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    node_ids: list[int] | None = Field(default=None, description="nodes to score; null = all")


class NodeScore(BaseModel):
    node_id: int
    score: float
    label: int


class ExplainRequest(BaseModel):
    node_id: int
    method: str = Field(default="auto", pattern="^(auto|gnnexplainer|attention)$")
    num_hops: int = Field(default=2, ge=1, le=4)
    max_nodes: int = Field(default=400, ge=10, le=2000)


class AdversarialRequest(BaseModel):
    artifact_path: str | None = Field(default=None, description="path to a precomputed artifact.json")


class HealthResponse(BaseModel):
    status: str
    model: str
    num_nodes: int
    num_edges: int
    num_illicit: int
    feature_dim: int
    device: str
    dataset: str
