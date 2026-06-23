"""The explanation output contract — the versioned JSON boundary to the Java backend / Angular UI.

Every field is a JSON primitive or list of primitives (no tensors/numpy), so ``to_json`` always
serializes. See spec §7.7 for the contract shape.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

SCHEMA_VERSION = "1.0"


@dataclass
class ExplanationContract:
    node_id: int
    score: float
    predicted_label: int
    top_edges: list
    top_features: list
    matched_typology: dict
    neighborhood_subgraph: dict
    faithfulness: dict
    model_version: str
    feature_spec_version: str
    timestamp: str
    explainer_version: str = "v1"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["neighborhood_subgraph"] = {k: v for k, v in d["neighborhood_subgraph"].items()
                                      if not k.startswith("_")}   # strip internal fields
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExplanationContract":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
