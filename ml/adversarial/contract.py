"""Versioned before/after JSON contract for the adversarial robustness artifact (spec §7.8, §8.2).

The FastAPI /adversarial endpoint serves ``to_dict()`` of the cached artifact; the UI animates the
naive-fooled vs hardened-holds story. Every field is a JSON primitive / list / dict.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

SCHEMA_VERSION = "1.0"


@dataclass
class AdversarialArtifactContract:
    seed: int
    split: str
    graph: dict                 # provenance/scale (source, num_nodes/edges, illicit ratio, hashes)
    attack: dict                # exact attack params (reproducibility)
    models: dict                # {naive, hardened} -> {model, ckpt, hardened, [adversarial_training]}
    metrics: dict               # naive/hardened x clean/perturbed -> compute_metrics() dict
    degradation: dict           # naive_recall_drop, hardened_recall_drop, robustness_gap, ...
    per_target: list            # one record per attacked target (scores before/after, edits, drift)
    perturbed_subgraph: dict    # one representative target, for UI animation
    constraint_violations: list
    summary: str
    generated_at: str
    artifact_version: str = "v1"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "AdversarialArtifactContract":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
