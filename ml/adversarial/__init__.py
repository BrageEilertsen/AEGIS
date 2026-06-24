"""Adversarial robustness subsystem (spec §7.8) — the showpiece.

A model-agnostic structural evasion attack (GreedyEdgeAttack), an adversarial-training defense
(train_epoch_adversarial) + robust median aggregation, and a reproducible before/after artifact
(run) emitting AdversarialArtifactContract for the app's /adversarial demo.

``run`` is imported lazily: the runner pulls ``ml.train``, which imports this package's defenses —
deferring keeps package import acyclic.
"""
from __future__ import annotations

from ml.adversarial.attacks import AttackConfig, GreedyEdgeAttack, query_scores, select_target_nodes
from ml.adversarial.contract import SCHEMA_VERSION, AdversarialArtifactContract
from ml.adversarial.defenses import (
    make_adversarial_helpers, train_epoch_adversarial, validate_robust_aggregation,
)


def run(*args, **kwargs):
    """Run the before/after artifact (see ml.adversarial.runner.run). Imported lazily."""
    from ml.adversarial.runner import run as _run
    return _run(*args, **kwargs)


__all__ = ["AttackConfig", "GreedyEdgeAttack", "query_scores", "select_target_nodes",
           "AdversarialArtifactContract", "SCHEMA_VERSION", "make_adversarial_helpers",
           "train_epoch_adversarial", "validate_robust_aggregation", "run"]
