"""AEGIS model-inference service core — a thin wrapper over the trained PyG checkpoint.

Loads the checkpoint once at startup (reusing the Phase 0-5 ml/ code), rebuilds the graph +
features with the saved train-only standardization (no leakage, same path as eval.py), caches the
node scores, and exposes score / flags / explain / adversarial / metrics. Owns NO product logic —
the Spring Boot BFF orchestrates and caps graphs; this just serves model inference.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from ml.adversarial.contract import AdversarialArtifactContract
from ml.common import compute_metrics, illicit_scores
from ml.explain import explain_node, load_checkpoint_and_model
from ml.features.assemble import assemble_features
from ml.train import build_graph, feature_config, resolve_device


class ModelService:
    """Holds the loaded model + featurized graph + cached scores for the serving endpoints."""

    def __init__(self, checkpoint: str, feature_cache: str):
        self.checkpoint = checkpoint
        self.feature_cache = feature_cache
        self.model, self.cfg, self.in_channels, self.feature_meta, self.seed = \
            load_checkpoint_and_model(checkpoint)
        self.model_type = self.cfg.get("model", "gcn")
        self.device = resolve_device(self.cfg.get("device", "auto"))
        # Rebuild the exact graph + features used at train time; re-apply saved standardization.
        data, df = build_graph(self.cfg, self.seed)
        data, _ = assemble_features(data, df, feature_config(self.cfg), feature_cache,
                                    standardization=self.feature_meta["standardization"])
        self.model.to(self.device)
        self.data = data.to(self.device)
        self.df = df
        with torch.no_grad():
            self._scores = illicit_scores(self.model(self.data))   # [N] illicit prob
        self.min_precision = float(self.cfg.get("eval", {}).get("min_precision", 0.9))

    # ---- info ----
    def info(self) -> dict:
        n = int(self.data.num_nodes)
        return {"model": self.model_type, "checkpoint": self.checkpoint,
                "num_nodes": n, "num_edges": int(self.data.edge_index.size(1)),
                "num_illicit": int((self.data.y == 1).sum()),
                "feature_dim": int(self.data.x.size(1)),
                "feature_groups": self.feature_meta.get("group_dims", {}),
                "device": str(self.device),
                "dataset": self.cfg.get("dataset", {}).get("variant", "?")}

    # ---- scoring ----
    def scores_for(self, node_ids: list[int] | None) -> list[dict]:
        ids = node_ids if node_ids is not None else range(len(self._scores))
        y = self.data.y.cpu().numpy()
        return [{"node_id": int(i), "score": round(float(self._scores[i]), 6), "label": int(y[i])}
                for i in ids]

    def flags(self, threshold: float, limit: int) -> list[dict]:
        idx = [(i, float(s)) for i, s in enumerate(self._scores) if s >= threshold]
        idx.sort(key=lambda t: -t[1])
        y = self.data.y.cpu().numpy()
        return [{"node_id": int(i), "score": round(s, 6), "label": int(y[i])}
                for i, s in idx[:limit]]

    # ---- explanation ----
    def explain(self, node_id: int, method: str = "auto", num_hops: int = 2,
                max_nodes: int = 400) -> dict:
        if not (0 <= node_id < int(self.data.num_nodes)):
            raise ValueError(f"node_id {node_id} out of range [0, {int(self.data.num_nodes)})")
        contract = explain_node(self.model, self.data, node_id, self.feature_meta,
                                model_type=self.model_type, method=method,
                                num_hops=num_hops, max_nodes=max_nodes)
        return contract.to_dict()

    # ---- metrics ----
    def metrics(self, split: str = "test") -> dict:
        mask = self.data[f"{split}_mask"].cpu().numpy()
        return compute_metrics(self.data.y.cpu().numpy()[mask], self._scores[mask], self.min_precision)

    # ---- adversarial before/after (served from a precomputed artifact) ----
    def adversarial(self, artifact_path: str | None) -> dict:
        path = artifact_path or os.environ.get("AEGIS_ADVERSARIAL_ARTIFACT", "")
        if path and Path(path).exists():
            import json
            return AdversarialArtifactContract.from_dict(json.loads(Path(path).read_text())).to_dict()
        raise FileNotFoundError(
            "no precomputed adversarial artifact available; run `python -m ml.adversarial ... "
            "--out-dir <dir>` and set AEGIS_ADVERSARIAL_ARTIFACT=<dir>/artifact.json")


@lru_cache(maxsize=1)
def get_service() -> ModelService:
    """Singleton service from env: AEGIS_CHECKPOINT, AEGIS_FEATURE_CACHE."""
    ckpt = os.environ.get("AEGIS_CHECKPOINT")
    fc = os.environ.get("AEGIS_FEATURE_CACHE", "cache/features")
    if not ckpt:
        raise RuntimeError("set AEGIS_CHECKPOINT=<run-dir>/best.pt")
    return ModelService(ckpt, fc)
