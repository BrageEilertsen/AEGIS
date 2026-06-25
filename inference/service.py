"""AEGIS model-inference service core — a thin wrapper over the trained PyG checkpoint.

Loads the checkpoint once at startup (reusing the Phase 0-5 ml/ code), rebuilds the graph +
features with the saved train-only standardization (no leakage, same path as eval.py), caches the
node scores, and exposes score / flags / explain / adversarial / metrics. Owns NO product logic —
the Spring Boot BFF orchestrates and caps graphs; this just serves model inference.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import torch

from ml.adversarial.contract import AdversarialArtifactContract
from ml.common import compute_metrics, illicit_scores
from ml.explain import explain_node, load_checkpoint_and_model
from ml.features.assemble import assemble_features
from ml.train import build_graph, feature_config, resolve_device

from inference.narrate import LLM_ENABLED, llm_summary, template_summary


class ModelService:
    """Holds the loaded model + featurized graph + cached scores for the serving endpoints."""

    def __init__(self, checkpoint: str, feature_cache: str):
        self.checkpoint = checkpoint
        self.feature_cache = feature_cache
        self.model, self.cfg, self.in_channels, self.feature_meta, self.seed = \
            load_checkpoint_and_model(checkpoint)
        self.model_type = self.cfg.get("model", "gcn")
        self.device = resolve_device(self.cfg.get("device", "auto"))
        # Fast path: load a pre-built, fully-featurized graph (tiny + instant, ~50MB RAM) instead
        # of rebuilding from the 6.9M-row CSV (~3.5GB peak) — lets the service run on small/low-RAM
        # hosts. The artifact is the exact `data` object built below, so results are identical.
        # Falls back to rebuilding from the raw CSV when AEGIS_PREBUILT_GRAPH is unset/missing.
        prebuilt = os.environ.get("AEGIS_PREBUILT_GRAPH", "")
        if prebuilt and Path(prebuilt).exists():
            data = torch.load(prebuilt, map_location="cpu", weights_only=False)
            self.df = None
        else:
            data, df = build_graph(self.cfg, self.seed)
            data, _ = assemble_features(data, df, feature_config(self.cfg), feature_cache,
                                        standardization=self.feature_meta["standardization"])
            self.df = df
        self.model.to(self.device)
        self.data = data.to(self.device)
        # GNNExplainer epochs: 120 (default) is faithful but ~100s/node on CPU. Lower it for a
        # snappy interactive demo (results cache per node in the BFF after the first call).
        self.gnnex_epochs = int(os.environ.get("AEGIS_GNNEX_EPOCHS", "40"))
        with torch.no_grad():
            self._scores = illicit_scores(self.model(self.data))   # [N] illicit prob
        self.min_precision = float(self.cfg.get("eval", {}).get("min_precision", 0.9))
        # Async LLM-summary machinery: /explain returns instantly with the deterministic template
        # summary and the fluent LLM rephrasing is generated in the background (slow on CPU), cached
        # per node, and served via get_summary so the UI can upgrade the text in place. Single worker
        # — one autoregressive generation at a time keeps the 1-vCPU container from thrashing.
        self._summaries: dict[int, str] = {}
        self._summary_inflight: set[int] = set()
        self._summary_lock = threading.Lock()
        self._summary_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-summary")

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
                                num_hops=num_hops, max_nodes=max_nodes,
                                gnnex_epochs=self.gnnex_epochs).to_dict()
        # Instant, grounded summary so the UI renders immediately; the fluent LLM version (if
        # enabled) is generated in the background and fetched via /summary.
        contract["summary"] = template_summary(contract)
        contract["summary_pending"] = LLM_ENABLED
        if LLM_ENABLED:
            self._submit_summary(node_id, contract)
        return contract

    # ---- async LLM narration ----
    def _submit_summary(self, node_id: int, contract: dict | None) -> None:
        """Queue background LLM generation for a node (no-op if cached or already in flight)."""
        with self._summary_lock:
            if node_id in self._summaries or node_id in self._summary_inflight:
                return
            self._summary_inflight.add(node_id)
        self._summary_pool.submit(self._generate_summary, node_id, contract)

    def _generate_summary(self, node_id: int, contract: dict | None) -> None:
        try:
            if contract is None:   # /summary hit a node whose /explain result wasn't seeded
                contract = explain_node(self.model, self.data, node_id, self.feature_meta,
                                        model_type=self.model_type,
                                        gnnex_epochs=self.gnnex_epochs).to_dict()
            text = llm_summary(contract) or template_summary(contract)
            with self._summary_lock:
                self._summaries[node_id] = text
        except Exception:
            with self._summary_lock:                       # fall back so the node never polls forever
                self._summaries[node_id] = template_summary(contract or {"node_id": node_id})
        finally:
            with self._summary_lock:
                self._summary_inflight.discard(node_id)

    def get_summary(self, node_id: int) -> dict:
        """Poll target for the async LLM summary. ready=True once available; kicks off generation if
        it hasn't started (e.g. the explanation was served from the BFF cache without re-hitting us)."""
        if not (0 <= node_id < int(self.data.num_nodes)):
            raise ValueError(f"node_id {node_id} out of range [0, {int(self.data.num_nodes)})")
        if not LLM_ENABLED:
            return {"ready": True, "summary": None}        # nothing to upgrade to; UI keeps template
        with self._summary_lock:
            if node_id in self._summaries:
                return {"ready": True, "summary": self._summaries[node_id]}
        self._submit_summary(node_id, None)
        return {"ready": False, "summary": None}

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
