"""Reproducible before/after artifact engine (spec §7.8, §12 acceptance).

Loads a naive and a hardened checkpoint, attacks the SAME target set on each, measures the 2×2×N
metric matrix (naive/hardened × clean/perturbed × {PR-AUC, recall@p, F1}), and emits a versioned
AdversarialArtifactContract the FastAPI /adversarial endpoint serves. Reuses the Phase 0-4 graph,
feature, model, and metric code; standardization is re-applied from each checkpoint (never recomputed).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from ml.adversarial.attacks import AttackConfig, GreedyEdgeAttack, query_scores, select_target_nodes
from ml.adversarial.contract import AdversarialArtifactContract
from ml.adversarial.featurize import MASKS, make_refeaturizer
from ml.common import compute_metrics, set_deterministic
from ml.features.assemble import assemble_features
from ml.features.spectral import graph_hash
from ml.models import build_model
from ml.train import build_graph, feature_config, resolve_device


def _load(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, ckpt["in_channels"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    feat_meta = ckpt["feature_meta"]
    assert feat_meta.get("standardization"), "checkpoint missing standardization (cannot featurize)"
    return model, cfg, feat_meta, int(ckpt.get("seed", 42))


@torch.no_grad()
def measure(model, data, split: str, min_precision: float) -> dict:
    mask = data[f"{split}_mask"].cpu().numpy()
    scores = query_scores(model, data)
    return compute_metrics(data.y.cpu().numpy()[mask], scores[mask], min_precision)


def _recall(m: dict) -> float:
    v = m.get("recall_at_precision")
    return float(v) if v is not None else 0.0


def _target_stats(records: list[dict], threshold: float) -> dict:
    """Attack success on the target set: fraction pushed below the detection threshold + mean drop."""
    if not records:
        return {"attack_success_rate": 0.0, "mean_score_drop": 0.0}
    below = sum(1 for r in records if r["score_after"] < threshold)
    drop = sum(max(0.0, -r["delta"]) for r in records) / len(records)
    return {"attack_success_rate": round(below / len(records), 4), "mean_score_drop": round(drop, 4)}


def _attack_model(ckpt_path, feature_cache, device, targets, cfg_attack, seed):
    """Load a checkpoint, attack `targets` on its own (clean) graph; return everything needed."""
    model, cfg, feat_meta, _ = _load(ckpt_path)
    model.to(device)
    data, df = build_graph(cfg, seed)
    data, _ = assemble_features(data, df, feature_config(cfg), feature_cache,
                               standardization=feat_meta["standardization"])
    data = data.to(device)
    base_masks = {name: data[name].cpu() for name in MASKS}
    refeat, raw_x, node_amount = make_refeaturizer(
        df, feature_config(cfg), feat_meta["standardization"], data.y.cpu(), base_masks, device)
    attack = GreedyEdgeAttack(model, data.edge_index.cpu(), raw_x, node_amount, refeat, cfg_attack)
    perturbed, records = attack.run(targets)
    return {"model": model, "cfg": cfg, "feat_meta": feat_meta, "clean": data,
            "perturbed": perturbed, "records": records, "df": df,
            "is_hardened": bool(cfg.get("train", {}).get("adversarial_training"))}


def run(naive_ckpt, hardened_ckpt, feature_cache, out_dir, attack_cfg: AttackConfig,
        seed: int = 42, split: str = "test") -> AdversarialArtifactContract:
    set_deterministic(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Targets are chosen on the NAIVE model's clean scores (the shared attack set).
    naive_model, naive_cfg, naive_meta, _ = _load(naive_ckpt)
    device = resolve_device(naive_cfg.get("device", "auto"))
    naive_model.to(device)
    min_precision = float(naive_cfg.get("eval", {}).get("min_precision", 0.9))
    data0, df0 = build_graph(naive_cfg, seed)
    data0, _ = assemble_features(data0, df0, feature_config(naive_cfg), feature_cache,
                                standardization=naive_meta["standardization"])
    data0 = data0.to(device)
    scores0 = query_scores(naive_model, data0)
    targets = select_target_nodes(scores0, data0.y, data0[f"{split}_mask"], attack_cfg)

    naive = _attack_model(naive_ckpt, feature_cache, device, targets, attack_cfg, seed)
    hardened = _attack_model(hardened_ckpt, feature_cache, device, targets, attack_cfg, seed)

    # The before/after attacks the SAME target node-indices on both models, so they must share the
    # exact graph structure; otherwise target ids point at unrelated nodes and the comparison is
    # silently wrong. Fail loud on a graph mismatch (operator passed checkpoints from different runs).
    if graph_hash(naive["clean"]) != graph_hash(hardened["clean"]):
        raise ValueError("naive and hardened checkpoints were trained on different graphs "
                         "(structure hashes differ); the shared-target before/after is invalid")

    metrics = {
        "naive": {"clean": measure(naive["model"], naive["clean"], split, min_precision),
                  "perturbed": measure(naive["model"], naive["perturbed"], split, min_precision)},
        "hardened": {"clean": measure(hardened["model"], hardened["clean"], split, min_precision),
                     "perturbed": measure(hardened["model"], hardened["perturbed"], split, min_precision)},
    }
    naive_drop = _recall(metrics["naive"]["clean"]) - _recall(metrics["naive"]["perturbed"])
    hard_drop = _recall(metrics["hardened"]["clean"]) - _recall(metrics["hardened"]["perturbed"])
    # Target-level signal: not floored by an unreachable whole-set precision. The clearest "naive
    # fooled vs hardened holds" story = what fraction of attacked flags fell below the detection
    # threshold, and the mean score drop, on each model over the SAME target set.
    thr = attack_cfg.score_threshold
    ns = _target_stats(naive["records"], thr)
    hs = _target_stats(hardened["records"], thr)
    degradation = {
        "naive_recall_drop": round(naive_drop, 4),
        "hardened_recall_drop": round(hard_drop, 4),
        "robustness_gap": round(naive_drop - hard_drop, 4),
        "naive_pr_auc_drop": round((metrics["naive"]["clean"].get("pr_auc") or 0)
                                   - (metrics["naive"]["perturbed"].get("pr_auc") or 0), 4),
        "hardened_pr_auc_drop": round((metrics["hardened"]["clean"].get("pr_auc") or 0)
                                      - (metrics["hardened"]["perturbed"].get("pr_auc") or 0), 4),
        "naive_attack_success_rate": ns["attack_success_rate"],
        "hardened_attack_success_rate": hs["attack_success_rate"],
        "naive_mean_score_drop": ns["mean_score_drop"],
        "hardened_mean_score_drop": hs["mean_score_drop"],
        "target_robustness_gap": round(ns["attack_success_rate"] - hs["attack_success_rate"], 4),
    }

    # Merge per-target records (naive + hardened keyed by node_id).
    hmap = {r["node_id"]: r for r in hardened["records"]}
    per_target = []
    for r in naive["records"]:
        h = hmap.get(r["node_id"], {})
        per_target.append({
            "node_id": r["node_id"], "y": 1,
            "naive_score_before": r["score_before"], "naive_score_after": r["score_after"],
            "naive_delta": r["delta"], "hardened_score_before": h.get("score_before"),
            "hardened_score_after": h.get("score_after"), "hardened_delta": h.get("delta"),
            "n_edits": r["n_edits"], "mules_added": r["mules_added"],
            "net_flow_drift": r["net_flow_drift"], "edits": r["edits"]})

    contract = AdversarialArtifactContract(
        seed=seed, split=split,
        graph={"source": naive_meta.get("dataset", "ibm_aml"),
               "num_nodes": int(data0.num_nodes), "num_edges": int(data0.edge_index.size(1)),
               "num_illicit": int((data0.y == 1).sum()),
               "illicit_ratio": round(float((data0.y == 1).float().mean()), 6),
               "graph_hash": graph_hash(data0.cpu()),
               "feature_spec_version": naive_meta.get("feature_spec_version", "?")},
        attack={"kind": "greedy_edge_perturbation", "moves": ["remove", "mule"],
                "budget_frac": attack_cfg.budget_frac, "budget_max": attack_cfg.budget_max,
                "net_flow_tol": attack_cfg.net_flow_tol, "score_threshold": attack_cfg.score_threshold,
                "top_k": attack_cfg.top_k, "n_targets": len(targets), "deterministic": True},
        models={"naive": {"model": naive_cfg.get("model"), "ckpt": str(naive_ckpt), "hardened": False},
                "hardened": {"model": hardened["cfg"].get("model"), "ckpt": str(hardened_ckpt),
                             "hardened": True,
                             "adversarial_training": hardened["cfg"].get("train", {}).get("adversarial_training")}},
        metrics=metrics, degradation=degradation, per_target=per_target,
        perturbed_subgraph=_representative_subgraph(naive, targets, scores0, int(data0.num_nodes)),
        constraint_violations=[r for r in naive["records"] if r["net_flow_drift"] > attack_cfg.net_flow_tol],
        summary=_summary(metrics, degradation),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    (out_dir / "artifact.json").write_text(contract.to_json())
    return contract


def _representative_subgraph(naive, targets, scores0, N) -> dict:
    if not naive["records"]:
        return {}
    rec = max(naive["records"], key=lambda r: abs(r["delta"]))
    t = rec["node_id"]
    clean_ei = naive["clean"].edge_index.cpu()
    pert_ei = naive["perturbed"].edge_index.cpu()

    def incident(ei):
        mask = (ei[0] == t) | (ei[1] == t)
        return [[int(s), int(d)] for s, d in ei[:, mask].t().tolist()]

    edges_after = incident(pert_ei)
    mules = sorted({n for e in edges_after for n in e if n >= N})
    return {"target_node": t, "edges_before": incident(clean_ei), "edges_after": edges_after,
            "injected_mules": mules,
            "node_scores_before": {str(t): rec["score_before"]},
            "node_scores_after": {str(t): rec["score_after"]}}


def _summary(metrics, deg) -> str:
    return (f"Structural attack pushed {deg['naive_attack_success_rate']:.0%} of the naive model's "
            f"flagged transactions below the detection threshold (mean score drop "
            f"{deg['naive_mean_score_drop']:.2f}); against the adversarially-trained model only "
            f"{deg['hardened_attack_success_rate']:.0%} fell (mean drop "
            f"{deg['hardened_mean_score_drop']:.2f}). Target robustness gap "
            f"{deg['target_robustness_gap']:+.0%}. Whole-set PR-AUC drop: naive "
            f"{deg['naive_pr_auc_drop']:+.3f} vs hardened {deg['hardened_pr_auc_drop']:+.3f}.")
