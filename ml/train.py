#!/usr/bin/env python3
"""AEGIS — single training entrypoint (Phase 2: spectral features + GCN baseline).

Stable CLI contract (do not change without updating cluster/train.slurm and the spec §11.3):

    python ml/train.py --config <yaml> --out-dir <dir> --feature-cache <dir> --seed <int>

Pipeline: build/cache the transaction graph (Phase 1) -> assemble raw+local+spectral features
(spectral cached, standardized train-only) -> train a GCN full-batch with weighted-CE/focal loss,
AdamW, LR schedule, early stopping on val PR-AUC -> checkpoint best.pt -> evaluate the temporal
test split, reporting PR-AUC / recall@precision / F1-illicit. Accuracy is never a headline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.common import (
    build_loss, compute_class_weights, compute_metrics, illicit_scores, set_deterministic,
)
from ml.adversarial.defenses import (
    make_adversarial_helpers, train_epoch_adversarial, validate_robust_aggregation,
)
from ml.data.elliptic import elliptic_temporal_masks, load_elliptic1, make_elliptic_fixture
from ml.data.graph import build_transaction_graph, load_or_build, subsample_legitimate
from ml.data.loaders import load_ibm_aml, make_synthetic_aml
from ml.features.assemble import assemble_features
from ml.models import build_model


# --------------------------------------------------------------------------------------------
# CLI + config
# --------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS GNN training entrypoint")
    p.add_argument("--config", required=True, type=Path, help="YAML config under experiments/")
    p.add_argument("--out-dir", required=True, type=Path, help="Per-run output directory")
    p.add_argument("--feature-cache", required=True, type=Path,
                   help="Persistent spectral-feature cache directory")
    p.add_argument("--seed", required=True, type=int, help="Random seed for reproducibility")
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def validate_config(cfg: dict) -> None:
    if cfg.get("model") not in {"gcn", "graphsage", "gat", "temporal"}:
        raise ValueError(f"config.model invalid: {cfg.get('model')!r}")
    ds = cfg.get("dataset", {})
    ds_name = ds.get("name", "ibm_aml")
    if ds_name not in {"ibm_aml", "elliptic"}:
        raise ValueError(f"dataset.name must be 'ibm_aml' or 'elliptic', got {ds_name!r}")
    if ds_name == "ibm_aml" and ds.get("graph", "transaction_as_node") != "transaction_as_node":
        raise ValueError("IBM-AML training uses graph: transaction_as_node only")
    ratios = cfg.get("split", {}).get("ratios", [0.6, 0.2, 0.2])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios must sum to 1, got {ratios}")
    if cfg.get("train", {}).get("loss", "weighted_ce") not in {"weighted_ce", "focal"}:
        raise ValueError("train.loss must be 'weighted_ce' or 'focal'")
    validate_robust_aggregation(cfg)


def resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto


# --------------------------------------------------------------------------------------------
# Data + features
# --------------------------------------------------------------------------------------------
def build_graph(cfg: dict, seed: int):
    """Return (data, df). df is the node-aligned IBM-AML frame, or None for Elliptic (features
    are provided by the dataset and assembled via the df-is-None path in assemble_features)."""
    ds = cfg.get("dataset", {})
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.6, 0.2, 0.2]))

    if ds.get("name", "ibm_aml") == "elliptic":
        if ds.get("fixture", False):
            data = make_elliptic_fixture(seed=seed, n_nodes=int(ds.get("fixture_nodes", 400)))
        else:
            data = load_elliptic1(ds.get("elliptic_cache_dir", "data/raw/elliptic"))
        for k, v in elliptic_temporal_masks(data.time_step, data.y, ratios).items():
            data[k] = v
        return data, None

    variant = ds.get("variant", "LI-Small")
    delta_t_seconds = int(float(ds.get("delta_t_hours", 24)) * 3600)
    subsample = ds.get("subsample_legit", None)
    max_out_per_in = ds.get("max_out_per_in", None)

    if ds.get("synthetic", False):
        df = make_synthetic_aml(n_legit=int(ds.get("synthetic_legit", 4000)), seed=seed)
        source = f"synthetic(seed={seed},legit={ds.get('synthetic_legit', 4000)})"
    else:
        df = load_ibm_aml(ds.get("trans_csv", f"data/raw/{variant}_Trans.csv"),
                          nrows=ds.get("nrows"))
        source = f"{variant}_Trans.csv" + (f"#{ds['nrows']}" if ds.get("nrows") else "")

    df = subsample_legitimate(df, subsample, seed=seed)
    spec = {"source": source, "variant": variant, "subsample_legit": subsample,
            "split_ratios": list(ratios), "kind": "transaction_as_node",
            "delta_t_seconds": delta_t_seconds}
    builder = lambda d: build_transaction_graph(d, delta_t_seconds, ratios, max_out_per_in)
    graph_cache_dir = ds.get("graph_cache_dir", "data/processed")
    data, _, _ = load_or_build(df, spec, graph_cache_dir, builder)
    return data, df


def feature_config(cfg: dict) -> dict:
    feats = cfg.get("features", {})
    fc = dict(feats.get("spectral", {}))
    fc["max_degree"] = feats.get("local", {}).get("max_degree", 2000) \
        if isinstance(feats.get("local"), dict) else 2000
    return fc


# --------------------------------------------------------------------------------------------
# Train / eval steps
# --------------------------------------------------------------------------------------------
def train_epoch(model, data, optimizer, loss_fn, grad_clip: float | None) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    m = data.train_mask
    loss = loss_fn(logits[m], data.y[m])
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def _make_neighbor_loader(data, cfg: dict):
    """Optional inductive neighbour-sampling loader (GraphSAGE/large graphs).

    Returns None — falling back to full-batch — when sampling is not configured OR when the
    backend (pyg-lib / torch-sparse) is unavailable (e.g. the login node). Full-batch stays the
    tested default everywhere; sampling only ever runs on a GPU node with the extras installed.
    """
    ns = cfg.get("train", {}).get("neighbor_sampling")
    if not ns:
        return None
    # The sampler needs pyg-lib or torch-sparse; NeighborLoader CONSTRUCTS fine without them but
    # raises ImportError on ITERATION, so check the backend up front rather than failing mid-epoch.
    if not (_module_available("pyg_lib") or _module_available("torch_sparse")):
        print("[sampling] WARNING: neighbor_sampling requested but no sampler backend "
              "(pyg-lib / torch-sparse) is installed; falling back to full-batch.")
        return None
    from torch_geometric.loader import NeighborLoader
    return NeighborLoader(data, num_neighbors=list(ns.get("num_neighbors", [25, 10])),
                          batch_size=int(ns.get("batch_size", 512)),
                          input_nodes=data.train_mask, shuffle=True)


def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def train_epoch_sampled(model, loader, optimizer, loss_fn, grad_clip, device) -> float:
    """Mini-batch training over NeighborLoader subgraphs (GPU only).

    Each batch is a sampled subgraph whose first ``batch.batch_size`` rows are the seed nodes
    (PyG guarantee); the same forward runs on the subgraph and the loss supervises the seeds.
    """
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        bs = batch.batch_size
        loss = loss_fn(logits[:bs], batch.y[:bs])
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, data, mask, min_precision: float) -> dict:
    model.eval()
    scores = illicit_scores(model(data))
    m = mask.cpu().numpy()
    y = data.y.cpu().numpy()[m]
    return compute_metrics(y, scores[m], min_precision)


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    validate_config(cfg)
    set_deterministic(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.get("device", "auto"))
    train_cfg = cfg.get("train", {})
    min_precision = float(cfg.get("eval", {}).get("min_precision", 0.9))

    print(f"=== AEGIS train.py | model={cfg.get('model')} | dataset={cfg.get('dataset', {}).get('name', 'ibm_aml')} "
          f"| device={device} | seed={args.seed} ===")

    # 1) Graph (Phase-1 cache) + features (spectral cached, standardized train-only)
    data, df = build_graph(cfg, args.seed)
    n_illicit = int((data.y == 1).sum())
    print(f"[graph] nodes={data.num_nodes} edges={data.edge_index.size(1)} "
          f"illicit={n_illicit} ({n_illicit / int(data.num_nodes):.4%}) "
          f"labeled_train={int(data.train_mask.sum())}")
    data, feat_meta = assemble_features(data, df, feature_config(cfg), args.feature_cache)
    print(f"[features] x={tuple(data.x.shape)} groups={feat_meta['group_dims']} "
          f"spectral_cache={'HIT' if feat_meta['spectral_cache']['hit'] else 'MISS'}")

    # 2) Model, loss, optimizer
    class_weights = compute_class_weights(data.y, data.train_mask)
    model = build_model(cfg, in_channels=data.x.size(1)).to(device)
    data = data.to(device)
    loader = _make_neighbor_loader(data, cfg)   # None -> full-batch (the tested default)
    if loader is not None:
        print(f"[sampling] neighbour sampling enabled: {train_cfg.get('neighbor_sampling')}")
    adv_helpers = make_adversarial_helpers(cfg, df, data, feature_config(cfg),
                                           feat_meta["standardization"], device)
    if adv_helpers is not None:
        print(f"[defense] adversarial training enabled: {train_cfg.get('adversarial_training')}")
    loss_fn = build_loss(train_cfg, class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 0.01)),
                                  weight_decay=float(train_cfg.get("weight_decay", 5e-4)))
    max_epochs = int(train_cfg.get("max_epochs", 200))
    sched_kind = train_cfg.get("lr_schedule", "plateau")
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        sp = train_cfg.get("lr_schedule_params", {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=float(sp.get("factor", 0.5)),
            patience=int(sp.get("patience", 5)), min_lr=float(sp.get("min_lr", 1e-6)))
    grad_clip = train_cfg.get("grad_clip_norm", None)
    patience = int(train_cfg.get("early_stopping", {}).get("patience", 20))

    # 3) Training loop with early stopping on val PR-AUC (maximize)
    best_val, best_epoch, best_state, no_improve = -1.0, -1, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        if adv_helpers is not None:
            loss = train_epoch_adversarial(model, data, adv_helpers, optimizer, loss_fn,
                                           grad_clip, epoch, args.seed)
        elif loader is not None:
            loss = train_epoch_sampled(model, loader, optimizer, loss_fn, grad_clip, device)
        else:
            loss = train_epoch(model, data, optimizer, loss_fn, grad_clip)
        val = evaluate(model, data, data.val_mask, min_precision)
        val_pr = val["pr_auc"] if val["pr_auc"] is not None else -1.0
        if sched_kind == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_pr)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": round(loss, 6),
                        "val_pr_auc": val_pr, "val_recall_at_p": val["recall_at_precision"],
                        "val_f1": val["f1_illicit"], "lr": lr_now})
        if val_pr > best_val:
            best_val, best_epoch = val_pr, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch % max(1, max_epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss {loss:.4f} | val PR-AUC {val_pr:.4f} | lr {lr_now:.2e}")
        if no_improve >= patience:
            print(f"  early stop at epoch {epoch} (best val PR-AUC {best_val:.4f} @ {best_epoch})")
            break

    # 4) Restore best, final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, data, data.test_mask, min_precision)
    val = evaluate(model, data, data.val_mask, min_precision)

    # 5) Persist checkpoint + metrics + run context
    torch.save({
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": cfg, "in_channels": int(data.x.size(1)), "feature_meta": feat_meta,
        "best_epoch": best_epoch, "val_pr_auc": best_val, "seed": args.seed,
    }, args.out_dir / "best.pt")

    metrics = {"best_epoch": best_epoch, "val": val, "test": test,
               "class_weights": class_weights.tolist()}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(args.out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_pr_auc",
                                          "val_recall_at_p", "val_f1", "lr"])
        w.writeheader()
        w.writerows(history)
    (args.out_dir / "run_context.json").write_text(json.dumps({
        "config_path": str(args.config), "out_dir": str(args.out_dir),
        "feature_cache": str(args.feature_cache), "seed": args.seed,
        "device": str(device), "config": cfg, "feature_meta": feat_meta,
    }, indent=2))

    print("\n=== TEST (temporal split) ===")
    print(f"  PR-AUC              : {test['pr_auc']}")
    print(f"  recall@p>={min_precision:<4}      : {test['recall_at_precision']} "
          f"(thr={test['threshold']})")
    print(f"  F1 (illicit)        : {test['f1_illicit']}")
    print(f"  ROC-AUC             : {test['roc_auc']}")
    print(f"  confusion           : {test['confusion_matrix']}")
    print(f"\n[done] artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
