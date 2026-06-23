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
from ml.data.graph import build_transaction_graph, load_or_build, subsample_legitimate
from ml.data.loaders import load_ibm_aml, make_synthetic_aml
from ml.features.assemble import assemble_features
from ml.models.gcn import build_model


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
    if ds.get("graph", "transaction_as_node") != "transaction_as_node":
        raise ValueError("Phase 2 trains on graph: transaction_as_node only")
    ratios = cfg.get("split", {}).get("ratios", [0.6, 0.2, 0.2])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios must sum to 1, got {ratios}")
    if cfg.get("train", {}).get("loss", "weighted_ce") not in {"weighted_ce", "focal"}:
        raise ValueError("train.loss must be 'weighted_ce' or 'focal'")


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
    """Return (data, df) for the transaction-as-node graph. df is node-aligned (row i == node i)."""
    ds = cfg.get("dataset", {})
    variant = ds.get("variant", "LI-Small")
    delta_t_seconds = int(float(ds.get("delta_t_hours", 24)) * 3600)
    subsample = ds.get("subsample_legit", None)
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.6, 0.2, 0.2]))
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

    print(f"=== AEGIS train.py (Phase 2) | device={device} | seed={args.seed} ===")

    # 1) Graph (Phase-1 cache) + features (spectral cached, standardized train-only)
    data, df = build_graph(cfg, args.seed)
    print(f"[graph] nodes={data.num_nodes} edges={data.edge_index.size(1)} "
          f"illicit={int(data.y.sum())} ({float(data.y.float().mean()):.4%})")
    data, feat_meta = assemble_features(data, df, feature_config(cfg), args.feature_cache)
    print(f"[features] x={tuple(data.x.shape)} groups={feat_meta['group_dims']} "
          f"spectral_cache={'HIT' if feat_meta['spectral_cache']['hit'] else 'MISS'}")

    # 2) Model, loss, optimizer
    class_weights = compute_class_weights(data.y, data.train_mask)
    model = build_model(cfg, in_channels=data.x.size(1)).to(device)
    data = data.to(device)
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
