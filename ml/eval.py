#!/usr/bin/env python3
"""AEGIS — evaluation entrypoint (Phase 2).

Loads a trained checkpoint, rebuilds the graph + features (reusing the cached spectral features
and the checkpoint's TRAIN-only standardization stats — never recomputed, no leakage), runs
inference, and reports the metrics that matter at ~2% positives on the chosen temporal split:
PR-AUC (headline), recall@fixed-precision, F1-illicit, ROC-AUC, confusion matrix. Accuracy is
never reported as a headline (spec §7.6).

    python ml/eval.py --run-dir <out-dir> --split test --feature-cache <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.common import compute_metrics, illicit_scores
from ml.features.assemble import assemble_features
from ml.models import build_model
from ml.train import build_graph, feature_config, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS evaluation entrypoint")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="A train.py --out-dir containing best.pt + run_context.json")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--feature-cache", type=Path, default=Path("cache/features"),
                   help="Persistent spectral-feature cache (reused from training)")
    p.add_argument("--min-precision", type=float, default=None,
                   help="Override precision threshold for recall@precision (default: from config)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt_path = args.run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]
    seed = int(ckpt.get("seed", 42))
    feat_meta = ckpt["feature_meta"]
    standardization = feat_meta.get("standardization")
    assert standardization and "mean" in standardization, \
        "checkpoint is missing train-only standardization stats — cannot evaluate without leakage"

    device = resolve_device(cfg.get("device", "auto"))
    min_precision = args.min_precision if args.min_precision is not None \
        else float(cfg.get("eval", {}).get("min_precision", 0.9))

    print(f"=== AEGIS eval.py (Phase 2) | run={args.run_dir} | split={args.split} "
          f"| device={device} ===")

    # Rebuild the exact graph + features; APPLY saved train standardization (do not recompute).
    data, df = build_graph(cfg, seed)
    data, _ = assemble_features(data, df, feature_config(cfg), args.feature_cache,
                                standardization=standardization)
    if data.x.size(1) != ckpt["in_channels"]:
        print(f"ERROR: feature width {data.x.size(1)} != checkpoint {ckpt['in_channels']}",
              file=sys.stderr)
        return 2

    model = build_model(cfg, in_channels=ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    data = data.to(device)

    mask = data.test_mask if args.split == "test" else data.val_mask
    model.eval()
    with torch.no_grad():
        scores = illicit_scores(model(data))
    m = mask.cpu().numpy()
    metrics = compute_metrics(data.y.cpu().numpy()[m], scores[m], min_precision)

    (args.run_dir / f"eval_metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    print(f"  PR-AUC              : {metrics['pr_auc']}")
    print(f"  recall@p>={min_precision:<4}      : {metrics['recall_at_precision']} "
          f"(thr={metrics['threshold']})")
    print(f"  F1 (illicit)        : {metrics['f1_illicit']}")
    print(f"  ROC-AUC             : {metrics['roc_auc']}")
    print(f"  confusion           : {metrics['confusion_matrix']}")
    print(f"\n[done] wrote eval_metrics_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
