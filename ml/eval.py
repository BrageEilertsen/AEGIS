#!/usr/bin/env python3
"""AEGIS — evaluation entrypoint.

Reports the metrics that matter at ~2% positives (spec §7.6); accuracy is meaningless here and
must never be the headline:

- PR-AUC on the illicit class               (headline)
- Recall at fixed precision                 (operational framing)
- F1 (illicit), confusion matrix, ROC-AUC   (secondary)
- Temporal generalization on later, unseen timesteps
- Benchmark comparison on Elliptic1 vs published baselines

Phase 0: skeleton only. Metric implementations land alongside the model in Phase 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS evaluation entrypoint")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="A train.py --out-dir containing a checkpoint to evaluate")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="Which temporal split to evaluate on")
    p.add_argument("--min-precision", type=float, default=0.9,
                   help="Precision threshold for the recall-at-fixed-precision metric")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== AEGIS eval.py (Phase 0 skeleton) ===")
    print(f"run-dir       : {args.run_dir}")
    print(f"split         : {args.split}")
    print(f"min-precision : {args.min_precision}")
    print()
    print("TODO (Phase 2+): load checkpoint + graph, run inference on the split, report "
          "PR-AUC, recall@precision>={:.2f}, F1-illicit, confusion matrix, ROC-AUC."
          .format(args.min_precision))
    print("Evaluation not yet implemented — exiting cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
