#!/usr/bin/env python3
"""AEGIS — adversarial before/after artifact CLI (spec §7.8, §12).

    python -m ml.adversarial \
        --naive-ckpt <run>/best.pt --hardened-ckpt <run_adv>/best.pt \
        --feature-cache cache/features --out-dir <dir> --seed 42 [--smoke-test]

Writes artifact.json (the contract the FastAPI /adversarial endpoint serves) and prints the
headline before/after degradation for the naive vs hardened model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.adversarial.attacks import AttackConfig
from ml.adversarial.runner import run


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS adversarial before/after artifact")
    p.add_argument("--naive-ckpt", required=True, type=Path)
    p.add_argument("--hardened-ckpt", required=True, type=Path)
    p.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--budget-frac", type=float, default=0.20)
    p.add_argument("--budget-max", type=int, default=15)
    p.add_argument("--smoke-test", action="store_true", help="small target set for a quick login-CPU run")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AttackConfig(budget_frac=args.budget_frac, budget_max=args.budget_max,
                       top_k=20 if args.smoke_test else args.top_k, seed=args.seed)
    contract = run(args.naive_ckpt, args.hardened_ckpt, args.feature_cache, args.out_dir,
                   cfg, seed=args.seed, split=args.split)
    d = contract.degradation
    print("=== AEGIS adversarial before/after ===")
    print(f"  targets attacked     : {contract.attack['n_targets']}")
    print(f"  naive recall drop    : {d['naive_recall_drop']}")
    print(f"  hardened recall drop : {d['hardened_recall_drop']}")
    print(f"  robustness gap       : {d['robustness_gap']}")
    print(f"  {contract.summary}")
    print(f"  wrote                : {args.out_dir / 'artifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
