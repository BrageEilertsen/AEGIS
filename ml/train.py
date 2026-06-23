#!/usr/bin/env python3
"""AEGIS — single training entrypoint.

Stable CLI contract (do not change without updating cluster/train.slurm and the spec §11.3):

    python ml/train.py --config <yaml> --out-dir <dir> --feature-cache <dir> --seed <int>

- --config         YAML under experiments/ — the single source of truth for hyperparameters.
- --out-dir        Per-run output directory (metrics, checkpoint, logs).
- --feature-cache  Persistent spectral-feature cache, keyed by graph hash (reused across runs).
- --seed           Fixed for reproducibility.

Phase 0: this is a skeleton. It parses args, loads the config, seeds RNGs, and reports what it
would do. The data pipeline, models, and training loop are wired in Phase 1+.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS GNN training entrypoint")
    p.add_argument("--config", required=True, type=Path,
                   help="YAML hyperparameter config under experiments/")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Per-run output directory (metrics, checkpoint, logs)")
    p.add_argument("--feature-cache", required=True, type=Path,
                   help="Persistent spectral-feature cache directory")
    p.add_argument("--seed", required=True, type=int,
                   help="Random seed for reproducibility")
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    import yaml  # local import so --help works without deps installed
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    set_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache.mkdir(parents=True, exist_ok=True)

    # Record the resolved run context for reproducibility (spec §7.5).
    run_context = {
        "config_path": str(args.config),
        "out_dir": str(args.out_dir),
        "feature_cache": str(args.feature_cache),
        "seed": args.seed,
        "config": cfg,
    }
    (args.out_dir / "run_context.json").write_text(json.dumps(run_context, indent=2))

    print("=== AEGIS train.py (Phase 0 skeleton) ===")
    print(f"config        : {args.config}")
    print(f"out-dir       : {args.out_dir}")
    print(f"feature-cache : {args.feature_cache}")
    print(f"seed          : {args.seed}")
    print(f"model (cfg)   : {cfg.get('model', '<unset>')}")
    print(f"dataset (cfg) : {cfg.get('dataset', {}).get('variant', '<unset>')}")
    print()
    print("TODO (Phase 1+): build PyG graph from the dataset, compute features "
          "(raw + local + spectral), train the model, evaluate (PR-AUC / "
          "recall@precision / F1-illicit), checkpoint to --out-dir.")
    print("Training loop not yet implemented — exiting cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
