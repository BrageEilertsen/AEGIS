#!/usr/bin/env python3
"""Phase 1 entrypoint — build (and cache) PyG graphs from IBM-AML and print stats.

    python ml/data/build_graph.py --config experiments/gcn_baseline.yaml
    python ml/data/build_graph.py --config experiments/gcn_baseline.yaml --synthetic   # offline smoke test

Reads the ``dataset`` / ``split`` sections of the experiment YAML (the single source of truth),
locates ``<raw-dir>/<variant>_Trans.csv`` (or generates a synthetic frame with ``--synthetic``),
builds the requested graph view(s), caches them content-addressed under ``--cache-dir``, and
prints summary stats.

Done-criterion for Phase 1 (spec §12): produces a valid PyG graph with labels and temporal
splits, with basic stats printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script (python ml/data/build_graph.py) as well as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.graph import (  # noqa: E402
    build_account_graph, build_transaction_graph, load_or_build,
    print_stats, subsample_legitimate,
)
from ml.data.loaders import load_ibm_aml, make_synthetic_aml  # noqa: E402


def load_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build IBM-AML PyG graphs (Phase 1)")
    p.add_argument("--config", required=True, type=Path, help="Experiment YAML under experiments/")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Dir holding <variant>_Trans.csv")
    p.add_argument("--cache-dir", type=Path, default=Path("data/processed"), help="Where built graphs are cached")
    p.add_argument("--graph", choices=["transaction_as_node", "account_as_node", "both"],
                   default=None, help="Override the graph view from the config")
    p.add_argument("--synthetic", action="store_true", help="Use the synthetic generator (no download needed)")
    p.add_argument("--synthetic-legit", type=int, default=4000, help="Legit transaction count for --synthetic")
    p.add_argument("--nrows", type=int, default=None, help="Cap rows read from the real CSV (quick iteration)")
    p.add_argument("--force", action="store_true", help="Rebuild even if a cached graph exists")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    ds = cfg.get("dataset", {})
    variant = ds.get("variant", "LI-Small")
    delta_t_hours = float(ds.get("delta_t_hours", 24))
    delta_t_seconds = int(delta_t_hours * 3600)
    subsample = ds.get("subsample_legit", None)
    ratios = tuple(cfg.get("split", {}).get("ratios", [0.6, 0.2, 0.2]))
    kind = args.graph or ds.get("graph", "transaction_as_node")

    # ---- load transactions (real or synthetic) ----
    if args.synthetic:
        print(f"[data] generating synthetic IBM-AML frame (seed={args.seed}, legit={args.synthetic_legit})")
        df = make_synthetic_aml(n_legit=args.synthetic_legit, seed=args.seed)
        source = f"synthetic(seed={args.seed},legit={args.synthetic_legit})"
    else:
        trans_csv = args.raw_dir / f"{variant}_Trans.csv"
        print(f"[data] loading {trans_csv}" + (f" (nrows={args.nrows})" if args.nrows else ""))
        df = load_ibm_aml(trans_csv, nrows=args.nrows)
        source = f"{variant}_Trans.csv" + (f"#{args.nrows}" if args.nrows else "")

    print(f"[data] {len(df):,} transactions | illicit={int(df['label'].sum()):,} "
          f"({df['label'].mean():.4%}) | time span "
          f"{df['t'].max() / 86400:.1f} days")

    df = subsample_legitimate(df, subsample, seed=args.seed)
    if subsample is not None:
        print(f"[data] after subsample_legit={subsample}: {len(df):,} transactions "
              f"({df['label'].mean():.4%} illicit)")

    base_spec = {
        "source": source, "variant": variant, "subsample_legit": subsample,
        "split_ratios": list(ratios),
    }

    kinds = ["transaction_as_node", "account_as_node"] if kind == "both" else [kind]
    for k in kinds:
        if k == "transaction_as_node":
            spec = {**base_spec, "kind": k, "delta_t_seconds": delta_t_seconds}
            builder = lambda d: build_transaction_graph(d, delta_t_seconds, ratios)
        else:
            spec = {**base_spec, "kind": k}
            builder = build_account_graph

        data, path, cached = load_or_build(df, spec, args.cache_dir, builder, force=args.force)
        print(f"\n[graph] {k}: {'loaded from cache' if cached else 'built'} -> {path}")
        print_stats(data)

    print("\n[done] Phase 1 graph construction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
