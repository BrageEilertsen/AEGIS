#!/usr/bin/env python3
"""AEGIS — explainability entrypoint (Phase 4).

Loads a trained checkpoint, rebuilds the graph + features (re-applying the saved train-only
standardization — same no-leakage path as eval.py), explains one flagged node, and writes the
versioned explanation contract (spec §7.7) as JSON.

    python ml/explain/cli.py --run-dir <out-dir> --node <idx> \
        --feature-cache cache/features --out-json explanation.json [--method auto]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.explain import explain_node, load_checkpoint_and_model
from ml.features.assemble import assemble_features
from ml.train import build_graph, feature_config, resolve_device


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEGIS explanation entrypoint")
    p.add_argument("--run-dir", required=True, type=Path, help="train.py --out-dir with best.pt")
    p.add_argument("--node", required=True, type=int, help="node index to explain")
    p.add_argument("--feature-cache", type=Path, default=Path("cache/features"))
    p.add_argument("--out-json", type=Path, default=None, help="where to write the contract JSON")
    p.add_argument("--method", default="auto", choices=["auto", "gnnexplainer", "attention"])
    p.add_argument("--num-hops", type=int, default=2)
    p.add_argument("--max-nodes", type=int, default=400)
    p.add_argument("--gnnex-epochs", type=int, default=120)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = args.run_dir / "best.pt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    model, cfg, _, feat_meta, seed = load_checkpoint_and_model(ckpt)
    device = resolve_device(cfg.get("device", "auto"))

    data, df = build_graph(cfg, seed)
    data, _ = assemble_features(data, df, feature_config(cfg), args.feature_cache,
                                standardization=feat_meta["standardization"])
    model.to(device)
    data = data.to(device)

    contract = explain_node(model, data, args.node, feat_meta, model_type=cfg.get("model"),
                            num_hops=args.num_hops, max_nodes=args.max_nodes,
                            gnnex_epochs=args.gnnex_epochs, method=args.method)

    out_path = args.out_json or (args.run_dir / f"explanation_node{args.node}.json")
    out_path.write_text(contract.to_json())
    print(f"=== explanation for node {args.node} ===")
    print(f"  score              : {contract.score}")
    print(f"  predicted_label    : {contract.predicted_label}")
    print(f"  matched_typology   : {contract.matched_typology['label']} "
          f"(conf {contract.matched_typology['confidence']})")
    print(f"  top_edges          : {len(contract.top_edges)}")
    print(f"  edge_importance_src: {contract.faithfulness['edge_importance_source']}")
    print(f"  wrote              : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
