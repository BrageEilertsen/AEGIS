"""Phase 1 correctness tests — graph construction + temporal split.

Runs on the synthetic IBM-AML frame (no download needed), so it works on the login node.
Run standalone:  python tests/test_phase1_graph.py
Or under pytest:  pytest tests/test_phase1_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.data.graph import (  # noqa: E402
    build_account_graph, build_transaction_graph, temporal_split_masks,
)
from ml.data.loaders import REQUIRED_COLUMNS, make_synthetic_aml  # noqa: E402

DT = 24 * 3600


def _df():
    return make_synthetic_aml(n_legit=2000, seed=7)


def test_synthetic_schema_and_labels():
    df = _df()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert {"t", "src_account", "dst_account", "label"} <= set(df.columns)
    assert df["label"].isin([0, 1]).all()
    assert df["label"].sum() > 0, "synthetic data should contain injected illicit transactions"
    # normalize() sorts by time
    assert (df["t"].to_numpy()[:-1] <= df["t"].to_numpy()[1:]).all()


def test_transaction_edges_respect_account_and_window():
    df = _df()
    data = build_transaction_graph(df, DT)
    t = df["t"].to_numpy(); src = df["src_account"].to_numpy(); dst = df["dst_account"].to_numpy()
    ei = data.edge_index.numpy()
    assert data.num_nodes == len(df)
    assert data.y.shape[0] == len(df)
    for a, b in ei.T:
        assert dst[a] == src[b], "edge must link through a shared account (receiver_i == sender_j)"
        assert 0 <= t[b] - t[a] <= DT, "edge must fall within the Δt window"


def test_temporal_split_is_ordered_disjoint_and_complete():
    df = _df()
    masks = temporal_split_masks(df["t"].to_numpy(), (0.6, 0.2, 0.2))
    tr, va, te = (masks[f"{s}_mask"].numpy() for s in ("train", "val", "test"))
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert (tr | va | te).all()
    t = df["t"].to_numpy()
    assert t[tr].max() <= t[va].min() <= t[va].max() <= t[te].min(), "splits must be time-ordered"


def test_account_graph_shapes():
    df = _df()
    data = build_account_graph(df)
    assert data.edge_index.size(1) == len(df), "one edge per transaction"
    assert data.edge_attr.size(0) == len(df)
    assert data.y.shape[0] == data.num_nodes
    assert data.y.max() <= 1


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
