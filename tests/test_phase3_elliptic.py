"""Phase 3 correctness tests — Elliptic1 benchmark code path (fixture only, NO download).

Login-CPU. Standalone: python tests/test_phase3_elliptic.py | pytest tests/test_phase3_elliptic.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.data.elliptic import UNKNOWN, elliptic_temporal_masks, make_elliptic_fixture
from ml.features.assemble import assemble_features
from ml.models import build_model


def test_elliptic_fixture_shape():
    d = make_elliptic_fixture(seed=0, n_nodes=400, n_features=166)
    assert d.x.shape == (400, 166)
    assert d.graph_kind == "elliptic_tx"
    assert int(d.time_step.min()) >= 1 and int(d.time_step.max()) <= 49
    frac_unknown = float((d.y == UNKNOWN).float().mean())
    assert 0.6 < frac_unknown < 0.9, f"unknown fraction {frac_unknown}"
    assert (d.y == 1).sum() > 0 and (d.y == 0).sum() > 0


def test_elliptic_masks_exclude_unknown():
    d = make_elliptic_fixture(seed=1, n_nodes=600)
    masks = elliptic_temporal_masks(d.time_step, d.y, (0.6, 0.2, 0.2))
    tr, va, te = (masks[f"{s}_mask"] for s in ("train", "val", "test"))
    for m in (tr, va, te):
        assert (d.y[m] != UNKNOWN).all(), "no unknown node may enter any split"
    # disjoint
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    # temporal order over labeled nodes (train earliest)
    if tr.any() and te.any():
        assert d.time_step[tr].float().mean() <= d.time_step[te].float().mean()


def test_elliptic_feature_assembly_df_none():
    d = make_elliptic_fixture(seed=2, n_nodes=400)
    for k, v in elliptic_temporal_masks(d.time_step, d.y).items():
        d[k] = v
    fcfg = {"laplacian_pe_k": 6, "centralities": ["pagerank", "eigenvector"]}
    with tempfile.TemporaryDirectory() as cd:
        d, meta = assemble_features(d, None, fcfg, cd)   # df=None -> Elliptic path
    assert meta["dataset"] == "elliptic"
    assert meta["group_dims"]["raw"] == 166 and meta["group_dims"]["local"] == 0
    assert d.x.size(1) == 166 + 6 + 2
    assert torch.isfinite(d.x).all()


def test_gat_forward_and_attention_on_elliptic_fixture():
    d = make_elliptic_fixture(seed=3, n_nodes=300)
    for k, v in elliptic_temporal_masks(d.time_step, d.y).items():
        d[k] = v
    with tempfile.TemporaryDirectory() as cd:
        d, _ = assemble_features(d, None, {"laplacian_pe_k": 4, "centralities": ["pagerank"]}, cd)
    model = build_model({"model": "gat", "arch": {"hidden_channels": 16, "num_layers": 2,
                                                  "heads": 4, "variant": "gat_v2"}}, d.x.size(1))
    out = model(d)
    assert out.shape == (d.num_nodes, 2) and torch.isfinite(out).all()
    att = model.extract_attention(d)
    assert torch.equal(att["edge_index"], d.edge_index)
    assert att["attention"].shape == (d.edge_index.size(1),)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
