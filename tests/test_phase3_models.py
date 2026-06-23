"""Phase 3 correctness tests — GraphSAGE / GAT models, attention, factory, sampling fallback.

Login-CPU, synthetic only. Standalone: python tests/test_phase3_models.py | pytest tests/test_phase3_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch_geometric.data import Data

from ml.models import GAT, GCN, GraphSAGE, build_model
from ml.train import _make_neighbor_loader

MODELS = ["gcn", "graphsage", "gat"]


def _toy_data(n=60, f=12, e=240, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = Data(x=torch.randn(n, f, generator=g),
             edge_index=torch.randint(0, n, (2, e), generator=g),
             y=torch.randint(0, 2, (n,), generator=g))
    d.num_nodes = n
    d.train_mask = torch.zeros(n, dtype=torch.bool); d.train_mask[: int(0.6 * n)] = True
    return d


def _cfg(model):
    arch = {"hidden_channels": 16, "num_layers": 2, "dropout": 0.5}
    if model == "gat":
        arch.update(heads=4, attention_dropout=0.3, variant="gat_v2")
    if model == "graphsage":
        arch.update(aggregator="mean")
    return {"model": model, "arch": arch}


def test_build_model_dispatch_all():
    d = _toy_data()
    assert isinstance(build_model(_cfg("gcn"), d.x.size(1)), GCN)
    assert isinstance(build_model(_cfg("graphsage"), d.x.size(1)), GraphSAGE)
    assert isinstance(build_model(_cfg("gat"), d.x.size(1)), GAT)
    try:
        build_model({"model": "nope"}, d.x.size(1)); assert False, "should raise"
    except NotImplementedError:
        pass


def test_forward_shape_fullbatch():
    d = _toy_data()
    for m in MODELS:
        model = build_model(_cfg(m), d.x.size(1))
        out = model(d)
        assert out.shape == (d.num_nodes, 2), f"{m} bad shape {out.shape}"
        assert torch.isfinite(out).all(), f"{m} non-finite logits"


def test_gat_attention_aligned_with_edges():
    d = _toy_data()
    model = build_model(_cfg("gat"), d.x.size(1))
    att = model.extract_attention(d, layer="last", head_reduce="mean")
    # add_self_loops=False -> returned edge_index is exactly the input edge set.
    assert att["edge_index"].shape == d.edge_index.shape
    assert torch.equal(att["edge_index"], d.edge_index)
    assert att["attention"].shape == (d.edge_index.size(1),)
    assert att["node_logits"].shape == (d.num_nodes, 2)


def test_gat_attention_perhead_and_reduce():
    d = _toy_data()
    model = build_model(_cfg("gat"), d.x.size(1))
    raw = model.extract_attention(d, layer=0, head_reduce="none")["attention"]
    assert raw.shape == (d.edge_index.size(1), 4), f"per-head shape {raw.shape}"
    mean = model.extract_attention(d, layer=0, head_reduce="mean")["attention"]
    mx = model.extract_attention(d, layer=0, head_reduce="max")["attention"]
    assert mean.shape == (d.edge_index.size(1),)
    assert not torch.allclose(mean, mx)


def test_gat_attention_layer_selection():
    d = _toy_data()
    model = build_model(_cfg("gat"), d.x.size(1))
    last = model.extract_attention(d, layer="last")
    alll = model.extract_attention(d, layer="all")
    assert isinstance(last["attention"], torch.Tensor)
    assert isinstance(last["num_heads"], int) and last["num_heads"] == 1   # output layer: heads=1
    assert isinstance(alll["attention"], list) and len(alll["attention"]) == 2
    # num_heads must match the per-layer structure (hidden heads=4, output heads=1)
    assert alll["num_heads"] == [4, 1]


def test_attention_unsupported_on_gcn_sage():
    assert GCN.supports_attention is False
    assert GraphSAGE.supports_attention is False
    assert GAT.supports_attention is True
    assert not hasattr(build_model(_cfg("gcn"), 12), "extract_attention")


def test_minibatch_seed_slicing_logic():
    # A NeighborLoader batch places the `batch_size` seed nodes first; the loss uses logits[:bs].
    d = _toy_data(n=30, f=8, e=80)
    model = build_model(_cfg("graphsage"), d.x.size(1))
    bs = 5
    out = model(d)
    seed_logits, seed_y = out[:bs], d.y[:bs]
    assert seed_logits.shape == (bs, 2) and seed_y.shape == (bs,)


def test_sampling_fallback_when_unavailable():
    # On the login node (no pyg-lib/torch-sparse) NeighborLoader can't build -> graceful None.
    d = _toy_data()
    cfg = {"train": {"neighbor_sampling": {"num_neighbors": [5, 5], "batch_size": 8}}}
    loader = _make_neighbor_loader(d, cfg)
    assert loader is None, "expected full-batch fallback when sampling backend is absent"
    assert _make_neighbor_loader(d, {"train": {}}) is None  # not configured -> None too


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
