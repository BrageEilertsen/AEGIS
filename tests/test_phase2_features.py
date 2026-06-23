"""Phase 2 correctness tests — features, spectral math, cache, leakage, losses, learning.

Runs on the synthetic IBM-AML frame (no download, CPU, login-node safe).
Standalone:  python tests/test_phase2_features.py    |    pytest tests/test_phase2_features.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

from ml.common import (
    build_loss, compute_class_weights, compute_metrics, focal_loss, illicit_scores,
    set_deterministic, weighted_ce_loss,
)
from ml.data.graph import build_transaction_graph
from ml.data.loaders import make_synthetic_aml
from ml.features.assemble import (
    apply_standardization, assemble_features, feature_cache_key, standardize_train_only,
)
from ml.features.spectral import _symmetrize_csr, graph_hash, laplacian_pe
from ml.models import build_model

DT = 24 * 3600


def _graph(n_legit=2000, seed=7):
    df = make_synthetic_aml(n_legit=n_legit, seed=seed)
    data = build_transaction_graph(df, DT, (0.6, 0.2, 0.2))
    return data, df


def _fcfg(k=8):
    return {"laplacian_pe_k": k, "centralities": ["pagerank", "eigenvector"], "max_degree": 2000}


# ---- assembly ----
def test_assemble_shapes_and_finite():
    data, df = _graph()
    with tempfile.TemporaryDirectory() as d:
        data, meta = assemble_features(data, df, _fcfg(8), d)
    g = meta["group_dims"]
    assert data.x.shape == (data.num_nodes, g["raw"] + g["local"] + g["spectral_pe"] + g["centralities"])
    assert g["spectral_pe"] == 8 and g["centralities"] == 2
    assert data.x.dtype == torch.float32
    assert torch.isfinite(data.x).all()


# ---- spectral math ----
def test_trivial_eigenvectors_match_component_count():
    data, _ = _graph()
    A = _symmetrize_csr(data.edge_index, int(data.num_nodes))
    ncomp, _ = connected_components(A, directed=False)
    # PE keeps k NON-trivial vectors; with a tiny k the first kept eigenvalue must be > ~0.
    pe, eigvals = laplacian_pe(data, k=4)
    assert pe.shape == (data.num_nodes, 4)
    eigvals = np.asarray(eigvals)
    assert (eigvals > 1e-8).sum() >= 1, "expected at least one non-trivial eigenvalue"
    assert ncomp >= 1


def test_sign_canonicalization_is_deterministic():
    data, _ = _graph()
    pe1, _ = laplacian_pe(data, k=6)
    pe2, _ = laplacian_pe(data, k=6)
    assert torch.equal(pe1, pe2), "PE must be deterministic (sign canonicalization stable)"


def test_zero_pad_when_few_nontrivial():
    # Tiny graph with fewer non-trivial eigenvectors than k -> zero-padded to width k.
    df = make_synthetic_aml(n_legit=20, seed=1)
    data = build_transaction_graph(df, DT, (0.6, 0.2, 0.2))
    pe, _ = laplacian_pe(data, k=50)
    assert pe.shape == (data.num_nodes, 50)
    assert (pe == 0).any(), "expected zero-padding columns on a tiny graph"


# ---- cache ----
def test_feature_cache_hit_and_key_sensitivity():
    data, df = _graph()
    gh = graph_hash(data)
    k_a = feature_cache_key(gh, _fcfg(8))
    k_b = feature_cache_key(gh, _fcfg(4))           # different k -> different key
    k_c = feature_cache_key(gh, _fcfg(8))           # same -> same key
    assert k_a != k_b and k_a == k_c
    with tempfile.TemporaryDirectory() as d:
        data1, m1 = assemble_features(data, df, _fcfg(8), d)
        pe1 = data1.x.clone()
        data2, df2 = _graph()
        data2, m2 = assemble_features(data2, df2, _fcfg(8), d)
        assert m1["spectral_cache"]["hit"] is False
        assert m2["spectral_cache"]["hit"] is True   # second run reuses the cache


# ---- leakage ----
def test_standardize_train_only_differs_from_global():
    data, df = _graph()
    with tempfile.TemporaryDirectory() as d:
        data, _ = assemble_features(data, df, _fcfg(8), d)
    X = data.x
    _, stats = standardize_train_only(X, data.train_mask)
    train_mean = torch.tensor(stats["mean"])
    global_mean = X.mean(dim=0)
    assert not torch.allclose(train_mean, global_mean, atol=1e-4), \
        "train-only stats must differ from global stats (proves no leakage)"
    # re-applying saved stats reproduces the transform
    re = apply_standardization(X, stats)
    direct = (X - train_mean) / (torch.tensor(stats["std"]) + 1e-8)
    assert torch.allclose(re, direct, atol=1e-5)


# ---- losses ----
def test_loss_numerics():
    torch.manual_seed(0)
    logits = torch.randn(100, 2, requires_grad=True)
    targets = torch.randint(0, 2, (100,))
    alpha = torch.tensor([0.5, 5.0])
    fl = focal_loss(logits, targets, alpha, gamma=2.0)
    wce = weighted_ce_loss(logits, targets, alpha)
    assert torch.isfinite(fl) and torch.isfinite(wce)
    fl.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # confident-correct -> ~0 ; confident-wrong -> large
    correct = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    wrong = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
    tgt = torch.tensor([0, 1])
    a = torch.tensor([1.0, 1.0])
    assert focal_loss(correct, tgt, a) < 1e-3
    assert focal_loss(wrong, tgt, a) > 1.0


def test_class_weights_train_only():
    data, _ = _graph()
    w = compute_class_weights(data.y, data.train_mask)
    assert w.shape == (2,) and (w > 0).all()


# ---- end-to-end learning (k=0 isolates generalizable signal from PE memorization) ----
def test_gcn_learns_generalizable_signal_without_pe():
    data, df = _graph(n_legit=4000, seed=0)
    cfg = {"model": "gcn", "arch": {"hidden_channels": 32, "num_layers": 2, "dropout": 0.5}}
    with tempfile.TemporaryDirectory() as d:
        data, _ = assemble_features(data, df, _fcfg(0), d)   # k=0: no Laplacian PE
    set_deterministic(0)
    model = build_model(cfg, data.x.size(1))
    cw = compute_class_weights(data.y, data.train_mask).clamp(max=5.0)
    loss_fn = build_loss({"loss": "weighted_ce"}, cw)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)
    y = data.y.numpy()
    best_val = -1.0
    for _ in range(150):
        model.train(); opt.zero_grad()
        out = model(data); loss_fn(out[data.train_mask], data.y[data.train_mask]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = illicit_scores(model(data))
        va = compute_metrics(y[data.val_mask.numpy()], s[data.val_mask.numpy()])["pr_auc"]
        best_val = max(best_val, va)
    test = compute_metrics(y[data.test_mask.numpy()], s[data.test_mask.numpy()])
    base_rate = float((y[data.test_mask.numpy()] == 1).mean())
    assert test["pr_auc"] > 5 * base_rate, \
        f"GCN should beat base rate {base_rate:.3f}; got test PR-AUC {test['pr_auc']:.3f}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
