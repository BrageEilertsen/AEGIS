"""Phase 4 correctness tests — explainability (adapter, GNNExplainer, attribution, subgraph,
typology, contract). Login-CPU, synthetic. Uses the injected typologies as a ground-truth oracle.

Standalone: python tests/test_phase4_explain.py | pytest tests/test_phase4_explain.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ml.data.graph import build_transaction_graph
from ml.data.loaders import make_synthetic_aml
from ml.explain import explain_node
from ml.explain.adapter import ExplainerAdapter
from ml.explain.attention import extract_gat_attention, normalize_importance
from ml.explain.attribution import attribute_features
from ml.explain.contract import ExplanationContract
from ml.explain.gnnexplainer import run_gnnexplainer
from ml.explain.subgraph import extract_neighborhood
from ml.explain.typology import match_typology
from ml.features.assemble import assemble_features
from ml.models import build_model

DT = 24 * 3600
_CACHE = tempfile.mkdtemp()


@lru_cache(maxsize=1)
def _setup():
    df, patterns = make_synthetic_aml(n_legit=800, seed=0, return_pattern_labels=True)
    data = build_transaction_graph(df, DT)
    fcfg = {"laplacian_pe_k": 6, "centralities": ["pagerank", "eigenvector"], "max_degree": 2000}
    data, meta = assemble_features(data, df, fcfg, _CACHE)
    model = build_model({"model": "gat", "arch": {"hidden_channels": 16, "num_layers": 2,
                                                  "heads": 4, "variant": "gat_v2"}}, data.x.size(1))
    model.eval()
    return data, df, patterns, meta, model


def _first_node(patterns, label, y):
    for i, (p, yi) in enumerate(zip(patterns, y.tolist())):
        if p == label and yi == 1:
            return i
    raise AssertionError(f"no node with pattern {label}")


def _sub(edge_index, target_rel):
    nodes = sorted(set(edge_index[0]) | set(edge_index[1]) | {target_rel})
    return {"edge_index": edge_index, "_target_rel": target_rel,
            "target_node_id": 0, "node_ids": nodes}


# ---- adapter ----
def test_adapter_logits_match_model():
    data, *_ = _setup()
    for m in ("gcn", "graphsage", "gat"):
        arch = {"hidden_channels": 16, "num_layers": 2}
        if m == "gat":
            arch.update(heads=4, variant="gat_v2")
        model = build_model({"model": m, "arch": arch}, data.x.size(1)).eval()
        with torch.no_grad():
            direct = model(data)
            via = ExplainerAdapter(model)(data.x, data.edge_index)
        assert torch.allclose(direct, via, atol=1e-5), f"{m} adapter mismatch"


# ---- attention ----
def test_gat_attention_extracted():
    data, _, _, _, model = _setup()
    att = extract_gat_attention(model, data)
    E = data.edge_index.size(1)
    assert att["attention"].shape == (E,)
    assert torch.equal(att["edge_index"], data.edge_index)
    norm = normalize_importance(att["attention"])
    assert float(norm.min()) >= 0.0 and float(norm.max()) <= 1.0


# ---- gnnexplainer ----
def test_gnnexplainer_mask_shapes():
    data, _, patterns, _, model = _setup()
    node = _first_node(patterns, "layering_chain", data.y)
    out = run_gnnexplainer(model, data, node, epochs=15)
    assert out["node_mask"].shape == (data.num_nodes, data.x.size(1))
    assert out["edge_mask"].shape == (data.edge_index.size(1),)


# ---- feature attribution ----
def test_feature_attribution_names_and_groups():
    data, _, _, meta, _ = _setup()
    nm = torch.rand(data.num_nodes, data.x.size(1))
    feats = attribute_features(nm, meta, node_idx=0, top_k=5)
    assert len(feats) == 5
    cols = set(meta["columns"])
    for f in feats:
        assert f["column_name"] in cols
        assert f["group"] in {"raw", "local", "spectral_pe", "centralities"}


# ---- subgraph ----
def test_subgraph_cap_and_target_present():
    data, *_ = _setup()
    scores = torch.zeros(data.num_nodes)
    deg = torch.bincount(torch.cat([data.edge_index[0], data.edge_index[1]]),
                         minlength=data.num_nodes)
    hub = int(deg.argmax())                                   # busiest node -> large neighbourhood
    full = extract_neighborhood(data, hub, scores, num_hops=2, max_nodes=400, max_edges=600)
    assert full["target_node_id"] == hub and hub in full["node_ids"]
    assert len(full["edge_index"][0]) == len(full["edge_importance"])
    capped = extract_neighborhood(data, hub, scores, num_hops=2, max_nodes=4, max_edges=4)
    assert hub in capped["node_ids"] and len(capped["node_ids"]) <= 4
    assert len(capped["edge_index"][0]) == len(capped["edge_importance"]) <= 4
    # was_capped iff the full neighbourhood exceeded a budget
    assert capped["was_capped"] == (len(full["node_ids"]) > 4 or len(full["edge_index"][0]) > 4)


# ---- typology scorers (hand-built shapes, deterministic) ----
def test_typology_scorer_fan_out():
    sg = _sub([[0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]], target_rel=0)
    assert match_typology(sg)["label"] == "fan_out"


def test_typology_scorer_layering_chain():
    sg = _sub([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], target_rel=2)
    assert match_typology(sg)["label"] == "layering_chain"


def test_typology_scorer_circular():
    sg = _sub([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], target_rel=0)
    out = match_typology(sg)
    assert out["label"] == "circular" and out["confidence"] == 1.0


def test_typology_handles_cycle_without_blowup():
    # A 2-cycle (e.g. same-minute reciprocal edges) must not hang/error; circular wins the tie.
    sg = _sub([[0, 1, 1], [1, 0, 2]], target_rel=0)
    out = match_typology(sg)
    assert out["label"] == "circular"


def test_subgraph_handles_gpu_data():
    # extract_neighborhood must run regardless of data's device (it works on CPU internally).
    if not torch.cuda.is_available():
        return
    data, *_ = _setup()
    gpu = data.to("cuda")
    sg = extract_neighborhood(gpu, 0, torch.zeros(data.num_nodes), num_hops=2)
    assert sg["target_node_id"] == 0 and 0 in sg["node_ids"]


# ---- typology oracle on the synthetic graph ----
# Both injected "layering_chain" and "circular" account patterns become forward PATHS in the
# time-ordered transaction-node graph, so the detectable typology there is the layering chain.
def test_typology_layering_chain_oracle():
    data, _, patterns, _, _ = _setup()
    node = _first_node(patterns, "layering_chain", data.y)
    sg = extract_neighborhood(data, node, torch.zeros(data.num_nodes), num_hops=3)
    out = match_typology(sg, pattern_labels=patterns)
    assert out["label"] == "layering_chain", f"got {out['label']} scores={out['scores']}"
    assert out["confidence"] >= 0.6 and out["ground_truth"] == "layering_chain"


def test_typology_legit_precision():
    data, _, patterns, _, _ = _setup()
    # an isolated/low-degree legit node should not be confidently labeled a typology
    legit = next(i for i, p in enumerate(patterns) if p == "legit")
    sg = extract_neighborhood(data, legit, torch.zeros(data.num_nodes), num_hops=2)
    out = match_typology(sg)
    assert out["label"] == "unknown" or out["confidence"] < 0.5


# ---- end-to-end contract ----
def test_explain_node_top_edges_inside_subgraph():
    data, _, patterns, meta, model = _setup()
    node = _first_node(patterns, "layering_chain", data.y)
    c = explain_node(model, data, node, meta, model_type="gat", gnnex_epochs=15)
    node_set = set(c.neighborhood_subgraph["node_ids"])
    for e in c.top_edges:
        assert e["source_node"] in node_set and e["target_node"] in node_set


def test_contract_json_roundtrip_and_version():
    data, _, patterns, meta, model = _setup()
    node = _first_node(patterns, "layering_chain", data.y)
    c = explain_node(model, data, node, meta, model_type="gat", gnnex_epochs=15)
    s = c.to_json()
    d = json.loads(s)
    assert d["schema_version"] == "1.0"
    assert "_target_rel" not in d["neighborhood_subgraph"]   # internal field stripped
    rt = ExplanationContract.from_dict(d)
    assert rt.node_id == c.node_id and rt.matched_typology["label"] == c.matched_typology["label"]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
