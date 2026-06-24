"""Explainability subsystem (spec §7.7) — required, not optional.

For any flagged node, ``explain_node`` produces a faithful, structured explanation:
edge importance (GAT native attention, or GNNExplainer edge mask for GCN/GraphSAGE), feature
attribution (GNNExplainer node mask, named via feature_meta), a focused k-hop neighbourhood
subgraph, and a matched laundering typology — assembled into the versioned ExplanationContract
that the Java backend → Angular UI consume. Explanations are tied to what the model used and their
limitations are stated honestly (the ``faithfulness`` block). PGExplainer is a documented stretch.
"""
from __future__ import annotations

from datetime import datetime, timezone

import torch

from ml.common import illicit_scores
from ml.explain.attention import extract_gat_attention, normalize_importance
from ml.explain.attribution import attribute_features
from ml.explain.contract import ExplanationContract
from ml.explain.gnnexplainer import run_gnnexplainer, spearman
from ml.explain.subgraph import extract_neighborhood
from ml.explain.typology import match_typology

_FAITHFULNESS_NOTE = (
    "Edge importance for GAT is the model's own attention (what it computed), not a post-hoc "
    "rationalization. GNNExplainer feature/edge masks identify a minimal sufficient subgraph and "
    "are correlational, not causal. Typology on unlabeled data is a structural heuristic."
)


def load_checkpoint_and_model(checkpoint_path):
    """Load a trained checkpoint -> (model.eval(), config, in_channels, feature_meta, seed)."""
    from ml.models import build_model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, ckpt["in_channels"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, int(ckpt["in_channels"]), ckpt["feature_meta"], int(ckpt.get("seed", 42))


def explain_node(model, data, node_idx: int, feature_meta: dict, *, model_type: str,
                 num_hops: int = 2, max_nodes: int = 400, gnnex_epochs: int = 120,
                 method: str = "auto", top_k_edges: int = 12, top_k_features: int = 8,
                 pattern_labels=None) -> ExplanationContract:
    """Produce the explanation contract for one flagged node (spec §7.7)."""
    model.eval()
    # Models that symmetrize edges inside forward() (e.g. GCN's to_undirected) would desync
    # GNNExplainer's edge-mask from the propagated edge count. Run the explanation on the already-
    # undirected graph so the model's transform is idempotent and edges stay consistent throughout.
    if getattr(model, "symmetrizes_edges", False):
        from torch_geometric.utils import to_undirected
        data = data.clone()
        data.edge_index = to_undirected(data.edge_index, num_nodes=int(data.num_nodes))
    with torch.no_grad():
        logits = model(data)
    scores_all = torch.as_tensor(illicit_scores(logits))
    score = float(scores_all[node_idx])
    predicted = int(logits[node_idx].argmax())

    # Feature attribution always comes from GNNExplainer's node mask; its edge mask is also the
    # edge-importance source for non-attention models and a faithfulness cross-check for GAT.
    gnn = run_gnnexplainer(model, data, node_idx, epochs=gnnex_epochs)
    node_mask, gnn_edge_mask = gnn["node_mask"], gnn["edge_mask"]

    supports_attn = getattr(model, "supports_attention", False)
    use_attention = (method == "attention") or (method == "auto" and supports_attn)
    if use_attention and supports_attn:
        att = extract_gat_attention(model, data)
        raw_edge_imp = att["attention"]
        source, fmethod = "attention", "gat_attention"
        rho = spearman(att["attention"], gnn_edge_mask)
    else:
        raw_edge_imp = gnn_edge_mask
        source, fmethod, rho = "gnnexplainer", "gnnexplainer", None

    subgraph = extract_neighborhood(data, node_idx, scores_all, num_hops=num_hops,
                                    max_nodes=max_nodes, edge_importance=raw_edge_imp)

    # top_edges drawn from the (capped) subgraph -> guaranteed inside node_ids (no hallucination).
    ei, imp, node_ids = subgraph["edge_index"], subgraph["edge_importance"], subgraph["node_ids"]
    order = sorted(range(len(imp)), key=lambda c: imp[c], reverse=True)[:top_k_edges]
    top_edges = [{"source_node": int(node_ids[ei[0][c]]), "target_node": int(node_ids[ei[1][c]]),
                  "importance": round(float(imp[c]), 4), "source": source} for c in order]

    top_features = attribute_features(node_mask, feature_meta, node_idx, top_k=top_k_features)
    typology = match_typology(subgraph, pattern_labels)
    faithfulness = {"method": fmethod, "edge_importance_source": source,
                    "attention_gnnexplainer_spearman": rho, "note": _FAITHFULNESS_NOTE}

    return ExplanationContract(
        node_id=int(node_idx), score=round(score, 6), predicted_label=predicted,
        top_edges=top_edges, top_features=top_features, matched_typology=typology,
        neighborhood_subgraph=subgraph, faithfulness=faithfulness,
        model_version=model_type, feature_spec_version=feature_meta.get("feature_spec_version", "?"),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


__all__ = ["explain_node", "load_checkpoint_and_model", "ExplanationContract"]
