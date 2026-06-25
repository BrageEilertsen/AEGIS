"""GNNExplainer integration: minimal-subgraph edge mask + feature mask (spec §7.7)."""
from __future__ import annotations

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from ml.explain.adapter import ExplainerAdapter, model_config


def run_gnnexplainer(model, data, node_idx: int, epochs: int = 120, lr: float = 0.01,
                     num_hops: int = 2) -> dict:
    """Explain the model's prediction for ``node_idx``.

    Runs on the node's ``num_hops`` computation subgraph rather than the full graph: for a
    k-layer GNN the prediction depends only on the k-hop neighbourhood, so this is faithful and
    dramatically faster / lighter (the full graph has ~180k nodes; the subgraph is tiny). The
    learned masks are scattered back to full-graph dimensions so callers are unchanged.

    Returns {node_mask[N, F], edge_mask[E], edge_index[2, E]}.
    """
    model.eval()
    n = int(data.num_nodes)
    subset, sub_edge_index, mapping, hop_edge_mask = k_hop_subgraph(
        int(node_idx), num_hops, data.edge_index, relabel_nodes=True, num_nodes=n)

    explainer = Explainer(
        model=ExplainerAdapter(model),
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config(),
    )
    expl = explainer(data.x[subset], sub_edge_index, index=int(mapping))

    # Scatter the subgraph masks back onto full-graph index space.
    node_mask = torch.zeros((n, data.x.size(1)), dtype=expl.node_mask.dtype)
    node_mask[subset] = expl.node_mask.detach()
    edge_mask = torch.zeros(data.edge_index.size(1), dtype=expl.edge_mask.dtype)
    if hop_edge_mask.any():
        edge_mask[hop_edge_mask] = expl.edge_mask.detach()
    return {"node_mask": node_mask, "edge_mask": edge_mask, "edge_index": data.edge_index}


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two per-edge importance vectors (faithfulness diagnostic)."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return float("nan")
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    if a.size < 2:
        return float("nan")
    rho = spearmanr(a, b)[0]                     # [0] = correlation (stable across scipy versions)
    return float(rho) if rho == rho else 0.0     # NaN -> 0 (e.g. constant input)
