"""GNNExplainer integration: minimal-subgraph edge mask + feature mask (spec §7.7)."""
from __future__ import annotations

import torch
from torch_geometric.explain import Explainer, GNNExplainer

from ml.explain.adapter import ExplainerAdapter, model_config


def run_gnnexplainer(model, data, node_idx: int, epochs: int = 120, lr: float = 0.01) -> dict:
    """Explain the model's prediction for ``node_idx``.

    Returns {node_mask[N, F], edge_mask[E], edge_index[2, E]}. The adapter feeds the model the
    exact (x, edge_index); GNNExplainer learns a minimal sufficient mask for the target node.
    """
    model.eval()
    explainer = Explainer(
        model=ExplainerAdapter(model),
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config(),
    )
    expl = explainer(data.x, data.edge_index, index=node_idx)
    return {"node_mask": expl.node_mask.detach(), "edge_mask": expl.edge_mask.detach(),
            "edge_index": data.edge_index}


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
