"""GAT native attention as faithful edge importance (spec §7.7)."""
from __future__ import annotations

import torch


def extract_gat_attention(model, data, layer: str = "last", head_reduce: str = "mean") -> dict:
    """Thin wrapper over ``GAT.extract_attention``. Returns {edge_index[2,E], attention[E], node_logits}."""
    if not getattr(model, "supports_attention", False):
        raise ValueError("model does not support attention extraction (not a GAT)")
    att = model.extract_attention(data, layer=layer, head_reduce=head_reduce)
    return {"edge_index": att["edge_index"], "attention": att["attention"],
            "node_logits": att["node_logits"]}


def normalize_importance(x: torch.Tensor) -> torch.Tensor:
    """Min-max scale to [0, 1] for UI comparability; safe on empty or all-equal input."""
    x = x.float()
    if x.numel() == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)
