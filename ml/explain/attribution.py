"""Feature attribution: turn a GNNExplainer node mask into named, grouped top features (spec §7.7)."""
from __future__ import annotations

import torch


def _group_of(feat_idx: int, group_dims: dict) -> str:
    """Map a feature index to its group via cumulative offsets (raw|local|spectral_pe|centralities)."""
    offset = 0
    for group in ("raw", "local", "spectral_pe", "centralities"):
        width = int(group_dims.get(group, 0))
        if feat_idx < offset + width:
            return group
        offset += width
    return "unknown"


def attribute_features(node_mask: torch.Tensor, feature_meta: dict, node_idx: int,
                       top_k: int = 8) -> list[dict]:
    """Top-k features for the flagged node, by |mask importance|, with names + groups.

    node_mask is [N, F]; we take the flagged node's row. Column names and group offsets come from
    the checkpoint's feature_meta (the same columns assembled at train time).
    """
    columns = feature_meta["columns"]
    group_dims = feature_meta.get("group_dims", {})
    assert len(columns) == node_mask.size(1), \
        f"feature_meta columns ({len(columns)}) != node_mask width ({node_mask.size(1)})"
    row = node_mask[node_idx].abs()
    k = min(top_k, row.numel())
    top = torch.topk(row, k)
    out = []
    for imp, idx in zip(top.values.tolist(), top.indices.tolist()):
        out.append({"column_name": columns[idx], "feature_index": int(idx),
                    "importance": float(imp), "group": _group_of(idx, group_dims)})
    return out
