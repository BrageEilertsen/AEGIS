"""k-hop neighborhood extraction with importance-weighted capping for the UI (spec §7.7, §8.4)."""
from __future__ import annotations

import torch
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected

from ml.explain.attention import normalize_importance


def extract_neighborhood(data, node_idx: int, node_scores_all, num_hops: int = 2,
                         max_nodes: int = 400, max_edges: int = 600,
                         edge_importance=None) -> dict:
    """Extract the directed k-hop neighborhood around ``node_idx``, capped by edge importance.

    Node selection uses the UNDIRECTED k-hop set (so upstream payers and downstream receivers are
    both captured — essential for seeing chains and cycles around a flagged transaction), then the
    original DIRECTED edges among those nodes are induced for faithful display and typology. Never
    returns the full graph (spec §8.4); over-budget neighbourhoods are capped greedily by edge
    importance while always keeping the target node.
    """
    n = int(data.num_nodes)
    ei_cpu = data.edge_index.cpu()   # neighborhood extraction is pure graph ops; keep it all on CPU
    undirected = to_undirected(ei_cpu, num_nodes=n)
    subset, _, _, _ = k_hop_subgraph(node_idx, num_hops, undirected, relabel_nodes=False, num_nodes=n)
    subset = subset.cpu()
    sub_ei, _, emask = subgraph(subset, ei_cpu, relabel_nodes=True,
                                num_nodes=n, return_edge_mask=True)
    target_rel = int((subset == node_idx).nonzero(as_tuple=False).item())
    if edge_importance is not None:
        sub_imp = edge_importance.detach().cpu().float()[emask.cpu()]
    else:
        sub_imp = torch.ones(sub_ei.size(1))

    orig_num_nodes, orig_num_edges = int(subset.numel()), int(sub_ei.size(1))
    was_capped = orig_num_nodes > max_nodes or orig_num_edges > max_edges

    if was_capped:
        order = torch.argsort(sub_imp, descending=True).tolist()
        kept_nodes = {target_rel}
        kept_cols: list[int] = []
        for c in order:
            if len(kept_cols) >= max_edges:
                break
            s, d = int(sub_ei[0, c]), int(sub_ei[1, c])
            new = {s, d} - kept_nodes
            if len(kept_nodes) + len(new) > max_nodes:
                continue
            kept_nodes.update({s, d})
            kept_cols.append(c)
        kept_sorted = sorted(kept_nodes)
        remap = {old: i for i, old in enumerate(kept_sorted)}
        if kept_cols:
            sub_ei = torch.stack([
                torch.tensor([remap[int(sub_ei[0, c])] for c in kept_cols], dtype=torch.long),
                torch.tensor([remap[int(sub_ei[1, c])] for c in kept_cols], dtype=torch.long)])
            sub_imp = sub_imp[torch.tensor(kept_cols, dtype=torch.long)]
        else:
            sub_ei, sub_imp = torch.empty((2, 0), dtype=torch.long), torch.zeros(0)
        node_ids = subset[torch.tensor(kept_sorted, dtype=torch.long)]
        target_rel = remap[target_rel]
    else:
        node_ids = subset

    scores = node_scores_all.detach().cpu() if torch.is_tensor(node_scores_all) \
        else torch.as_tensor(node_scores_all)
    return {
        "target_node_id": int(node_idx),
        "node_ids": node_ids.tolist(),
        "edge_index": sub_ei.tolist(),
        "node_labels": data.y.cpu()[node_ids].tolist(),
        "node_scores": scores[node_ids].tolist(),
        "edge_importance": normalize_importance(sub_imp).tolist(),
        "was_capped": bool(was_capped),
        "num_hops": int(num_hops),
        "original_num_nodes": orig_num_nodes,
        "original_num_edges": orig_num_edges,
        "_target_rel": int(target_rel),   # internal: relabeled target index for typology (stripped on serialize)
    }
