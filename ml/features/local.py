"""Local-graph node features (spec §7.2, group 2) — Phase 2.

All vectorized on the sparse ``edge_index`` (O(N+E)). No networkx on the full graph and no
betweenness (O(nm), intractable at ~7M nodes — explicitly deferred). Clustering uses a bounded
triangle count with a hub guard. Returns **unstandardized** features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import scatter

from ml.data.loaders import COL_AMT_PAID


def in_out_degree(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(out_degree, in_degree) via bincount on the edge endpoints. O(E)."""
    out_deg = torch.bincount(edge_index[0], minlength=n).float()
    in_deg = torch.bincount(edge_index[1], minlength=n).float()
    return out_deg, in_deg


def _symmetric_adj(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """0/1 symmetric adjacency (max(A, Aᵀ)) with no self-loops, as scipy CSR."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data = np.ones(row.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0
    return A


def clustering_coefficient(edge_index: torch.Tensor, n: int, max_degree: int = 2000) -> torch.Tensor:
    """Undirected local clustering coefficient via a bounded triangle count.

    t_i = 0.5 * sum_j (A^2)_{ij} A_{ij};  C_i = 2 t_i / (d_i (d_i - 1)).
    Nodes with degree > ``max_degree`` are excluded from the A@A product (their rows/cols are
    zeroed) and assigned C_i = 0, bounding fill-in to O(sum_{non-hub} d_i^2). Exact on the
    synthetic graph; tractable on the real one.
    """
    A = _symmetric_adj(edge_index, n)
    deg = np.asarray(A.sum(axis=1)).ravel()
    hub = deg > max_degree
    if hub.any():
        keep = (~hub).astype(np.float32)
        D = sp.diags(keep)
        A = D @ A @ D  # zero out hub rows and columns
        A.eliminate_zeros()
    A2 = A @ A
    tri = 0.5 * np.asarray(A2.multiply(A).sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        coef = np.where(deg > 1, 2.0 * tri / (deg * (deg - 1.0)), 0.0)
    coef[hub] = 0.0
    return torch.tensor(np.nan_to_num(coef), dtype=torch.float)


def reciprocity(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Per-node fraction of out-edges (u->v) whose reverse (v->u) also exists. O(E)."""
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    eid = src * n + dst
    rev = dst * n + src
    is_recip = np.isin(rev, eid).astype(np.float32)            # does the reverse edge exist?
    recip_sum = np.bincount(src, weights=is_recip, minlength=n)
    out_cnt = np.bincount(src, minlength=n).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(out_cnt > 0, recip_sum / out_cnt, 0.0)
    return torch.tensor(np.nan_to_num(r), dtype=torch.float)


def neighbor_amount_stats(edge_index: torch.Tensor, n: int,
                          node_amount: torch.Tensor) -> torch.Tensor:
    """[N, 3] mean / max / std of neighbours' (log1p) amounts over incident edges. O(E).

    Treats the graph as undirected for neighbourhood: for edge (i, j), j's amount contributes to
    i and i's amount to j. std via E[x^2] - E[x]^2.
    """
    src, dst = edge_index[0], edge_index[1]
    idx = torch.cat([src, dst])
    vals = torch.cat([node_amount[dst], node_amount[src]])
    mean = scatter(vals, idx, dim=0, dim_size=n, reduce="mean")
    meansq = scatter(vals * vals, idx, dim=0, dim_size=n, reduce="mean")
    mx = scatter(vals, idx, dim=0, dim_size=n, reduce="max")
    var = (meansq - mean * mean).clamp_min(0.0)
    std = var.sqrt()
    out = torch.stack([mean, mx, std], dim=1)
    return torch.nan_to_num(out)


def build_local_features(data, df: pd.DataFrame, max_degree: int = 2000
                         ) -> tuple[torch.Tensor, list[str]]:
    """Return ([N, 8] float tensor, names). NOT standardized.

    Columns: log1p(in_deg), log1p(out_deg), log1p(total_deg), clustering_coef, reciprocity,
    neighbor_amt_mean, neighbor_amt_max, neighbor_amt_std.
    """
    n = int(data.num_nodes)
    ei = data.edge_index
    out_deg, in_deg = in_out_degree(ei, n)
    total_deg = out_deg + in_deg
    clustering = clustering_coefficient(ei, n, max_degree=max_degree)
    recip = reciprocity(ei, n)
    node_amount = torch.tensor(np.log1p(df[COL_AMT_PAID].to_numpy(dtype=float)), dtype=torch.float)
    namt = neighbor_amount_stats(ei, n, node_amount)

    feats = torch.stack([
        torch.log1p(in_deg), torch.log1p(out_deg), torch.log1p(total_deg),
        clustering, recip, namt[:, 0], namt[:, 1], namt[:, 2],
    ], dim=1)
    names = ["log_in_deg", "log_out_deg", "log_total_deg", "clustering_coef",
             "reciprocity", "neighbor_amt_mean", "neighbor_amt_max", "neighbor_amt_std"]
    return feats.float(), names
