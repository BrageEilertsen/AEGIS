"""Graph construction from IBM-AML transactions (spec §6.3, §6.4) — Phase 1.

Two graph views over the same transactions:

1. Transaction-as-node (primary, matches Elliptic): each transaction is a node; a directed edge
   v_i -> v_j exists when the receiver account of v_i is the sender account of v_j and the time
   gap satisfies ``0 <= t_j - t_i <= Δt`` (Δt configurable). Captures money flow. This is the
   graph the GNN trains on for node classification.
2. Account-as-node (for visualization clarity): accounts are nodes; each transaction is a
   directed edge sender -> receiver carrying amount / currency / format / time as edge features.
   More legible for the analyst UI.

Also: a temporal 60/20/20 split by transaction time (NOT random — prevents leakage), basic
stats, and a content-addressed cache so a built graph is reused across runs.

Phase 1 attaches a *minimal* raw feature matrix ``x`` so the ``Data`` object is immediately
usable; the full raw + local + spectral feature pipelines land in Phase 2 (``ml/features/``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .loaders import (
    COL_AMT_PAID, COL_AMT_RECEIVED, COL_CUR_PAID, COL_PAYMENT_FORMAT,
)


# --------------------------------------------------------------------------------------------
# Minimal Phase-1 node features (Phase 2 replaces/augments these via ml/features/)
# --------------------------------------------------------------------------------------------
def _codes(series: pd.Series) -> np.ndarray:
    """Stable integer codes for a categorical column."""
    return series.astype("category").cat.codes.to_numpy()


def _minimal_tx_features(df: pd.DataFrame) -> torch.Tensor:
    """A small, dependency-free raw feature matrix for the transaction-as-node graph.

    Columns: log1p(amount paid), log1p(amount received), payment-format code, currency code,
    time-of-day fraction, day index. Standardized to zero mean / unit variance. Intentionally
    minimal — the real feature engineering (raw + local + spectral) is Phase 2.
    """
    amt_paid = np.log1p(df[COL_AMT_PAID].to_numpy(dtype=float))
    amt_recv = np.log1p(df[COL_AMT_RECEIVED].to_numpy(dtype=float))
    fmt = _codes(df[COL_PAYMENT_FORMAT]).astype(float)
    cur = _codes(df[COL_CUR_PAID]).astype(float)
    t = df["t"].to_numpy(dtype=float)
    tod = (t % 86400) / 86400.0          # time-of-day fraction
    day = t // 86400                     # day index

    feats = np.stack([amt_paid, amt_recv, fmt, cur, tod, day], axis=1)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats - mean) / std
    return torch.tensor(feats, dtype=torch.float)


# --------------------------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------------------------
def subsample_legitimate(df: pd.DataFrame, ratio: float | None, seed: int = 42) -> pd.DataFrame:
    """Keep ALL illicit transactions; subsample legitimate ones to ``ratio * n_illicit``.

    ``ratio=None`` returns the frame unchanged. Used to keep large graphs tractable while
    preserving every positive (spec §6.3). Realistic ratios must be preserved in the held-out
    evaluation set — callers should subsample BEFORE the temporal split only when documented.
    """
    if ratio is None:
        return df
    illicit = df[df["label"] == 1]
    legit = df[df["label"] == 0]
    n_keep = min(len(legit), int(ratio * len(illicit)))
    legit_keep = legit.sample(n=n_keep, random_state=seed)
    out = pd.concat([illicit, legit_keep]).sort_values("t", kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------------
# Temporal split (60/20/20 by transaction time)
# --------------------------------------------------------------------------------------------
def temporal_split_masks(
    t: np.ndarray, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> dict[str, torch.Tensor]:
    """Boolean train/val/test masks split by time quantile (earlier -> train).

    Splitting on the time quantile (not the raw time range) keeps the proportions exact even
    when transactions cluster in time.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "split ratios must sum to 1"
    order = np.argsort(t, kind="stable")
    n = len(t)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    def mask(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    return {"train_mask": mask(train_idx), "val_mask": mask(val_idx), "test_mask": mask(test_idx)}


# --------------------------------------------------------------------------------------------
# Transaction-as-node graph
# --------------------------------------------------------------------------------------------
def build_transaction_graph(
    df: pd.DataFrame,
    delta_t_seconds: int,
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    max_out_per_in: int | None = None,
) -> Data:
    """Build the transaction-as-node flow graph.

    An edge i -> j is added when ``dst_account[i] == src_account[j]`` (money continues through
    the same account) and ``0 <= t[j] - t[i] <= delta_t_seconds``.

    Implementation: group transactions by the account that links them. For each account, the
    *incoming* transactions (where it is the receiver) connect to *outgoing* transactions (where
    it is the sender) that fall within the Δt window. Outgoing times are sorted per account so
    the window is found with binary search — O(E log n) rather than O(n^2).

    ``max_out_per_in`` optionally caps the out-edges generated per incoming transaction, a guard
    against pathological hub accounts; ``None`` means no cap.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    t = df["t"].to_numpy()
    src = df["src_account"].to_numpy()
    dst = df["dst_account"].to_numpy()

    # For each account: indices of transactions where it is the sender (outgoing).
    out_by_account: dict[str, list[int]] = {}
    for i in range(n):
        out_by_account.setdefault(src[i], []).append(i)
    # Sort each account's outgoing transactions by time ONCE, and cache the sorted index array
    # and its time array. Building these per account up front (not per incoming transaction)
    # keeps the windowing at O(E log) instead of Σ_account(in_deg·out_deg) — the latter is
    # quadratic on hub accounts (mules/shells), which are common in AML data.
    out_idx_by_account: dict[str, np.ndarray] = {}
    out_t_by_account: dict[str, np.ndarray] = {}
    for acct, idxs in out_by_account.items():
        arr = np.array(idxs, dtype=np.int64)
        arr = arr[np.argsort(t[arr], kind="stable")]
        out_idx_by_account[acct] = arr
        out_t_by_account[acct] = t[arr]

    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        idx_arr = out_idx_by_account.get(dst[i])
        if idx_arr is None:
            continue
        times = out_t_by_account[dst[i]]
        lo = np.searchsorted(times, t[i], side="left")                    # t_j >= t_i
        hi = np.searchsorted(times, t[i] + delta_t_seconds, side="right")  # t_j <= t_i + Δt
        window = idx_arr[lo:hi]
        if max_out_per_in is not None and window.size > max_out_per_in:
            window = window[:max_out_per_in]
        for j in window.tolist():
            if j != i:
                rows.append(i)
                cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=_minimal_tx_features(df),
        edge_index=edge_index,
        y=torch.tensor(df["label"].to_numpy(), dtype=torch.long),
    )
    data.t = torch.tensor(t, dtype=torch.long)
    data.num_nodes = n
    for k, v in temporal_split_masks(t, split_ratios).items():
        data[k] = v
    data.graph_kind = "transaction_as_node"
    data.delta_t_seconds = int(delta_t_seconds)
    return data


# --------------------------------------------------------------------------------------------
# Account-as-node graph (for the analyst-facing visualization)
# --------------------------------------------------------------------------------------------
def build_account_graph(df: pd.DataFrame) -> Data:
    """Build the account-as-node graph: accounts are nodes, transactions are directed edges.

    Edge features: log1p(amount paid), currency code, payment-format code, time. A node's label
    is 1 if the account took part in *any* illicit transaction (as sender or receiver) — a
    derived, heuristic label for visualization, not the supervised training target.
    """
    df = df.reset_index(drop=True)
    accounts = pd.Index(pd.unique(pd.concat([df["src_account"], df["dst_account"]])))
    idx_of = {a: i for i, a in enumerate(accounts)}

    src_idx = df["src_account"].map(idx_of).to_numpy()
    dst_idx = df["dst_account"].map(idx_of).to_numpy()
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

    edge_attr = torch.tensor(np.stack([
        np.log1p(df[COL_AMT_PAID].to_numpy(dtype=float)),
        _codes(df[COL_CUR_PAID]).astype(float),
        _codes(df[COL_PAYMENT_FORMAT]).astype(float),
        df["t"].to_numpy(dtype=float),
    ], axis=1), dtype=torch.float)

    # Derived account label: involved in any illicit transaction.
    illicit_accounts = set(df.loc[df["label"] == 1, "src_account"]) | \
                       set(df.loc[df["label"] == 1, "dst_account"])
    y = torch.tensor([1 if a in illicit_accounts else 0 for a in accounts], dtype=torch.long)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = len(accounts)
    data.account_keys = list(accounts)
    data.graph_kind = "account_as_node"
    return data


# --------------------------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------------------------
def graph_stats(data: Data) -> dict:
    """Compute summary stats for a built graph (also nicely printable via ``print_stats``)."""
    n = int(data.num_nodes)
    e = int(data.edge_index.size(1))
    y = data.y.numpy()
    stats = {
        "graph_kind": getattr(data, "graph_kind", "?"),
        "num_nodes": n,
        "num_edges": e,
        "avg_out_degree": round(e / n, 3) if n else 0.0,
        "num_illicit": int((y == 1).sum()),
        "illicit_ratio": round(float((y == 1).mean()), 6) if n else 0.0,
    }
    if hasattr(data, "delta_t_seconds"):
        stats["delta_t_hours"] = round(data.delta_t_seconds / 3600, 3)
    if hasattr(data, "train_mask"):
        for split in ("train", "val", "test"):
            m = data[f"{split}_mask"].numpy()
            cnt = int(m.sum())
            ill = int((y[m] == 1).sum())
            stats[f"{split}_n"] = cnt
            stats[f"{split}_illicit"] = ill
            stats[f"{split}_illicit_ratio"] = round(ill / cnt, 6) if cnt else 0.0
    return stats


def print_stats(data: Data) -> dict:
    stats = graph_stats(data)
    width = max(len(k) for k in stats)
    print(f"--- graph stats: {stats['graph_kind']} ---")
    for k, v in stats.items():
        print(f"  {k.ljust(width)} : {v}")
    return stats


# --------------------------------------------------------------------------------------------
# Content-addressed cache (reuse a built graph across runs)
# --------------------------------------------------------------------------------------------
CACHE_VERSION = "v1"  # bump when construction logic changes so stale graphs are not reused


def graph_cache_key(spec: dict) -> str:
    """Stable hash of the construction parameters that determine the graph."""
    payload = json.dumps({"version": CACHE_VERSION, **spec}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_or_build(
    df: pd.DataFrame,
    spec: dict,
    cache_dir: str | Path,
    builder,
    force: bool = False,
):
    """Return a cached ``Data`` for ``spec`` if present, else build it via ``builder(df)`` and cache.

    ``spec`` must capture everything that affects the graph (variant, kind, Δt, subsample, split).
    Cached as ``<cache_dir>/<key>.pt`` with a sidecar ``<key>.json`` recording the spec.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = graph_cache_key(spec)
    pt_path = cache_dir / f"{key}.pt"
    json_path = cache_dir / f"{key}.json"

    if pt_path.exists() and not force:
        data = torch.load(pt_path, weights_only=False)
        return data, pt_path, True

    data = builder(df)
    torch.save(data, pt_path)
    json_path.write_text(json.dumps(spec, indent=2, sort_keys=True))
    return data, pt_path, False
