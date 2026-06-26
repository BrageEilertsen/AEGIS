"""Account-centric graph construction (the ML-deepening track).

The primary graph (``graph.py``) is transaction-as-node: each transaction is a node, edges link
transactions that chain in time. On the IBM-AML LI-Small data that graph is effectively edgeless
(average degree ~0.14) because the public file is a *sample* of transactions, so a transaction's
in/out neighbours are usually absent — the GNN degrades to an MLP and architecture barely moves
PR-AUC.

This module rebuilds the graph **account-centric** over the FULL transaction file: accounts are
nodes, transactions are (aggregated, directed) edges, and a node is labelled illicit if it takes
part in any laundering transaction. Accounts recur across many transactions, so the graph is dense
(average degree ~9.8, 96% of accounts have ≥2 links) — the laundering motifs (fan-in/out, chains,
cycles) appear as connected subgraphs the GNN can actually reason over.

Node features are per-account behaviour aggregates (in/out counts, mean/σ amount, distinct
counterparties, currency/format diversity, reuse ratio). ``build_account_graph`` returns a PyG
``Data`` with ``x``, ``edge_index`` (undirected), ``y`` and standardized features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .loaders import (
    COL_AMT_PAID, COL_CUR_PAID, COL_FROM_ACCT, COL_FROM_BANK, COL_IS_LAUNDERING,
    COL_PAYMENT_FORMAT, COL_TO_ACCT, COL_TO_BANK, account_key,
)

# Human-readable names of the per-account feature columns (order matches the matrix below).
FEATURE_NAMES = [
    "log_out_count", "log_in_count", "mean_out_amount", "mean_in_amount",
    "std_out_amount", "std_in_amount", "log_out_counterparties", "log_in_counterparties",
    "log_total_activity", "out_in_ratio", "total_amount", "log_distinct_currencies",
    "log_distinct_formats", "counterparty_reuse_ratio",
]


def build_account_graph(df: pd.DataFrame, *, standardize: bool = True) -> Data:
    """Build the account-as-node graph for node classification from a raw IBM-AML frame."""
    src = account_key(df[COL_FROM_BANK], df[COL_FROM_ACCT])
    dst = account_key(df[COL_TO_BANK], df[COL_TO_ACCT])
    codes = pd.Categorical(pd.concat([src, dst], ignore_index=True))
    n = len(codes.categories)
    s = codes.codes[: len(df)].astype(np.int64)
    d = codes.codes[len(df):].astype(np.int64)

    amt = np.log1p(df[COL_AMT_PAID].to_numpy(dtype=float))
    lt = df[COL_IS_LAUNDERING].to_numpy(dtype=np.int64)
    cur = pd.Categorical(df[COL_CUR_PAID]).codes
    fmt = pd.Categorical(df[COL_PAYMENT_FORMAT]).codes

    # Node label: account participates in at least one laundering transaction.
    y = np.zeros(n, dtype=np.int64)
    y[s[lt == 1]] = 1
    y[d[lt == 1]] = 1

    def bc(idx, w=None):
        return np.bincount(idx, weights=w, minlength=n)

    out_c, in_c = bc(s), bc(d)
    out_a, in_a = bc(s, amt), bc(d, amt)
    out_a2, in_a2 = bc(s, amt * amt), bc(d, amt * amt)
    out_m, in_m = out_a / np.maximum(out_c, 1), in_a / np.maximum(in_c, 1)
    out_sd = np.sqrt(np.maximum(out_a2 / np.maximum(out_c, 1) - out_m ** 2, 0))
    in_sd = np.sqrt(np.maximum(in_a2 / np.maximum(in_c, 1) - in_m ** 2, 0))
    out_deg = pd.DataFrame({"a": s, "b": d}).groupby("a")["b"].nunique().reindex(range(n), fill_value=0).to_numpy()
    in_deg = pd.DataFrame({"a": d, "b": s}).groupby("a")["b"].nunique().reindex(range(n), fill_value=0).to_numpy()
    n_cur = pd.DataFrame({"a": s, "c": cur[: len(df)]}).groupby("a")["c"].nunique().reindex(range(n), fill_value=0).to_numpy()
    n_fmt = pd.DataFrame({"a": s, "f": fmt[: len(df)]}).groupby("a")["f"].nunique().reindex(range(n), fill_value=0).to_numpy()

    feats = np.stack([
        np.log1p(out_c), np.log1p(in_c), out_m, in_m, out_sd, in_sd,
        np.log1p(out_deg), np.log1p(in_deg), np.log1p(out_c + in_c),
        (out_c + 1) / (in_c + 1), out_a + in_a, np.log1p(n_cur), np.log1p(n_fmt),
        (out_deg + 1) / (out_c + 1),
    ], axis=1).astype(np.float32)
    if standardize:
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    # Edges: unique directed account pairs, made undirected for symmetric message passing.
    pair = pd.DataFrame({"s": s, "d": d}).drop_duplicates().to_numpy().T
    edge_index = torch.from_numpy(np.concatenate([pair, pair[::-1]], axis=1)).long()

    data = Data(x=torch.from_numpy(feats), edge_index=edge_index, y=torch.from_numpy(y))
    data.num_nodes = n
    data.feature_names = FEATURE_NAMES
    return data
