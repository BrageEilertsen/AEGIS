"""Raw / intrinsic node features (spec §7.2, group 1) — Phase 2.

Expands Phase-1's minimal feature set into a documented raw group. Returns **unstandardized**
features; standardization happens once, train-only, in ``ml/features/assemble.py`` (the single
leakage chokepoint). Bank ids are feature-hashed (fixed width, stable across splits/variants)
rather than one-hot.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch

from ml.data.loaders import (
    COL_AMT_PAID, COL_AMT_RECEIVED, COL_CUR_PAID, COL_CUR_RECEIVED,
    COL_FROM_BANK, COL_PAYMENT_FORMAT, COL_TO_BANK,
)


def _codes(series: pd.Series) -> np.ndarray:
    """Stable integer codes for a categorical column."""
    return series.astype("category").cat.codes.to_numpy()


def _hash_bucket(series: pd.Series, n_buckets: int = 64) -> np.ndarray:
    """Deterministic feature-hash of high-cardinality ids into [0, n_buckets).

    Uses md5 (not Python's salted ``hash``) so the mapping is identical across runs/processes.
    """
    def h(v: object) -> int:
        return int(hashlib.md5(str(v).encode()).hexdigest(), 16) % n_buckets
    return series.map(h).to_numpy()


def _cyclical(values: np.ndarray, period: float) -> np.ndarray:
    """[sin, cos] encoding of a periodic quantity (avoids the wrap-around discontinuity)."""
    ang = 2.0 * np.pi * (values / period)
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)


def build_raw_features(df: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    """Return ([N, d_raw] float tensor, column names). NOT standardized.

    df must be the normalized, node-aligned frame (row i == transaction node i).
    """
    amt_paid = df[COL_AMT_PAID].to_numpy(dtype=float)
    amt_recv = df[COL_AMT_RECEIVED].to_numpy(dtype=float)
    log_paid = np.log1p(amt_paid)
    log_recv = np.log1p(amt_recv)
    amt_ratio = np.log1p(amt_paid / np.maximum(amt_recv, 1e-9))

    fmt = _codes(df[COL_PAYMENT_FORMAT]).astype(float)
    cur_paid = _codes(df[COL_CUR_PAID]).astype(float)
    cur_recv = _codes(df[COL_CUR_RECEIVED]).astype(float)
    cross_currency = (df[COL_CUR_PAID].to_numpy() != df[COL_CUR_RECEIVED].to_numpy()).astype(float)

    from_bank = _hash_bucket(df[COL_FROM_BANK]).astype(float)
    to_bank = _hash_bucket(df[COL_TO_BANK]).astype(float)

    t = df["t"].to_numpy(dtype=float)
    tod = _cyclical(t % 86400.0, 86400.0)          # time of day
    dow = _cyclical((t // 86400.0) % 7.0, 7.0)     # day of week
    day_index = (t // 86400.0)

    cols = {
        "log_amt_paid": log_paid,
        "log_amt_recv": log_recv,
        "log_amt_ratio": amt_ratio,
        "payment_format_code": fmt,
        "currency_paid_code": cur_paid,
        "currency_recv_code": cur_recv,
        "cross_currency": cross_currency,
        "from_bank_hash": from_bank,
        "to_bank_hash": to_bank,
        "tod_sin": tod[:, 0], "tod_cos": tod[:, 1],
        "dow_sin": dow[:, 0], "dow_cos": dow[:, 1],
        "day_index": day_index,
    }
    names = list(cols.keys())
    feats = np.stack([cols[n] for n in names], axis=1)
    return torch.tensor(feats, dtype=torch.float), names
