"""Feature assembly: compose raw + local + spectral, cache the spectral group, standardize once.

This is the single orchestration point for Phase-2 features and the single **leakage chokepoint**:
standardization stats and (elsewhere) class weights are computed on the TRAIN mask only. Only the
expensive spectral group is persistently cached (raw/local are cheap and recomputed each run).

Cache composition (two levels, on top of Phase-1's graph cache):
- graph.graph_cache_key  -> keys the built graph by construction params (Δt, subsample, ...).
- spectral.graph_hash    -> structure-only hash of the graph, the spectral cache namespace.
- feature_cache_key      -> sha1(graph_hash, FEATURE_SPEC_VERSION, k, sorted(centralities)).
Changing k or the centrality list invalidates the spectral cache; changing raw/local config does
not. Bump FEATURE_SPEC_VERSION on any spectral-logic change to invalidate all caches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from ml.features.local import build_local_features
from ml.features.raw import build_raw_features
from ml.features.spectral import compute_spectral_features, graph_hash

FEATURE_SPEC_VERSION = "v2"   # v2: per-connected-component Laplacian PE (was global eigendecomp)


def feature_cache_key(gh: str, feature_cfg: dict) -> str:
    """sha1[:16] over the spectral-affecting knobs only."""
    spec = {
        "graph_hash": gh,
        "version": FEATURE_SPEC_VERSION,
        "k": int(feature_cfg.get("laplacian_pe_k", 16)),
        "centralities": sorted(feature_cfg.get("centralities", [])),
    }
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def load_or_compute_spectral(data, feature_cfg: dict, cache_dir: str | Path,
                             force: bool = False) -> tuple[dict, dict]:
    """Return (spectral tensors, cache_meta). Mirrors graph.load_or_build.

    On hit: torch.load the cached tensors. On miss: compute, save ``<key>.pt`` + sidecar JSON.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gh = graph_hash(data)
    key = feature_cache_key(gh, feature_cfg)
    pt_path = cache_dir / f"{key}.pt"
    json_path = cache_dir / f"{key}.json"

    k = int(feature_cfg.get("laplacian_pe_k", 16))
    which = list(feature_cfg.get("centralities", []))

    if pt_path.exists() and not force:
        tensors = torch.load(pt_path, weights_only=False)
        return tensors, {"hit": True, "key": key, "graph_hash": gh, "path": str(pt_path)}

    feats = compute_spectral_features(data, k, which)
    torch.save(feats, pt_path)
    json_path.write_text(json.dumps({
        "graph_hash": gh, "version": FEATURE_SPEC_VERSION, "k": k,
        "centralities": which, **feats["meta"],
    }, indent=2, sort_keys=True))
    return feats, {"hit": False, "key": key, "graph_hash": gh, "path": str(pt_path)}


def standardize_train_only(X: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Standardize using TRAIN-mask statistics only — the one leakage chokepoint.

    Returns (X_std, {'mean':[d], 'std':[d]}).
    """
    Xt = X[train_mask]
    mean = Xt.mean(dim=0)
    std = Xt.std(dim=0)
    X_std = (X - mean) / (std + 1e-8)
    return X_std, {"mean": mean.tolist(), "std": std.tolist()}


def apply_standardization(X: torch.Tensor, stats: dict) -> torch.Tensor:
    """Re-apply saved train stats (eval / inference path — never recompute)."""
    mean = torch.tensor(stats["mean"], dtype=X.dtype)
    std = torch.tensor(stats["std"], dtype=X.dtype)
    return (X - mean) / (std + 1e-8)


def assemble_features(data, df: pd.DataFrame, feature_cfg: dict, cache_dir: str | Path,
                      force_feature_cache: bool = False, standardization: dict | None = None
                      ) -> tuple[object, dict]:
    """Build [raw | local | spectral_pe | centralities], standardize, write into data.x.

    If ``standardization`` is given (eval/inference), it is applied; otherwise stats are computed
    train-only and returned. Returns (data, feat_meta).
    """
    raw_x, raw_names = build_raw_features(df)
    local_x, local_names = build_local_features(
        data, df, max_degree=int(feature_cfg.get("max_degree", 2000)))
    spectral, cache_meta = load_or_compute_spectral(
        data, feature_cfg, cache_dir, force=force_feature_cache)

    pe = spectral["lap_pe"]
    cent = spectral["centralities"]
    pe_names = [f"lap_pe_{i}" for i in range(pe.size(1))]
    cent_names = spectral["centrality_names"]

    X = torch.cat([raw_x, local_x, pe, cent], dim=1)
    columns = raw_names + local_names + pe_names + cent_names

    if standardization is None:
        X_std, stats = standardize_train_only(X, data.train_mask)
    else:
        X_std, stats = apply_standardization(X, standardization), standardization

    data.x = X_std.float()
    feat_meta = {
        "columns": columns,
        "n_features": len(columns),
        "group_dims": {"raw": raw_x.size(1), "local": local_x.size(1),
                       "spectral_pe": pe.size(1), "centralities": cent.size(1)},
        "standardization": stats,
        "spectral_cache": cache_meta,
        "feature_spec_version": FEATURE_SPEC_VERSION,
    }
    return data, feat_meta
