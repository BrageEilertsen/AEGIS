"""Re-featurize a (possibly perturbed) graph with the checkpoint's train-only standardization.

Shared by the attack runner and the adversarial-training defense. Re-derives the structure-derived
feature groups (local + spectral) from the given edge_index, keeps raw features fixed, and re-applies
the saved standardization — never recomputes stats (no leakage). Lives in its own module so both
``runner`` and ``defenses`` (and, via defenses, ``train``) can import it without a cycle.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ml.data.loaders import COL_AMT_PAID
from ml.features.assemble import apply_standardization
from ml.features.local import build_local_features
from ml.features.raw import build_raw_features
from ml.features.spectral import compute_spectral_features

MASKS = ("train_mask", "val_mask", "test_mask")


def make_refeaturizer(df, feature_cfg, standardization, base_y, base_masks, device):
    """Return (refeaturize(edge_index, raw_x_ext, node_amount_ext) -> Data on device, raw_x, node_amount).

    Injected nodes appended beyond the original N (mules) get y=0 and no split mask.
    """
    raw_x = build_raw_features(df)[0]
    node_amount = torch.tensor(np.log1p(df[COL_AMT_PAID].to_numpy(dtype=float)), dtype=torch.float)
    N = raw_x.size(0)
    k = int(feature_cfg.get("laplacian_pe_k", 0))
    cents = list(feature_cfg.get("centralities", []))
    max_degree = int(feature_cfg.get("max_degree", 2000))

    def refeaturize(edge_index, raw_x_ext, node_amount_ext) -> Data:
        n = int(raw_x_ext.size(0))
        d = Data(edge_index=edge_index); d.num_nodes = n
        local_x, _ = build_local_features(d, None, max_degree=max_degree, node_amount=node_amount_ext)
        if k > 0 or cents:
            spec = compute_spectral_features(d, k, cents)
            pe, cent = spec["lap_pe"], spec["centralities"]
        else:
            pe = cent = torch.zeros((n, 0), dtype=torch.float)
        X = apply_standardization(torch.cat([raw_x_ext, local_x, pe, cent], dim=1), standardization)
        out = Data(x=X.float(), edge_index=edge_index); out.num_nodes = n
        m = n - N
        out.y = torch.cat([base_y, torch.zeros(m, dtype=base_y.dtype)]) if m > 0 else base_y.clone()
        for name in MASKS:
            mk = base_masks[name]
            out[name] = torch.cat([mk, torch.zeros(m, dtype=torch.bool)]) if m > 0 else mk.clone()
        return out.to(device)

    return refeaturize, raw_x, node_amount
