"""Elliptic1 Bitcoin benchmark (spec §6.2, §7.6) — Phase 3.

The published academic standard: ~203k transaction nodes, 165/166 provided features, 49 timesteps,
~2% illicit / ~21% licit / ~77% unknown. We report PR-AUC / recall@precision / F1-illicit against
published GCN/GAT baselines for external credibility.

- ``load_elliptic1``: wraps ``torch_geometric.datasets.EllipticBitcoinDataset`` (downloads ~hundreds
  of MB; the cluster TLS-intercepts HTTPS so the caller must export
  ``SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`` — see CLAUDE.md). Label convention (PyG):
  ``y == 0`` licit, ``y == 1`` illicit, ``y == 2`` unknown. Illicit is class 1, matching the rest
  of the pipeline. Real download runs on a GPU node / by a human — never in tests.
- ``elliptic_temporal_masks``: leakage-safe split over the 49 timesteps, every split intersected
  with ``y != 2`` so unknown nodes never enter loss/metrics (spec §6.4, §15 label-noise mitigation).
- ``make_elliptic_fixture``: a tiny Elliptic-SHAPED ``Data`` for login-CPU tests — never downloads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

UNKNOWN = 2  # PyG label for unlabeled Elliptic nodes


def load_elliptic1(cache_dir: str | Path, force_reload: bool = False) -> Data:
    """Load the real Elliptic1 dataset as a PyG ``Data`` with ``time_step`` attached.

    Requires network + the system CA bundle (TLS interception). Sets ``data.time_step`` from the
    first feature column (the Elliptic convention) and ``data.graph_kind = 'elliptic_tx'``.
    """
    from torch_geometric.datasets import EllipticBitcoinDataset
    ds = EllipticBitcoinDataset(root=str(Path(cache_dir)), force_reload=force_reload)
    data = ds[0]
    # Elliptic's first local feature is the timestep (1..49); used for the temporal split.
    ts = data.x[:, 0].round().long()
    lo, hi = int(ts.min()), int(ts.max())
    if not (1 <= lo and hi <= 49):
        print(f"[elliptic] WARNING: time_step from x[:,0] looks off (range {lo}..{hi}); "
              f"verify the dataset layout before trusting the temporal split.")
    data.time_step = ts
    data.graph_kind = "elliptic_tx"
    data.num_nodes = data.x.size(0)
    return data


def elliptic_temporal_masks(time_step: torch.Tensor, y: torch.Tensor,
                            ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
                            ) -> dict[str, torch.Tensor]:
    """Leakage-safe temporal split over labeled nodes only (unknown ``y==2`` excluded everywhere).

    Labeled nodes are ordered by timestep; the earliest ``ratios[0]`` go to train, etc. Unknown
    nodes appear in no split (they remain in the graph for message passing).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    n = time_step.numel()
    labeled = (y != UNKNOWN).nonzero(as_tuple=False).flatten()
    ts = time_step[labeled]
    order = labeled[torch.argsort(ts, stable=True)]
    m = order.numel()
    n_train = int(ratios[0] * m)
    n_val = int(ratios[1] * m)
    parts = {"train_mask": order[:n_train],
             "val_mask": order[n_train:n_train + n_val],
             "test_mask": order[n_train + n_val:]}

    def mask(idx):
        out = torch.zeros(n, dtype=torch.bool)
        out[idx] = True
        return out

    return {k: mask(v) for k, v in parts.items()}


def make_elliptic_fixture(seed: int = 0, n_nodes: int = 400, n_features: int = 166,
                          n_timesteps: int = 49) -> Data:
    """Tiny Elliptic-SHAPED Data for login-CPU tests (no download).

    x[N, n_features] (column 0 = timestep), a sparse directed edge_index, y with ~77% unknown(2) /
    ~21% licit(0) / ~2% illicit(1), and a faint planted signal (illicit nodes get a feature offset)
    so a forward pass is meaningful.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, n_features, generator=g)
    time_step = torch.randint(1, n_timesteps + 1, (n_nodes,), generator=g)
    x[:, 0] = time_step.float()                       # column 0 holds the timestep (Elliptic convention)

    # Label mix: ~77% unknown, ~21% licit, ~2% illicit.
    r = torch.rand(n_nodes, generator=g)
    y = torch.full((n_nodes,), UNKNOWN, dtype=torch.long)
    y[r < 0.23] = 0                                   # licit
    y[r < 0.025] = 1                                  # illicit (faint planted signal below)
    x[y == 1, 1:] += 1.5                              # illicit nodes shifted on the real features

    n_edges = n_nodes * 3
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    edge_index = torch.stack([src, dst])

    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.graph_kind = "elliptic_tx"
    data.num_nodes = n_nodes
    return data
