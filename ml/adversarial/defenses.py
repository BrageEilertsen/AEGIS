"""Adversarial-robustness defenses (spec §7.8).

Primary: **adversarial training** — each epoch, mix the clean loss with a loss on a structurally
perturbed training graph (edges incident to TRAIN illicit nodes randomly dropped), so the model
stops over-relying on any single laundering edge and resists the edge-removal/structuring attack.
This is train-only and leakage-safe (val/test never perturbed; standardization re-applied from the
clean train stats via the refeaturizer). Cheap (one perturb + one re-featurize per epoch) vs running
the full greedy attack inside training.

Secondary: **robust median aggregation** — a GraphSAGE ``aggregator: median`` variant flows through
the existing build_model knob (no new architecture); validated here.
"""
from __future__ import annotations

import torch

from ml.adversarial.featurize import make_refeaturizer


def validate_robust_aggregation(cfg: dict) -> None:
    arch = cfg.get("arch", {})
    if arch.get("aggregator") in {"median", "trimmed_mean"} and cfg.get("model") != "graphsage":
        raise ValueError(f"aggregator '{arch.get('aggregator')}' requires model: graphsage")


def make_adversarial_helpers(cfg, df, clean_data, feature_cfg, standardization, device):
    """Build the per-epoch adversarial-training helpers, or None if not configured.

    Returns (refeaturize, raw_x, node_amount, adv_cfg). Raises for the Elliptic path (no df), which
    is out of scope for adversarial training.
    """
    adv = cfg.get("train", {}).get("adversarial_training")
    if not adv:
        return None
    if df is None:
        raise NotImplementedError("adversarial_training is only supported on the IBM-AML feature path")
    base_masks = {n: clean_data[n].cpu() for n in ("train_mask", "val_mask", "test_mask")}
    refeat, raw_x, node_amount = make_refeaturizer(
        df, feature_cfg, standardization, clean_data.y.cpu(), base_masks, device)
    return refeat, raw_x, node_amount, adv


def _perturb_train_edges(edge_index, y, train_mask, frac, generator):
    """Drop a deterministic random `frac` of edges incident to TRAIN illicit nodes (augmentation)."""
    ei = edge_index.cpu()
    illicit_train = ((y == 1) & train_mask).cpu()
    incident = illicit_train[ei[0]] | illicit_train[ei[1]]
    inc_idx = incident.nonzero(as_tuple=False).flatten()
    if inc_idx.numel() == 0:
        return ei
    n_drop = max(1, round(frac * inc_idx.numel()))   # always perturb at least one illicit edge
    perm = torch.randperm(inc_idx.numel(), generator=generator)[:n_drop]
    keep = torch.ones(ei.size(1), dtype=torch.bool)
    keep[inc_idx[perm]] = False
    return ei[:, keep]


def train_epoch_adversarial(model, clean_data, helpers, optimizer, loss_fn, grad_clip,
                            epoch: int, seed: int) -> float:
    """One epoch of mixed clean + structurally-perturbed full-batch loss (train nodes only)."""
    refeat, raw_x, node_amount, adv = helpers
    frac = float(adv.get("budget_frac", 0.15))
    lam = float(adv.get("fraction", 0.3))
    g = torch.Generator().manual_seed(seed * 100003 + epoch)        # deterministic per epoch
    pert_ei = _perturb_train_edges(clean_data.edge_index, clean_data.y, clean_data.train_mask, frac, g)
    pert = refeat(pert_ei, raw_x, node_amount)                      # no mules -> raw/amount unchanged

    model.train()
    optimizer.zero_grad()
    m = clean_data.train_mask
    clean_logits = model(clean_data)
    pert_logits = model(pert)
    loss = (1 - lam) * loss_fn(clean_logits[m], clean_data.y[m]) \
        + lam * loss_fn(pert_logits[m], clean_data.y[m])
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())
