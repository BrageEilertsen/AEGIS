"""Shared training/eval utilities — losses, metrics, determinism (spec §7.4, §7.6).

Used by both ml/train.py and ml/eval.py so the loss and metric definitions live in exactly one
place. Metrics are computed on the ILLICIT class from softmax probabilities; accuracy is never a
headline (meaningless at ~2% positives).
"""
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score,
)


# --------------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------------
def set_deterministic(seed: int) -> None:
    """Seed all RNGs and request deterministic algorithms (best-effort).

    GPU runs may still differ slightly across drivers; CPU runs are fully deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------------------------
# Imbalance: class weights + losses (computed on TRAIN mask only by the caller)
# --------------------------------------------------------------------------------------------
def compute_class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights from the masked (train) labels: w_c = n / (C * count_c).

    Missing classes get weight 0 (they contribute nothing). Returns a [num_classes] float tensor.
    """
    yt = y[mask]
    n = yt.numel()
    counts = torch.bincount(yt, minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float)
    nz = counts > 0
    weights[nz] = n / (num_classes * counts[nz])
    return weights


def weighted_ce_loss(logits: torch.Tensor, targets: torch.Tensor,
                     class_weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=class_weights.to(logits.device))


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: torch.Tensor,
               gamma: float = 2.0) -> torch.Tensor:
    """Numerically-stable multi-class focal loss (Lin et al. 2017), log-softmax form.

    loss = - alpha_t * (1 - p_t)^gamma * log p_t, computed in log-space (no log(0)).
    ``alpha`` is a per-class weight vector (here the inverse-frequency class weights).
    """
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets.view(-1, 1)).squeeze(1)
    pt = logpt.exp()
    at = alpha.to(logits.device)[targets]
    loss = -at * (1.0 - pt).pow(gamma) * logpt
    return loss.mean()


def build_loss(train_cfg: dict, class_weights: torch.Tensor) -> Callable:
    """Return a loss(logits, targets) callable selected by train_cfg['loss']."""
    kind = train_cfg.get("loss", "weighted_ce")
    if kind == "weighted_ce":
        return lambda logits, targets: weighted_ce_loss(logits, targets, class_weights)
    if kind == "focal":
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        return lambda logits, targets: focal_loss(logits, targets, class_weights, gamma)
    raise ValueError(f"unknown train.loss '{kind}' (expected 'weighted_ce' or 'focal')")


# --------------------------------------------------------------------------------------------
# Metrics (illicit class)
# --------------------------------------------------------------------------------------------
def illicit_scores(logits: torch.Tensor) -> np.ndarray:
    """Softmax probability of the illicit class (index 1)."""
    return F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray,
                        min_precision: float) -> tuple[float, float]:
    """Highest recall achievable while precision >= min_precision, and the score threshold for it.

    Returns (recall, threshold). If no operating point reaches the target precision, returns
    (0.0, 1.0) — meaning "to hit that precision you'd flag nothing".
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have length len(thresholds)+1; the last point (recall=0) has no threshold.
    best_recall, best_thr = 0.0, 1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall, best_thr = float(r), float(t)
    return best_recall, best_thr


def compute_metrics(y_true: np.ndarray, scores: np.ndarray,
                    min_precision: float = 0.9) -> dict:
    """PR-AUC (headline), recall@fixed-precision, F1-illicit, ROC-AUC, confusion matrix.

    The decision threshold is the one that achieves recall@min_precision; F1 and the confusion
    matrix are reported at that threshold. Degrades gracefully when a split has one class only.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())
    out = {
        "n_total": int(y_true.size),
        "n_pos": n_pos,
        "pos_rate": round(n_pos / y_true.size, 6) if y_true.size else 0.0,
    }
    if n_pos == 0 or n_pos == y_true.size:
        # Single-class split: ranking metrics undefined.
        out.update({"pr_auc": None, "roc_auc": None, "recall_at_precision": None,
                    "threshold": None, "f1_illicit": None, "confusion_matrix": None})
        return out

    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    recall, thr = recall_at_precision(y_true, scores, min_precision)
    out["recall_at_precision"] = recall
    out["min_precision"] = min_precision
    out["threshold"] = thr
    preds = (scores >= thr).astype(int)
    out["f1_illicit"] = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out
