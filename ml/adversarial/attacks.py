"""Structural evasion attack on flagged transactions (spec §7.8).

A model-agnostic, deterministic greedy edge-perturbation attack: it only queries
``model.forward(data)`` (no surrogate / gradients needed, matching our interface and a real
launderer who only observes flags). Two moves, both flow-preserving and label-fixed:
  - remove: drop an incident flow edge (cuts the dense-subgraph signal the GNN keys on).
  - mule:   subdivide an incident edge i->target through an injected pass-through node m
            (i->m, m->target) — structuring / layering that dilutes local density.
After every edit the structure-derived features (local + spectral) are re-derived and the
checkpoint's train-only standardization re-applied (via the ``refeaturize`` closure); raw
per-transaction features are held fixed (net-flow-bound), so net flow is preserved by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ml.common import illicit_scores


@dataclass
class AttackConfig:
    budget_frac: float = 0.20       # edits per target = ceil(budget_frac * incident edges)
    budget_max: int = 15            # hard cap on edits per target
    net_flow_tol: float = 0.01      # max allowed per-account flow drift (moves preserve flow -> ~0)
    score_threshold: float = 0.50   # a node is "flagged" if illicit_score >= this
    top_k: int = 50                 # number of targets
    allow_mule_injection: bool = True
    max_mules_per_target: int = 2
    seed: int = 42


@torch.no_grad()
def query_scores(model, data) -> np.ndarray:
    model.eval()
    return illicit_scores(model(data))


def select_target_nodes(scores: np.ndarray, y: torch.Tensor, test_mask: torch.Tensor,
                        cfg: AttackConfig) -> list[int]:
    """Correctly-flagged illicit TEST nodes (leakage-safe), highest score first, capped to top_k."""
    y = y.cpu().numpy()
    m = test_mask.cpu().numpy()
    cand = [n for n in range(len(scores)) if m[n] and y[n] == 1 and scores[n] >= cfg.score_threshold]
    cand.sort(key=lambda n: -scores[n])
    return cand[:cfg.top_k]


class GreedyEdgeAttack:
    """Greedy best-first structural attack against ``model``, accumulating edits on a shared graph."""

    def __init__(self, model, edge_index, raw_x, node_amount, refeaturize, cfg: AttackConfig):
        self.model = model
        self.cfg = cfg
        self.refeaturize = refeaturize          # (edge_index, raw_x, node_amount) -> Data (on device)
        self.edge_index = edge_index.cpu()
        self.raw_x = raw_x.cpu()
        self.node_amount = node_amount.cpu()
        self.num_nodes = int(raw_x.size(0))

    # --- candidate moves (deterministic order: removes then mules, by (src,dst)) ---
    def _incident_pairs(self, target: int) -> list[tuple[int, int]]:
        ei = self.edge_index
        mask = (ei[0] == target) | (ei[1] == target)
        pairs = {(int(s), int(d)) for s, d in ei[:, mask].t().tolist()}
        return sorted(pairs)

    def _candidates(self, target: int, mules_used: int) -> list[tuple]:
        pairs = self._incident_pairs(target)
        cands = [("remove", s, d) for s, d in pairs]
        if self.cfg.allow_mule_injection and mules_used < self.cfg.max_mules_per_target:
            cands += [("mule", s, d) for s, d in pairs]
        return cands

    def _apply(self, edit, edge_index, raw_x, node_amount):
        op, s, d = edit
        keep = ~((edge_index[0] == s) & (edge_index[1] == d))
        ei = edge_index[:, keep]
        if op == "remove":
            return ei, raw_x, node_amount, raw_x.size(0)
        m = raw_x.size(0)                                   # new mule node appended at the end
        ei = torch.cat([ei, torch.tensor([[s, m], [m, d]])], dim=1)
        raw_x = torch.cat([raw_x, raw_x[d:d + 1]], dim=0)   # mule copies the receiver's raw features
        node_amount = torch.cat([node_amount, node_amount[d:d + 1]])
        return ei, raw_x, node_amount, m + 1

    def _net_flow_drift(self, edit) -> float:
        # remove changes no transaction; mule is a balanced pass-through -> per-account totals
        # unchanged. Raw amounts are never edited, so drift is 0 by construction.
        return 0.0

    def attack_single(self, target: int) -> dict:
        incident = len(self._incident_pairs(target))
        budget = min(self.cfg.budget_max, math.ceil(self.cfg.budget_frac * incident)) if incident else 0
        s0 = float(query_scores(self.model, self.refeaturize(
            self.edge_index, self.raw_x, self.node_amount))[target])
        score_before, edits, mules = s0, [], 0
        max_drift = 0.0
        for _ in range(budget):
            best, best_drop, best_state = None, 1e-9, None
            for edit in self._candidates(target, mules):
                drift = self._net_flow_drift(edit)
                if drift > self.cfg.net_flow_tol:
                    continue
                ei2, rx2, na2, _ = self._apply(edit, self.edge_index, self.raw_x, self.node_amount)
                s = float(query_scores(self.model, self.refeaturize(ei2, rx2, na2))[target])
                drop = s0 - s
                if drop > best_drop:                        # strict best; ties keep earlier candidate
                    best, best_drop, best_state = edit, drop, (ei2, rx2, na2, s)
            if best is None:
                break                                       # honest early stop: nothing helps
            self.edge_index, self.raw_x, self.node_amount, s_new = best_state
            self.num_nodes = int(self.raw_x.size(0))
            edits.append([best[0], int(best[1]), int(best[2])])
            mules += 1 if best[0] == "mule" else 0
            max_drift = max(max_drift, self._net_flow_drift(best))
            s0 = s_new
        return {"node_id": int(target), "y": 1,
                "score_before": round(score_before, 6), "score_after": round(s0, 6),
                "delta": round(s0 - score_before, 6), "n_edits": len(edits),
                "mules_added": mules, "net_flow_drift": max_drift, "budget": budget, "edits": edits}

    def run(self, targets: list[int]) -> tuple[object, list[dict]]:
        records = [self.attack_single(t) for t in targets]
        perturbed = self.refeaturize(self.edge_index, self.raw_x, self.node_amount)
        return perturbed, records
