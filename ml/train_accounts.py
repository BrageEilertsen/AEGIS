"""Account-graph experiment: does the graph STRUCTURE add signal beyond per-account features?

Trains two models on the same features and the same split:
  * MLP        — features only, ignores the graph (the "no structure" baseline)
  * GraphSAGE  — message-passes over the account graph

A clear PR-AUC / ROC-AUC gap means the account-centric reconstruction recovered real structure the
transaction graph lacked. Run:

    python -m ml.train_accounts --csv data/raw/LI-Small_Trans.csv --epochs 120
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import SAGEConv

from ml.data.account_graph import build_account_graph


class MLP(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.l1, self.l2, self.o = nn.Linear(f, h), nn.Linear(h, h), nn.Linear(h, 2)

    def forward(self, x, edge_index=None):
        x = F.dropout(F.relu(self.l1(x)), 0.3, self.training)
        return self.o(F.relu(self.l2(x)))


class GraphSAGE(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.c1, self.c2, self.c3 = SAGEConv(f, h), SAGEConv(h, h), SAGEConv(h, h)
        self.o = nn.Linear(h, 2)

    def forward(self, x, edge_index):
        x = F.dropout(F.relu(self.c1(x, edge_index)), 0.3, self.training)
        x = F.dropout(F.relu(self.c2(x, edge_index)), 0.3, self.training)
        return self.o(F.relu(self.c3(x, edge_index)))


def stratified_masks(y, seed=42):
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    def cut(a):
        return a[: int(.7 * len(a))], a[int(.7 * len(a)): int(.85 * len(a))], a[int(.85 * len(a)):]
    (trp, vap, tep), (trn, van, ten) = cut(pos), cut(neg)
    masks = []
    for part in [(trp, trn), (vap, van), (tep, ten)]:
        m = torch.zeros(n, dtype=torch.bool)
        m[np.concatenate(part)] = True
        masks.append(m)
    return masks


def train_eval(model, data, masks, weight, epochs):
    train_m, val_m, test_m = masks
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state, t0 = -1.0, None, time.time()
    for e in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        F.cross_entropy(out[train_m], data.y[train_m], weight=weight).backward()
        opt.step()
        if e % 10 == 0 or e == epochs - 1:
            model.eval()
            with torch.no_grad():
                p = F.softmax(model(data.x, data.edge_index), 1)[:, 1].numpy()
            vap = average_precision_score(data.y[val_m].numpy(), p[val_m.numpy()])
            if vap > best_val:
                best_val, best_state = vap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = F.softmax(model(data.x, data.edge_index), 1)[:, 1].numpy()
    tm = test_m.numpy()
    return {"pr_auc": average_precision_score(data.y[test_m].numpy(), p[tm]),
            "roc_auc": roc_auc_score(data.y[test_m].numpy(), p[tm]),
            "val_pr_auc": best_val, "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/LI-Small_Trans.csv")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"loading {args.csv} …")
    df = pd.read_csv(args.csv, usecols=[
        "From Bank", "Account", "To Bank", "Account.1", "Amount Paid",
        "Payment Currency", "Payment Format", "Is Laundering"])
    data = build_account_graph(df)
    deg = data.edge_index.shape[1] / data.num_nodes
    print(f"account graph: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges "
          f"(avg degree {deg:.2f}), {int(data.y.sum()):,} illicit ({100*data.y.float().mean():.3f}%)")

    masks = stratified_masks(data.y.numpy(), args.seed)
    pos_w = float((data.y[masks[0]] == 0).sum()) / float((data.y[masks[0]] == 1).sum())
    weight = torch.tensor([1.0, pos_w])
    base = data.y[masks[2]].float().mean().item()
    print(f"base rate (random PR-AUC) = {base:.4f}\n")

    results = {}
    for name, model in [("MLP (no graph)", MLP(data.x.shape[1], args.hidden)),
                        ("GraphSAGE", GraphSAGE(data.x.shape[1], args.hidden))]:
        r = train_eval(model, data, masks, weight, args.epochs)
        results[name] = r
        print(f"{name:16s} test PR-AUC={r['pr_auc']:.3f}  ROC-AUC={r['roc_auc']:.3f}  ({r['seconds']:.0f}s)")

    lift = results["GraphSAGE"]["pr_auc"] / max(results["MLP (no graph)"]["pr_auc"], 1e-9)
    print(f"\nGraph lift: GraphSAGE PR-AUC is {lift:.2f}x the feature-only MLP, "
          f"{results['GraphSAGE']['pr_auc']/base:.0f}x the base rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
