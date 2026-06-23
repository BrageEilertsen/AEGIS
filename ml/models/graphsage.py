"""GraphSAGE — inductive, neighbor-sampling model (spec §7.3, step 2) — Phase 3.

A SAGEConv stack with the common ``forward(data) -> logits[N, 2]`` interface, so it drops into
``build_model`` and the train/eval loops unchanged. SAGEConv consumes the **directed**
``edge_index`` (it aggregates over each node's incoming neighbours) — unlike GCN, which symmetrizes
— so directionality of money flow is preserved. Inductive neighbour sampling is an optional
train-loop path (``train.neighbor_sampling`` in the config); the model itself is sampling-agnostic
because a sampled mini-batch is just a smaller ``Data`` flowing through the same forward.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    supports_attention = False

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, aggregator: str = "mean", num_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, num_classes, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))
            self.convs.append(SAGEConv(hidden_channels, num_classes, aggr=aggregator))

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index   # directed; SAGE aggregates incoming neighbours
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits [N, num_classes]
