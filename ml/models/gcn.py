"""GCN — baseline model (spec §7.3, step 1) — Phase 2.

A stack of PyG ``GCNConv`` layers. GCN assumes an undirected graph, so the directed flow edges
are symmetrized inside ``forward``. The ``forward(data) -> logits`` signature is the common model
interface so GraphSAGE / GAT (Phase 3) drop into ``build_model`` without changing train/eval.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, num_classes))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = to_undirected(data.edge_index, num_nodes=x.size(0))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits [N, num_classes]


def build_model(config: dict, in_channels: int) -> torch.nn.Module:
    """Factory keyed on config['model']. Phase 2 = GCN; GraphSAGE/GAT raise (Phase 3)."""
    model_type = config.get("model", "gcn")
    arch = config.get("arch", {})
    if model_type == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=int(arch.get("hidden_channels", 64)),
            num_layers=int(arch.get("num_layers", 2)),
            dropout=float(arch.get("dropout", 0.5)),
            num_classes=2,
        )
    raise NotImplementedError(f"model '{model_type}' is not implemented in Phase 2 (GCN only)")
