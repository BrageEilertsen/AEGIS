"""GAT — Graph Attention Network, the primary model (spec §7.3, step 3) — Phase 3.

GATv2Conv (default; ``variant`` selects v1/v2) with the common ``forward(data) -> logits[N, 2]``
interface. Built with ``add_self_loops=False`` so the attention returned by each layer is **1:1
aligned with the input ``edge_index``** — the per-edge attention then maps straight back to
original transaction edges with no self-loop filtering, which the Phase-4 explainability subsystem
(spec §7.7) consumes via ``extract_attention``.

Attention from every layer is buffered on each forward; ``extract_attention`` re-runs a clean
(eval, no-grad) forward and assembles the explanation contract. Heads are concatenated on hidden
layers and averaged on the output layer (standard GAT).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(torch.nn.Module):
    supports_attention = True

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, heads: int = 8, attention_dropout: float = 0.3,
                 variant: str = "gat_v2", num_classes: int = 2):
        super().__init__()
        self.dropout = dropout
        Conv = GATv2Conv if variant == "gat_v2" else GATConv
        self.convs = torch.nn.ModuleList()
        kw = dict(dropout=attention_dropout, add_self_loops=False)
        if num_layers == 1:
            self.convs.append(Conv(in_channels, num_classes, heads=1, concat=False, **kw))
        else:
            self.convs.append(Conv(in_channels, hidden_channels, heads=heads, concat=True, **kw))
            for _ in range(num_layers - 2):
                self.convs.append(
                    Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, **kw))
            self.convs.append(
                Conv(hidden_channels * heads, num_classes, heads=1, concat=False, **kw))
        self._attn: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        self._attn = []
        for i, conv in enumerate(self.convs):
            x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
            self._attn.append((ei.detach(), alpha.detach()))   # alpha: [E, heads]
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits [N, num_classes]

    @torch.no_grad()
    def extract_attention(self, data, layer: "str | int" = "last",
                          head_reduce: str = "mean") -> dict:
        """Per-edge attention for explainability (spec §7.7).

        Returns a dict with ``edge_index`` (== data.edge_index, since add_self_loops=False),
        head-reduced ``attention`` [E], raw ``attention_per_head`` [E, H], ``node_logits`` [N, 2],
        the resolved ``layer`` and ``num_heads``. For layer="all", ``attention`` /
        ``attention_per_head`` are lists per layer.
        """
        was_training = self.training
        self.eval()
        logits = self.forward(data)
        if was_training:
            self.train()

        n_layers = len(self._attn)
        if layer == "last":
            idxs = [n_layers - 1]
        elif layer == "all":
            idxs = list(range(n_layers))
        else:
            idx = int(layer)
            idxs = [idx if idx >= 0 else n_layers + idx]

        def reduce(alpha: torch.Tensor) -> torch.Tensor:
            if head_reduce == "mean":
                return alpha.mean(dim=1)
            if head_reduce == "max":
                return alpha.max(dim=1).values
            if head_reduce == "none":
                return alpha
            raise ValueError(f"unknown head_reduce '{head_reduce}'")

        per_head = [self._attn[i][1] for i in idxs]
        reduced = [reduce(a) for a in per_head]
        single = len(idxs) == 1
        return {
            "edge_index": self._attn[idxs[0]][0],
            "attention": reduced[0] if single else reduced,
            "attention_per_head": per_head[0] if single else per_head,
            "node_logits": logits,
            "layer": idxs[0] if single else idxs,
            "num_heads": per_head[0].size(1) if single else [a.size(1) for a in per_head],
        }
