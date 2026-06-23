"""GNN models behind a common interface: GCN -> GraphSAGE -> GAT (-> temporal) (spec §7.3).

All models expose the same ``forward(data) -> logits[N, 2]`` so train/eval and the serving layer
stay model-agnostic. ``build_model`` is the single dispatch point.
"""
from __future__ import annotations

import torch

from ml.models.gat import GAT
from ml.models.gcn import GCN
from ml.models.graphsage import GraphSAGE


def build_model(config: dict, in_channels: int) -> torch.nn.Module:
    """Instantiate the model named by config['model'] with the common arch knobs."""
    model_type = config.get("model", "gcn")
    arch = config.get("arch", {})
    common = dict(in_channels=in_channels,
                  hidden_channels=int(arch.get("hidden_channels", 64)),
                  num_layers=int(arch.get("num_layers", 2)),
                  dropout=float(arch.get("dropout", 0.5)),
                  num_classes=2)
    if model_type == "gcn":
        return GCN(**common)
    if model_type == "graphsage":
        return GraphSAGE(**common, aggregator=str(arch.get("aggregator", "mean")))
    if model_type == "gat":
        return GAT(**common, heads=int(arch.get("heads", 8)),
                   attention_dropout=float(arch.get("attention_dropout", 0.3)),
                   variant=str(arch.get("variant", "gat_v2")))
    raise NotImplementedError(f"model '{model_type}' is not implemented")


__all__ = ["build_model", "GCN", "GraphSAGE", "GAT"]
