"""Adapter bridging AEGIS models (forward(data)) to PyG's Explainer (forward(x, edge_index))."""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.explain import ModelConfig


class ExplainerAdapter(torch.nn.Module):
    """Wrap an AEGIS model so PyG's ``Explainer`` can call it as ``forward(x, edge_index)``.

    The trained ``model`` is held in place (no copy), so ``model.extract_attention`` still works on
    the same object afterwards. The adapter does NOT symmetrize edges — GCN.forward owns its
    ``to_undirected``; GraphSAGE/GAT use directed edges — so explanations reflect the exact topology
    each model received in training.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(Data(x=x, edge_index=edge_index))


def model_config() -> ModelConfig:
    """Node-level multiclass config matching our raw-logit GNN outputs."""
    return ModelConfig(mode="multiclass_classification", task_level="node", return_type="raw")
