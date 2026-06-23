"""Graph construction from the IBM-AML dataset (Phase 1).

Two graph views over the same transactions (spec §6.3):

1. Transaction-as-node (primary, matches Elliptic): each transaction is a node; a directed edge
   v_i -> v_j exists when the receiver account of v_i is the sender account of v_j, v_j occurs
   after v_i, and 0 <= t_j - t_i <= Δt (Δt configurable). Captures money flow.
2. Account-as-node (for visualization clarity): accounts are nodes, transactions are edges with
   amount / currency / timestamp as edge features. More legible for the analyst UI.

Outputs a clean, documented PyG Data/HeteroData object with labels and a temporal 60/20/20 split
(spec §6.4). Keep all illicit transactions; subsample the legitimate majority where needed, but
preserve realistic ratios in the held-out evaluation set.

Phase 0: placeholder. Implemented in Phase 1.
"""
from __future__ import annotations

# TODO (Phase 1):
#   - load_ibm_aml(path) -> pandas frames (transactions, accounts, patterns)
#   - build_transaction_graph(df, delta_t) -> torch_geometric.data.Data
#   - build_account_graph(df) -> torch_geometric.data.Data
#   - temporal_split(data, ratios=(0.6, 0.2, 0.2)) -> train/val/test masks by transaction time
#   - print_stats(data): node/edge counts, illicit ratio per split
