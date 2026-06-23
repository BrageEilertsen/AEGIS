"""Spectral / global node features — the technical signature of the project (spec §7.2, group 3).

- Laplacian positional encodings: first k non-trivial eigenvectors of the graph Laplacian.
- Spectral-clustering community assignments.
- Centralities: PageRank, eigenvector centrality, betweenness (where tractable).

These encode global position and community structure that local message passing captures slowly.
k is configurable. Eigendecompositions are EXPENSIVE — cache the results keyed by graph hash in
the persistent --feature-cache directory so they are reused across every run, never recomputed
per job (spec §7.2, §11.5). Document any approximation used for large graphs.

Phase 0: placeholder. Implemented in Phase 2.
"""
from __future__ import annotations

# TODO (Phase 2):
#   - graph_hash(data) -> str  (stable key for the feature cache)
#   - laplacian_pe(data, k, cache_dir) -> Tensor [num_nodes, k]   (cached by graph hash)
#   - centralities(data) -> Tensor [num_nodes, c]
#   - spectral_communities(data, n_clusters) -> Tensor [num_nodes]
