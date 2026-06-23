"""Explainability subsystem (spec §7.7) — required, not optional.

GNNExplainer / PGExplainer for the minimal responsible subgraph + feature subset, GAT attention
as edge-importance, and typology matching (structuring/smurfing, layering chain, fan-in/out,
circular flow) over the explained subgraph.

Output contract (consumed by the Java backend -> frontend explanation panel):
    { score, top_edges: [...], top_features: [...], matched_typology, neighborhood_subgraph }

Explanations must be FAITHFUL (reflect what the model used), not post-hoc rationalizations.

Phase 0: placeholder. Implemented in Phase 4.
"""
