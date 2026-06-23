"""Typology matching: label the explained subgraph as a known laundering pattern (spec §7.7).

Structural rules over the relabeled, capped k-hop subgraph (≤ max_nodes, so small-graph algorithms
are fine here — never run on the full graph). Each rule yields a confidence in [0, 1]; the best
match wins, else "unknown". On synthetic data the injected pattern labels serve as a test oracle;
on real data the IBM patterns file can supply ground truth, but the rules always work standalone.

NOTE on representation: in the transaction-as-node graph an edge goes from an earlier transaction
to a later one (flow forward in time), so the graph is effectively ACYCLIC — a "circular" flow at
the *account* level (A→B→C→A) appears here as a forward *path* of transactions, not a directed
cycle. Hence on this graph the reliably-detectable typology is the **layering chain** (a long
directed path of pass-through transactions); fan-in / fan-out / structuring / circular are
account-level shapes, best seen on the account-as-node graph (graph.build_account_graph) — the
rules below still encode them faithfully for that view and for explicit account subgraphs.
"""
from __future__ import annotations

PATTERNS = ("fan_in", "fan_out", "structuring", "layering_chain", "circular")


def _neighbors(edge_index):
    src, dst = edge_index[0], edge_index[1]
    out, inc = {}, {}
    for s, d in zip(src, dst):
        out.setdefault(s, []).append(d)
        inc.setdefault(d, []).append(s)
    return out, inc


def _fan_out_score(out, inc, t) -> float:
    out_t = len(out.get(t, []))
    in_t = len(inc.get(t, []))
    if out_t == 0:
        return 0.0
    distinct = len(set(out[t]))
    return min(out_t / 6.0, 1.0) * (1.0 - in_t / (out_t + 1)) * (distinct / out_t)


def _fan_in_score(out, inc, t) -> float:
    in_t = len(inc.get(t, []))
    out_t = len(out.get(t, []))
    if in_t == 0:
        return 0.0
    distinct = len(set(inc[t]))
    return min(in_t / 6.0, 1.0) * (1.0 - out_t / (in_t + 1)) * (distinct / in_t)


def _structuring_score(out, inc, t, amounts=None) -> float:
    # Needs per-edge amounts to measure near-threshold uniformity; unavailable on the transaction-
    # node graph, so fall back to the fan-out shape at reduced weight (documented heuristic).
    return _fan_out_score(out, inc, t) * 0.6


_LONGEST_PATH_BUDGET = 20000   # DFS-step cap; keeps cyclic/dense subgraphs bounded (subgraph ≤ max_nodes)


def _longest_path_through(out, inc, t) -> int:
    """Node count of the longest directed SIMPLE path through t.

    Explores each path under its own visited set (no cross-path memo), so it is correct even when
    the subgraph contains cycles — exact on the acyclic transaction-flow graph (the common case) and
    a bounded lower bound on cyclic graphs (where the `circular` rule takes priority anyway). A step
    budget guards against exponential blow-up on pathologically dense capped subgraphs.
    """
    def longest(adj, start) -> int:
        best, calls = 1, 0
        stack = [(start, 1, frozenset((start,)))]
        while stack:
            calls += 1
            if calls > _LONGEST_PATH_BUDGET:
                break
            u, depth, visited = stack.pop()
            if depth > best:
                best = depth
            for v in adj.get(u, []):
                if v not in visited:
                    stack.append((v, depth + 1, visited | {v}))
        return best
    # On a DAG the forward and backward longest paths from t share only t, so summing is exact;
    # on a cycle this may over-count, but `circular` outranks `layering_chain` in the tie-break.
    return longest(out, t) + longest(inc, t) - 1


def _layering_chain_score(out, inc, t) -> float:
    return max(0.0, min((_longest_path_through(out, inc, t) - 2) / 3.0, 1.0))


def _circular_score(edge_index, t) -> float:
    try:
        import networkx as nx
    except ImportError:
        return 0.0
    if not edge_index[0]:
        return 0.0
    g = nx.DiGraph()
    g.add_edges_from(zip(edge_index[0], edge_index[1]))
    if t not in g:
        return 0.0
    for comp in nx.strongly_connected_components(g):
        if t in comp and len(comp) >= 2:
            return 1.0
    return 0.0


def match_typology(subgraph: dict, pattern_labels=None) -> dict:
    """Return {label, confidence, justification, scores{...}, ground_truth}."""
    ei = subgraph["edge_index"]
    t = subgraph["_target_rel"]
    out, inc = _neighbors(ei)

    scores = {
        "fan_out": _fan_out_score(out, inc, t),
        "fan_in": _fan_in_score(out, inc, t),
        "structuring": _structuring_score(out, inc, t),
        "layering_chain": _layering_chain_score(out, inc, t),
        "circular": _circular_score(ei, t),
    }
    # Tie-break by specificity (a directed cycle also looks like a pass-through chain): a cycle is
    # the more specific finding, so circular wins ties over layering_chain, etc. max() returns the
    # first element of PRIORITY achieving the maximum score.
    PRIORITY = ("circular", "layering_chain", "fan_in", "fan_out", "structuring")
    label = max(PRIORITY, key=lambda k: scores[k])
    confidence = scores[label]
    if confidence < 0.5:
        label, confidence = "unknown", confidence

    out_t, in_t = len(out.get(t, [])), len(inc.get(t, []))
    justifications = {
        "fan_out": f"Target fans out to {len(set(out.get(t, [])))} distinct receivers (out={out_t}, in={in_t}).",
        "fan_in": f"Target fans in from {len(set(inc.get(t, [])))} distinct senders (in={in_t}, out={out_t}).",
        "structuring": f"Fan-out shape suggestive of structuring (out={out_t}); amounts unavailable to confirm.",
        "layering_chain": f"Directed pass-through chain of {_longest_path_through(out, inc, t)} transactions through the target.",
        "circular": "Target lies on a directed cycle within its neighbourhood.",
        "unknown": "No laundering typology matched above the confidence threshold.",
    }

    ground_truth = None
    if pattern_labels is not None:
        ground_truth = str(pattern_labels[subgraph["target_node_id"]])

    return {"label": label, "confidence": round(float(confidence), 4),
            "justification": justifications[label],
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "ground_truth": ground_truth}
