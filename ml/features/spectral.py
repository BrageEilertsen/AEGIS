"""Spectral / global node features — the technical signature (spec §7.2, group 3) — Phase 2.

Laplacian positional encodings + centralities on the (symmetrized) transaction graph:
  directed A  ->  A_sym = max(A, Aᵀ)  ->  symmetric normalized Laplacian
  L = I - D^{-1/2} A_sym D^{-1/2}  (symmetric, eigenvalues in [0, 2])  ->  smallest eigenpairs.

Design decisions (from the Phase-2 design synthesis):
- Symmetric-normalized Laplacian (not random-walk): symmetric, so scipy ``eigsh`` applies.
- ``which='SA'`` (smallest algebraic) — robust Lanczos, avoids the singular shift-invert that
  ``sigma=0`` would hit on a Laplacian (0 is an eigenvalue) and the unreliable ``which='SM'``.
- Dense ``numpy.linalg.eigh`` fallback for small graphs (<= DENSE_MAX nodes) — reliable, and the
  synthetic smoke graph lands here.
- Drop trivial eigenvectors (eigenvalue < TRIVIAL_TOL, one per connected component), keep the
  next k non-trivial, zero-pad to k if fewer exist, and canonicalize eigenvector signs.

These eigendecompositions are the expensive part and are cached by ml/features/assemble.py.
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

DENSE_MAX = 5000        # use dense eigh at/below this (per-component) node count
TRIVIAL_TOL = 1e-8      # eigenvalues below this are treated as trivial (~constant per component)


def graph_hash(data) -> str:
    """Stable structure-only hash: (num_nodes, canonicalized edge list, graph_kind).

    Independent of node features / feature config, so it is the cache namespace and detects any
    change to the graph topology.
    """
    ei = data.edge_index.cpu().numpy()
    order = np.lexsort((ei[1], ei[0]))
    canon = ei[:, order].astype(np.int64).tobytes()
    h = hashlib.sha1()
    h.update(str(int(data.num_nodes)).encode())
    h.update(str(getattr(data, "graph_kind", "?")).encode())
    h.update(canon)
    return h.hexdigest()[:16]


def _adjacency(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """Directed 0/1 adjacency as scipy CSR (no self-loops)."""
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    A = sp.coo_matrix((np.ones(row.shape[0], np.float64), (row, col)), shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _symmetrize_csr(edge_index: torch.Tensor, n: int) -> sp.csr_matrix:
    """A_sym = max(A, Aᵀ) as a 0/1 matrix — directed flow -> undirected topology."""
    A = _adjacency(edge_index, n)
    A = A.maximum(A.T)
    A.data[:] = 1.0
    return A.tocsr()


def _normalized_laplacian(A_sym: sp.csr_matrix) -> sp.csr_matrix:
    """L = I - D^{-1/2} A_sym D^{-1/2}; isolated nodes get D^{-1/2} = 0."""
    n = A_sym.shape[0]
    deg = np.asarray(A_sym.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = sp.diags(dinv_sqrt)
    return (sp.identity(n, format="csr") - D @ A_sym @ D).tocsr()


def _canonical_sign(vecs: np.ndarray) -> np.ndarray:
    """Per column, flip sign so the largest-|value| entry is positive (deterministic)."""
    if vecs.size == 0:
        return vecs
    idx = np.argmax(np.abs(vecs), axis=0)
    signs = np.sign(vecs[idx, np.arange(vecs.shape[1])])
    signs[signs == 0] = 1.0
    return vecs * signs


def _smallest_eigh(L: sp.csr_matrix, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the m smallest (eigenvalues, eigenvectors) of symmetric PSD L, ascending.

    Used per connected component, so m is small (<= k+1). Dense for small blocks; sparse Lanczos
    (smallest-algebraic) for a large component. No silent cap: callers bound m by component size.
    """
    n = L.shape[0]
    m = min(m, n)
    if n <= DENSE_MAX or m >= n - 1:
        w, v = np.linalg.eigh(L.toarray())
        return w[:m], v[:, :m]
    ncv = min(n - 1, max(2 * m + 1, 20))
    w, v = eigsh(L, k=m, which="SA", ncv=ncv, maxiter=5000, tol=1e-6)
    order = np.argsort(w)
    return w[order], v[:, order]


def laplacian_pe(data, k: int) -> tuple[torch.Tensor, np.ndarray]:
    """Laplacian positional encodings, computed PER CONNECTED COMPONENT: ([N, k] tensor, eigvals[k]).

    The transaction-as-node flow graph is highly fragmented (many short chains + isolated nodes),
    so a *global* eigendecomposition is both wrong and intractable: the trivial near-zero block has
    one vector per component (potentially tens of thousands), which would swamp any fixed eigsh
    budget and silently collapse the PE to zeros. Instead, within each connected component we take
    the k smallest NON-trivial eigenvectors of that component's normalized Laplacian (the global
    Laplacian is block-diagonal by component anyway), sign-canonicalize, and place them in that
    component's node rows. Singletons and components smaller than k+1 are zero-padded. Cost is
    bounded: only multi-node components are decomposed, each needing just k+1 small eigenpairs.

    Returns the PE matrix and, for diagnostics, the kept eigenvalues of the largest component.
    """
    n = int(data.num_nodes)
    A_sym = _symmetrize_csr(data.edge_index, n)
    ncomp, labels = connected_components(A_sym, directed=False)

    pe = np.zeros((n, k), dtype=float)
    rep_eigvals = np.zeros(k, dtype=float)      # eigenvalues of the largest component (for meta)
    largest = 0

    # Group nodes by component, then process only components with >= 2 nodes (skip singletons
    # in bulk — there can be millions of isolated transactions).
    order = np.argsort(labels, kind="stable")
    starts = np.unique(labels[order], return_index=True)[1]
    bounds = np.append(starts, n)
    sizes = np.diff(bounds)
    for pos in np.where(sizes >= 2)[0]:
        grp = order[bounds[pos]:bounds[pos + 1]]
        cn = grp.size
        sub = A_sym[grp][:, grp]
        Lsub = _normalized_laplacian(sub)
        w, v = _smallest_eigh(Lsub, min(k + 1, cn))   # +1 to clear this component's single trivial vec
        nz = np.where(w > TRIVIAL_TOL)[0][:k]
        if nz.size == 0:
            continue
        vv = _canonical_sign(v[:, nz])
        pe[np.ix_(grp, np.arange(vv.shape[1]))] = vv
        if cn > largest:
            largest = cn
            ev = w[nz]
            rep_eigvals = np.concatenate([ev, np.zeros(k - ev.size)])[:k]
    return torch.tensor(pe, dtype=torch.float), rep_eigvals


def _pagerank(edge_index: torch.Tensor, n: int, damping: float = 0.85,
              max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank via sparse power iteration on the DIRECTED graph; teleport handles dangling nodes."""
    A = _adjacency(edge_index, n)
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv = np.where(out_deg > 0, 1.0 / out_deg, 0.0)
    M = sp.diags(inv) @ A                    # row-stochastic (rows of dangling nodes are 0)
    Mt = M.T.tocsr()
    dangling = (out_deg == 0)
    p = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangle = damping * p[dangling].sum() / n
        p_new = (1.0 - damping) / n + damping * (Mt @ p) + dangle
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p


def _eigenvector_centrality(edge_index: torch.Tensor, n: int) -> np.ndarray:
    """|dominant eigenvector| of the symmetrized adjacency (eigsh which='LA')."""
    A_sym = _symmetrize_csr(edge_index, n)
    if n <= DENSE_MAX:
        w, v = np.linalg.eigh(A_sym.toarray())
        vec = v[:, -1]
    else:
        _, v = eigsh(A_sym, k=1, which="LA", maxiter=5000, tol=1e-6)
        vec = v[:, 0]
    vec = np.abs(vec)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def centralities(data, which: list[str]) -> tuple[torch.Tensor, list[str]]:
    """[N, c] centrality features + names. Supports 'pagerank', 'eigenvector'."""
    n = int(data.num_nodes)
    cols, names = [], []
    for name in which:
        if name == "pagerank":
            cols.append(_pagerank(data.edge_index, n))
        elif name == "eigenvector":
            cols.append(_eigenvector_centrality(data.edge_index, n))
        elif name == "betweenness":
            raise NotImplementedError(
                "betweenness centrality is O(nm) and intractable at scale — not supported "
                "(spec §7.2 allows skipping it; documented as deferred).")
        else:
            raise ValueError(f"unknown centrality '{name}'")
        names.append(f"centrality_{name}")
    if not cols:
        return torch.zeros((n, 0), dtype=torch.float), names
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float), names


def compute_spectral_features(data, k: int, which_centralities: list[str]) -> dict:
    """Pure compute (no I/O): the expensive group cached by assemble.py.

    Returns {'lap_pe':[N,k], 'centralities':[N,c], 'eigenvalues':[k], 'centrality_names':[...],
    'meta':{...}}.
    """
    pe, eigvals = laplacian_pe(data, k)
    cent, cent_names = centralities(data, which_centralities)
    n = int(data.num_nodes)
    A_sym = _symmetrize_csr(data.edge_index, n)
    ncomp, _ = connected_components(A_sym, directed=False)
    # PE coverage: fraction of nodes with any non-zero positional encoding. On a fragmented graph
    # (many singletons / tiny components) this is legitimately well below 1.0 — surfaced here so a
    # degenerate spectral signature is visible in run_context rather than silently all-zeros.
    pe_coverage = float((pe.abs().sum(dim=1) > 0).float().mean()) if k > 0 else 0.0
    return {
        "lap_pe": pe,
        "centralities": cent,
        "eigenvalues": torch.tensor(eigvals, dtype=torch.float),
        "centrality_names": cent_names,
        "meta": {"n_nodes": n, "n_components": int(ncomp), "k": int(k),
                 "pe_coverage": round(pe_coverage, 4),
                 "centralities": list(which_centralities)},
    }
