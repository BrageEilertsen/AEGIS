"""Account-centric graph builder: node identity, label propagation, undirected edges, features."""
import pandas as pd

from ml.data.account_graph import FEATURE_NAMES, build_account_graph


def _frame():
    # 3 accounts: (1,A)=0, (1,B)=1, (2,C)=2.  B->C is laundering.
    return pd.DataFrame({
        "From Bank": [1, 1, 1],
        "Account": ["A", "B", "A"],
        "To Bank": [1, 2, 2],
        "Account.1": ["B", "C", "C"],
        "Amount Paid": [100.0, 5000.0, 250.0],
        "Payment Currency": ["USD", "USD", "EUR"],
        "Payment Format": ["ACH", "Wire", "ACH"],
        "Is Laundering": [0, 1, 0],
    })


def test_nodes_labels_and_edges():
    data = build_account_graph(_frame(), standardize=False)

    assert data.num_nodes == 3                       # A, B, C
    assert data.x.shape == (3, len(FEATURE_NAMES))

    # the laundering transaction B->C taints exactly accounts B and C
    assert int(data.y.sum()) == 2

    # 3 distinct directed pairs -> 6 undirected directed entries, and the edge set is symmetric
    assert data.edge_index.shape[1] == 6
    edges = {(int(a), int(b)) for a, b in zip(*data.edge_index.tolist())}
    assert all((b, a) in edges for (a, b) in edges)


def test_standardization_is_applied_by_default():
    data = build_account_graph(_frame())
    # standardized columns are ~zero-mean (within float tolerance) over the 3 nodes
    assert abs(float(data.x.mean())) < 1e-5
