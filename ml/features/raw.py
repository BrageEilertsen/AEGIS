"""Raw / intrinsic node features (spec §7.2, group 1).

Transaction amount, currency, payment format/type, timestamp encodings, sender/receiver bank
identifiers (hashed/encoded), and any dataset-provided columns (Elliptic ships 166).

Phase 0: placeholder. Implemented in Phase 2.
"""
from __future__ import annotations

# TODO (Phase 2): build_raw_features(df) -> Tensor [num_nodes, d_raw]
