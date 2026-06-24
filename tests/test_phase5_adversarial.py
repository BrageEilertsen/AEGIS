"""Phase 5 correctness tests — structural attack, adversarial-training defense, before/after artifact.

Login-CPU, synthetic. Standalone: python tests/test_phase5_adversarial.py | pytest tests/test_phase5_adversarial.py
The end-to-end artifact test trains two small models inline (~1-2 min); component tests are fast.
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

import ml.train as train
from ml.adversarial import (
    AttackConfig, GreedyEdgeAttack, make_adversarial_helpers, query_scores,
    run, select_target_nodes, train_epoch_adversarial, validate_robust_aggregation,
)
from ml.adversarial.contract import AdversarialArtifactContract
from ml.adversarial.featurize import make_refeaturizer
from ml.common import build_loss, compute_class_weights
from ml.data.graph import build_transaction_graph
from ml.data.loaders import make_synthetic_aml
from ml.features.assemble import assemble_features
from ml.models import build_model

DT = 24 * 3600
_FC = tempfile.mkdtemp()
_FCFG = {"laplacian_pe_k": 6, "centralities": ["pagerank", "eigenvector"], "max_degree": 2000}


@lru_cache(maxsize=1)
def _setup():
    df = make_synthetic_aml(n_legit=400, seed=0)
    data = build_transaction_graph(df, DT)
    data, meta = assemble_features(data, df, _FCFG, _FC)
    model = build_model({"model": "gat", "arch": {"hidden_channels": 16, "num_layers": 2,
                                                  "heads": 4, "variant": "gat_v2"}}, data.x.size(1))
    model.eval()
    refeat, raw_x, node_amount = make_refeaturizer(
        df, _FCFG, meta["standardization"], data.y.cpu(),
        {n: data[n].cpu() for n in ("train_mask", "val_mask", "test_mask")}, torch.device("cpu"))
    return df, data, meta, model, refeat, raw_x, node_amount


def _attack():
    df, data, meta, model, refeat, raw_x, node_amount = _setup()
    cfg = AttackConfig(top_k=5, score_threshold=0.0, budget_frac=0.5, budget_max=4)
    atk = GreedyEdgeAttack(model, data.edge_index.cpu(), raw_x, node_amount, refeat, cfg)
    targets = select_target_nodes(query_scores(model, data), data.y, data.test_mask, cfg)
    return atk, data, targets


# ---- attack ----
def test_targets_test_only():
    _, data, _, model, *_ = _setup()
    cfg = AttackConfig(top_k=10, score_threshold=0.0)
    targets = select_target_nodes(query_scores(model, data), data.y, data.test_mask, cfg)
    tm = data.test_mask.cpu().numpy()
    assert targets and all(tm[t] and int(data.y[t]) == 1 for t in targets)


def test_budget_and_netflow_respected():
    atk, data, targets = _attack()
    _, records = atk.run(targets)
    for r in records:
        assert r["budget"] <= atk.cfg.budget_max
        assert r["n_edits"] <= r["budget"]
        assert r["net_flow_drift"] <= atk.cfg.net_flow_tol


def test_delta_non_positive():
    atk, data, targets = _attack()
    _, records = atk.run(targets)
    assert records and all(r["delta"] <= 1e-6 for r in records)  # attack never raises the score


def test_greedy_determinism():
    runs = []
    for _ in range(2):
        atk, _, targets = _attack()
        _, records = atk.run(targets)
        runs.append([r["edits"] for r in records])
    assert runs[0] == runs[1]


# ---- defense ----
def test_validate_robust_aggregation():
    validate_robust_aggregation({"model": "graphsage", "arch": {"aggregator": "median"}})
    validate_robust_aggregation({"model": "gat", "arch": {"aggregator": "mean"}})
    try:
        validate_robust_aggregation({"model": "gat", "arch": {"aggregator": "median"}})
        assert False, "median on non-graphsage should raise"
    except ValueError:
        pass


def test_make_helpers_none_without_config():
    df, data, meta, *_ = _setup()
    assert make_adversarial_helpers({"train": {}}, df, data, _FCFG, meta["standardization"],
                                    torch.device("cpu")) is None


def test_train_epoch_adversarial_runs():
    df, data, meta, model, *_ = _setup()
    cfg = {"train": {"adversarial_training": {"fraction": 0.5, "budget_frac": 0.3}}}
    helpers = make_adversarial_helpers(cfg, df, data, _FCFG, meta["standardization"], torch.device("cpu"))
    assert helpers is not None
    loss_fn = build_loss({"loss": "weighted_ce"}, compute_class_weights(data.y, data.train_mask))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = train_epoch_adversarial(model, data, helpers, opt, loss_fn, 1.0, epoch=1, seed=42)
    assert loss == loss and loss >= 0  # finite, non-negative


# ---- end-to-end before/after (trains two small models) ----
def _write_cfg(path, hardened: bool):
    arch = {"hidden_channels": 16, "num_layers": 2, "dropout": 0.5, "heads": 4,
            "variant": "gat_v2", "attention_dropout": 0.3}
    tr = {"loss": "weighted_ce", "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.0005,
          "max_epochs": 35, "grad_clip_norm": 1.0,
          "early_stopping": {"monitor": "val_pr_auc", "patience": 35}, "lr_schedule": "plateau"}
    if hardened:
        arch["add_self_loops"] = True
        tr["adversarial_training"] = {"fraction": 0.5, "budget_frac": 0.5, "budget_max": 8, "seed": 42}
    cfg = {"config_version": 1, "device": "cpu", "model": "gat", "arch": arch,
           "dataset": {"name": "ibm_aml", "variant": "synthetic", "synthetic": True,
                       "synthetic_legit": 500, "graph": "transaction_as_node", "delta_t_hours": 24,
                       "max_out_per_in": 200, "graph_cache_dir": tempfile.mkdtemp()},
           "features": {"spectral": {"laplacian_pe_k": 6, "centralities": ["pagerank", "eigenvector"]}},
           "split": {"type": "temporal", "ratios": [0.6, 0.2, 0.2]},
           "train": tr, "eval": {"min_precision": 0.9}}
    Path(path).write_text(yaml.safe_dump(cfg))


def test_artifact_before_after_robustness():
    d = Path(tempfile.mkdtemp())
    fc = str(d / "fc")
    _write_cfg(d / "naive.yaml", hardened=False)
    _write_cfg(d / "hard.yaml", hardened=True)
    train.main(["--config", str(d / "naive.yaml"), "--out-dir", str(d / "naive"),
                "--feature-cache", fc, "--seed", "42"])
    train.main(["--config", str(d / "hard.yaml"), "--out-dir", str(d / "hard"),
                "--feature-cache", fc, "--seed", "42"])
    cfg = AttackConfig(top_k=6, score_threshold=0.4, budget_frac=0.5, budget_max=5, seed=42)
    contract = run(d / "naive" / "best.pt", d / "hard" / "best.pt", fc, d / "art", cfg, seed=42)

    # Code-correctness invariants (the robustness GAP itself is an empirical result demonstrated by
    # the artifact on realistic data — see cluster/adversarial.slurm — not a unit-test invariant).
    js = json.loads(contract.to_json())                 # JSON-serializable + round-trips
    assert js["schema_version"] == "1.0"
    assert AdversarialArtifactContract.from_dict(js).seed == 42
    deg = contract.degradation
    for key in ("naive_attack_success_rate", "hardened_attack_success_rate",
                "target_robustness_gap", "naive_mean_score_drop", "hardened_mean_score_drop"):
        assert isinstance(deg[key], (int, float))
    assert contract.attack["n_targets"] > 0 and contract.per_target            # targets attacked
    assert all(r["naive_delta"] <= 1e-6 for r in contract.per_target)          # attack never raises score
    assert contract.constraint_violations == []                               # net flow preserved
    assert "schema_version" in js and js["summary"]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
