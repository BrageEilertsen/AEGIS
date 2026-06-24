# AEGIS

**Adversarially-robust, Explainable, Graph-based Intelligence System for financial-crime detection.**

AEGIS detects money-laundering patterns in transaction networks using a Graph Neural Network
(GNN) trained from scratch, and exposes the results through an interactive web app. Money
laundering looks like a *shape* in a network (smurfing fan-outs, layering chains, circular
flows) rather than a single bad transaction — graph models catch what per-transaction models
miss. Every flag ships with a faithful explanation (salient edges/features + matched laundering
typology), and the system demonstrates adversarial robustness: a naïve model fooled by graph
perturbation, a hardened model that holds.

The detailed build specification and the cluster operational ground rules are kept as local,
uncommitted working documents (they contain environment-specific infrastructure details); they
are intentionally not part of this public repository.

## Repository map

| Path | Contents |
|---|---|
| `ml/` | Python ML core — PyTorch + PyTorch Geometric. Features, models, training, eval. |
| `ml/data/` | Dataset loading + graph construction (transaction-as-node / account-as-node). |
| `ml/features/` | Raw, local, and spectral (Laplacian PE, centralities) feature pipelines. |
| `ml/models/` | GCN → GraphSAGE → GAT (→ temporal), behind a common interface. |
| `ml/explain/` | GNNExplainer / PGExplainer / attention + typology matching. |
| `ml/adversarial/` | Structural evasion attacks + defenses. |
| `data/` | Dataset download scripts + prep (raw data is gitignored). |
| `experiments/` | YAML configs — the single source of truth for hyperparameters. |
| `cluster/` | `train.slurm`, `environment.yml`, cluster notes. |
| `outputs/`, `logs/` | Per-run outputs and Slurm logs (gitignored). |
| `cache/features/` | Persistent spectral-feature cache, keyed by graph hash (gitignored). |
| `inference/` | Python FastAPI inference service (built on the Mac, later phase). |
| `api/` | Java 21 / Spring Boot backend (built on the Mac, later phase). |
| `frontend/` | Angular + Cytoscape.js UI (built on the Mac, later phase). |
| `prototype/` | Streamlit/Gradio intermediate prototype. |
| `infra/` | Dockerfiles, docker-compose, Azure deploy config. |

Only the ML core (`ml/`, `data/`, `experiments/`, `cluster/`) is developed on the Simula
cluster. The Java backend, Angular frontend, and FastAPI service are built locally on the Mac.

## Running training (on the cluster, via Slurm)

The single training entrypoint exposes a stable CLI:

```bash
python ml/train.py --config <yaml> --out-dir <dir> --feature-cache <dir> --seed <int>
```

Heavy training runs through Slurm on a GPU node (the login node has no usable GPU):

```bash
mkdir -p /home/brageei/AEGIS/logs          # must exist before submitting
sbatch cluster/train.slurm experiments/gcn_baseline.yaml
```

Monitor with `squeue -u $USER` and `tail -f logs/aegis_train-<jobid>.out`.

## Environment

Python venv at `/home/brageei/AEGIS/env`, PyTorch + PyG on CUDA **12.1** (`cu121` wheels). The
compiled PyG extras (`torch_scatter`, `torch_sparse`) must be installed and verified on a GPU
node — not on the login node. See [`cluster/`](cluster/) for pinned dependencies.

## Building the graph (Phase 1)

Once `data/raw/LI-Small_Trans.csv` is downloaded (`bash data/download_ibm_aml.sh`, needs a Kaggle
token), build and cache the PyG graph(s) and print stats:

```bash
source env/bin/activate
python ml/data/build_graph.py --config experiments/gcn_baseline.yaml --graph both
```

No data yet? Smoke-test the whole pipeline offline on a synthetic IBM-AML-schema frame:

```bash
python ml/data/build_graph.py --config experiments/gcn_baseline.yaml --synthetic --graph both
python tests/test_phase1_graph.py        # correctness tests (edges, Δt window, temporal split)
```

Built graphs are cached content-addressed under `data/processed/` (gitignored), keyed by variant,
graph view, Δt, subsampling, and split — so they are reused across runs.

## Training & evaluation

```bash
python ml/train.py --config experiments/gcn_baseline.yaml --out-dir outputs/gcn --feature-cache cache/features --seed 42
python ml/eval.py  --run-dir outputs/gcn --split test --feature-cache cache/features
python ml/explain/cli.py --run-dir outputs/gcn --node <idx> --feature-cache cache/features   # explanation contract
python -m ml.adversarial --naive-ckpt <naive>/best.pt --hardened-ckpt <hardened>/best.pt \
    --feature-cache cache/features --out-dir outputs/adv --smoke-test                          # before/after artifact
```

Each config (`experiments/*.yaml`) is the single source of truth for its run. Every `_synthetic`
config runs on the login-node CPU with no download; the `lismall` / `elliptic` configs run on a GPU
node via `cluster/*.slurm`. The full test suite (`tests/test_phase*.py`, 47 tests) runs on CPU.

## Status — cluster ML core (Phases 1–5) complete

All verified on synthetic/fixture data on the login-node CPU; the same code runs on the real
datasets via Slurm once they are downloaded.

- **Phase 0 — scaffold, venv, Slurm, stable train CLI.**
- **Phase 1 — data & graph construction:** IBM-AML loader, transaction/account graphs, configurable
  Δt, temporal 60/20/20 split, content-addressed cache (`ml/data/`).
- **Phase 2 — spectral features + GCN:** raw + local + Laplacian-PE/centrality features
  (per-connected-component, train-only standardized, cached), GCN, PR-AUC / recall@precision /
  F1-illicit metrics (`ml/features/`, `ml/models/gcn.py`, `ml/common.py`).
- **Phase 3 — GraphSAGE + GAT + Elliptic1 benchmark:** common `forward(data)` interface, GAT
  per-edge attention, inductive-sampling option, Elliptic loader (`ml/models/`, `ml/data/elliptic.py`).
- **Phase 4 — explainability:** GNNExplainer + GAT attention + feature attribution + k-hop subgraph
  + typology matching → versioned explanation contract (`ml/explain/`).
- **Phase 5 — adversarial robustness:** structural evasion attack + adversarial-training/self-loop
  defense + reproducible before/after artifact (`ml/adversarial/`).

**Next (built on the Mac, not this cluster):** Phase 6 Streamlit/Gradio prototype, Phase 7 Spring
Boot + Angular + FastAPI, Phase 8 Azure deploy. Build order & acceptance criteria per spec §12.
