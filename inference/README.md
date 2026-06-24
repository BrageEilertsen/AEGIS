# AEGIS inference service (FastAPI)

Thin model server wrapping the trained PyG checkpoint (spec §8.2). The Spring Boot BFF calls it;
it owns no product logic.

## Run
```bash
source ../env/bin/activate            # repo venv (torch cu121 + PyG + the ml/ package)
pip install -r requirements.txt
AEGIS_CHECKPOINT=../outputs/<run>/best.pt AEGIS_FEATURE_CACHE=../cache/features \
  uvicorn inference.app:app --host 0.0.0.0 --port 8000   # run from the repo root
# optional: AEGIS_ADVERSARIAL_ARTIFACT=outputs/adversarial_<job>/artifact.json
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET  | `/health`, `/info` | model + graph summary |
| GET  | `/metrics?split=test` | PR-AUC / recall@precision / F1 / confusion (illicit class) |
| GET  | `/flags?threshold=&limit=` | flagged transactions (score ≥ threshold), highest first |
| POST | `/score` `{node_ids?}` | per-node illicit probability |
| POST | `/explain` `{node_id, method, num_hops, max_nodes}` | ExplanationContract (spec §7.7) |
| POST | `/adversarial` `{artifact_path?}` | before/after AdversarialArtifactContract (spec §7.8) |

Reuses `ml/` (explain_node, adversarial, compute_metrics) — no reimplementation. Startup rebuilds
the graph + features with the checkpoint's saved standardization (no leakage), then caches scores.
