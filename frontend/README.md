# AEGIS frontend (Angular + Cytoscape.js)

Interactive analyst UI (spec §8.4): metrics panel, flagged-transaction list, the capped flagged
subgraph (Cytoscape, never the full graph), a slide-in explanation panel (score, matched typology,
top features/edges), and the adversarial before/after demo.

```bash
cd frontend && npm install
npm start            # ng serve on http://localhost:4200 (expects the Spring API on :8080)
```
Talks to the Spring Boot BFF (`http://localhost:8080/api`); override via `window.AEGIS_API`.
