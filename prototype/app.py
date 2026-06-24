"""AEGIS — Streamlit prototype (spec §9): the intermediate UI de-risk over the FastAPI service.

Loads dataset/model info, lists flagged transactions, shows a faithful per-flag explanation
(score, matched typology, top features/edges, neighbourhood subgraph), and the adversarial
before/after — all by calling the inference service. Reimplemented properly later as Angular +
Spring Boot; this proves the end-to-end story early.

    AEGIS_API_URL=http://localhost:8000 streamlit run prototype/app.py
(start the service first: AEGIS_CHECKPOINT=outputs/<run>/best.pt uvicorn inference.app:app --port 8000)
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("AEGIS_API_URL", "http://localhost:8000")
st.set_page_config(page_title="AEGIS — AML graph detection", layout="wide")


@st.cache_data(ttl=30)
def api_get(path: str, **params):
    return httpx.get(f"{API}{path}", params=params, timeout=120).json()


def api_post(path: str, body: dict):
    return httpx.post(f"{API}{path}", json=body, timeout=600).json()


def _subgraph_dot(sg: dict) -> str:
    """Build a Graphviz DOT string for the neighbourhood subgraph (target highlighted)."""
    ids = sg["node_ids"]
    target = sg["target_node_id"]
    labels = sg.get("node_labels", [0] * len(ids))
    scores = sg.get("node_scores", [0.0] * len(ids))
    lines = ["digraph G {", "rankdir=LR; node [shape=circle, style=filled, fontsize=8];"]
    for i, nid in enumerate(ids):
        if nid == target:
            color = "#e74c3c"             # target: red
        elif labels[i] == 1:
            color = "#e67e22"             # known illicit: orange
        else:
            g = int(255 * (1 - min(scores[i], 1.0)))
            color = f"#ff{g:02x}{g:02x}"  # score gradient toward red
        lines.append(f'  "{nid}" [fillcolor="{color}"];')
    src, dst = sg["edge_index"] if sg["edge_index"] else ([], [])
    for s, d in zip(src, dst):
        lines.append(f'  "{ids[s]}" -> "{ids[d]}";')
    lines.append("}")
    return "\n".join(lines)


st.title("🛡️ AEGIS — money-laundering detection on transaction graphs")
st.caption(f"GNN-based detection · explainable flags · adversarial robustness · API: {API}")

try:
    info = api_get("/health")
except Exception as e:
    st.error(f"Cannot reach the inference service at {API} — start it first. ({e})")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", info["model"].upper())
c2.metric("Transactions", f"{info['num_nodes']:,}")
c3.metric("Flow edges", f"{info['num_edges']:,}")
c4.metric("Illicit (labelled)", f"{info['num_illicit']:,}")

tab_metrics, tab_flags, tab_adv = st.tabs(["📊 Metrics", "🚩 Flagged + explanation", "⚔️ Adversarial demo"])

# ---- metrics ----
with tab_metrics:
    m = api_get("/metrics", split="test")
    st.subheader("Held-out test performance (illicit class)")
    a, b, c, d = st.columns(4)
    a.metric("PR-AUC", f"{m['pr_auc']:.3f}" if m['pr_auc'] is not None else "n/a")
    b.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m['roc_auc'] is not None else "n/a")
    rp = m["recall_at_precision"]
    c.metric(f"Recall @ p≥{m.get('min_precision', 0.9)}", f"{rp:.3f}" if rp is not None else "0.000")
    d.metric("F1 (illicit)", f"{m['f1_illicit']:.3f}" if m['f1_illicit'] is not None else "n/a")
    st.caption("Accuracy is meaningless at ~2% positives — PR-AUC and recall-at-precision are the headline.")
    cm = m.get("confusion_matrix")
    if cm:
        st.write("Confusion matrix (at the recall-at-precision threshold):")
        st.table({"": ["actual licit", "actual illicit"],
                  "pred licit": [cm["tn"], cm["fn"]], "pred illicit": [cm["fp"], cm["tp"]]})

# ---- flags + explanation ----
with tab_flags:
    thr = st.slider("Flag threshold (illicit probability)", 0.0, 1.0, 0.5, 0.05)
    flags = api_get("/flags", threshold=thr, limit=100)
    st.write(f"**{len(flags)} flagged transactions** (showing top by score)")
    st.dataframe(flags, use_container_width=True, height=240)
    if flags:
        node = st.selectbox("Explain a flagged transaction (node id)", [f["node_id"] for f in flags])
        if st.button("Explain"):
            with st.spinner("Computing faithful explanation (GNNExplainer + typology)…"):
                ex = api_post("/explain", {"node_id": int(node), "num_hops": 2, "max_nodes": 150})
            left, right = st.columns([1, 1])
            with left:
                st.metric("Illicit score", f"{ex['score']:.3f}")
                ty = ex["matched_typology"]
                st.write(f"**Matched typology:** `{ty['label']}` (confidence {ty['confidence']})")
                st.caption(ty["justification"])
                st.write("**Top contributing features**")
                st.bar_chart({f["column_name"]: f["importance"] for f in ex["top_features"]})
                st.caption(f"Edge importance via *{ex['faithfulness']['edge_importance_source']}*. "
                           + ex["faithfulness"]["note"])
            with right:
                st.write("**Responsible neighbourhood** (flagged node in red)")
                st.graphviz_chart(_subgraph_dot(ex["neighborhood_subgraph"]))

# ---- adversarial ----
with tab_adv:
    st.subheader("Adversarial robustness — naïve fooled vs hardened holds")
    art = api_post("/adversarial", {})
    if isinstance(art, dict) and art.get("degradation"):
        d = art["degradation"]
        st.write(art.get("summary", ""))
        a, b = st.columns(2)
        a.metric("Naïve attack success", f"{d['naive_attack_success_rate']:.0%}",
                 help="fraction of flagged transactions pushed below threshold by the structural attack")
        b.metric("Hardened attack success", f"{d['hardened_attack_success_rate']:.0%}",
                 delta=f"{-d['target_robustness_gap']:.0%}", delta_color="inverse")
        st.json(art["degradation"])
    else:
        st.info("No precomputed adversarial artifact. Generate one with "
                "`python -m ml.adversarial --naive-ckpt … --hardened-ckpt … --out-dir <dir>` "
                "and set AEGIS_ADVERSARIAL_ARTIFACT on the service.")
