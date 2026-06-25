"""Plain-English narration of an explanation contract via a small, local Hugging Face model.

The model is GROUNDED: it is given ONLY the structured evidence already produced by the GNN
explainer (score, predicted label, matched typology, top features, salient edges) and instructed
to invent nothing. This keeps the natural-language summary faithful to the model's actual reasons
— the whole point of an explainable system. Runs in-process on CPU (no external API, no key).

Config:
  AEGIS_LLM_SUMMARY=1|0          enable/disable the LLM (default on; off -> deterministic template)
  AEGIS_LLM_MODEL=<hf repo id>   instruct model to use (default Qwen/Qwen2.5-0.5B-Instruct, ~1GB)

If the model can't be loaded for any reason, narration falls back to a deterministic template built
straight from the contract, so the summary box always renders.
"""
from __future__ import annotations

import functools
import os

_ENABLED = os.environ.get("AEGIS_LLM_SUMMARY", "1").lower() in ("1", "true", "yes")
_MODEL = os.environ.get("AEGIS_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_SYSTEM = (
    "You are a financial-crime analyst assistant. In 2 short plain-English sentences, explain to a "
    "compliance officer WHY this transaction was flagged, using ONLY the evidence provided. Do not "
    "invent amounts, account names, or reasons not in the evidence. Be concrete and concise."
)


@functools.lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, torch_dtype=torch.float32)
    model.eval()
    return tok, model


def _evidence(c: dict) -> str:
    feats = ", ".join(f"{f['column_name']} ({f['importance']:.2f})" for f in c.get("top_features", [])[:5])
    edges = ", ".join(f"#{e['source_node']}->#{e['target_node']}" for e in c.get("top_edges", [])[:5])
    typ = c.get("matched_typology", {}) or {}
    pred = "ILLICIT" if c.get("predicted_label") == 1 else "licit"
    return (
        f"Flagged transaction #{c.get('node_id')}\n"
        f"Model score: {float(c.get('score', 0)):.4f} illicit probability (predicted {pred})\n"
        f"Matched laundering typology: {typ.get('label', 'n/a')} (confidence {typ.get('confidence', '?')})\n"
        f"Top contributing features: {feats or 'n/a'}\n"
        f"Most influential edges: {edges or 'none (isolated transaction)'}"
    )


def _template(c: dict) -> str:
    """Deterministic, LLM-free summary — always available as a fallback."""
    typ = (c.get("matched_typology", {}) or {}).get("label", "an anomalous pattern")
    pct = round(100 * float(c.get("score", 0)), 1)
    feats = [f["column_name"] for f in c.get("top_features", [])[:3]]
    shape = (f"it matches the '{typ}' laundering typology" if c.get("top_edges")
             else "its own transaction features are anomalous (it has no connected neighbourhood)")
    feat_txt = (", driven mainly by " + ", ".join(feats)) if feats else ""
    return (f"Transaction #{c.get('node_id')} was flagged with {pct}% illicit probability because {shape}{feat_txt}. "
            f"See the highlighted features and edges below for the supporting evidence.")


def narrate(contract: dict) -> str:
    """Return a 2-3 sentence grounded summary of the explanation contract."""
    if not _ENABLED:
        return _template(contract)
    try:
        import torch
        tok, model = _model()
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _evidence(contract)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=110, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return text or _template(contract)
    except Exception:
        return _template(contract)
