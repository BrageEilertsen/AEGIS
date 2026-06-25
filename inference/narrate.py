"""Plain-English narration of an explanation contract.

Two layers, both GROUNDED in the structured evidence the GNN explainer already produced (score,
predicted label, matched typology, top features, salient edges) — nothing is invented:

  * ``template_summary``  deterministic, instant, no model. Always available; rendered immediately.
  * ``llm_summary``       a small local HF model (Qwen2.5-0.5B-Instruct) rephrases the SAME evidence
                          into fluent prose. Slower (autoregressive on CPU), so the service runs it
                          in the background and the UI upgrades the template text once it's ready.

To keep the LLM faithful it is given a glossary of what each feature GROUP means and told to refer
to features only via those descriptions — so it can't hallucinate that, say, a Laplacian positional
encoding "indicates money laundering". Loaded in bfloat16 (~1GB, half of fp32) so it fits alongside
the graph + torch runtime in a small container.

Config:
  AEGIS_LLM_SUMMARY=1|0          enable/disable the LLM (default on; off -> template only)
  AEGIS_LLM_MODEL=<hf repo id>   instruct model to use (default Qwen/Qwen2.5-0.5B-Instruct)
"""
from __future__ import annotations

import functools
import os

LLM_ENABLED = os.environ.get("AEGIS_LLM_SUMMARY", "1").lower() in ("1", "true", "yes")
_MODEL = os.environ.get("AEGIS_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Plain-English meaning of each feature GROUP (from feature_meta.group_dims). The LLM is restricted
# to these descriptions so it never invents what an individual engineered feature "means".
_GROUP_GLOSSARY = {
    "raw": "raw transaction attributes (amount, currency, payment format, timing)",
    "local": "local graph structure around the transaction (its in/out degree and immediate mixing)",
    "spectral_pe": "the transaction's structural position in the payment network (spectral/positional encoding)",
    "centralities": "how central this account is in the overall money-flow network (centrality)",
}

_SYSTEM = (
    "You are a financial-crime analyst assistant. In 2 short, plain-English sentences, explain to a "
    "compliance officer WHY this transaction was flagged, using ONLY the evidence provided. Refer to "
    "contributing factors using the plain descriptions given in the evidence — do NOT guess what a "
    "named feature means, and do NOT invent amounts, account names, counterparties, or reasons that "
    "are not in the evidence. Be concrete and concise."
)


@functools.lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL)
    # bfloat16 halves the weight footprint (~2GB fp32 -> ~1GB) so the model fits alongside the
    # graph + torch runtime in a small (2Gi) container without OOM-killing the worker; bf16 keeps
    # fp32's exponent range so greedy CPU decoding stays coherent. low_cpu_mem_usage avoids the
    # transient init+load doubling during from_pretrained.
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    return tok, model


def _feature_phrases(c: dict, k: int = 4) -> str:
    """Top features rendered as 'group description (importance)', deduped by group — keeps the LLM
    grounded in what each feature actually represents rather than its opaque column name."""
    seen, out = set(), []
    for f in c.get("top_features", []):
        g = f.get("group", "")
        if g in seen:
            continue
        seen.add(g)
        desc = _GROUP_GLOSSARY.get(g, g or "an engineered feature")
        out.append(f"{desc} (weight {float(f.get('importance', 0)):.2f})")
        if len(out) >= k:
            break
    return "; ".join(out) or "n/a"


def _trim_to_sentence(text: str) -> str:
    """Cut a generation back to its last complete sentence so a token limit never leaves a dangling
    fragment (e.g. '...indicating'). Falls back to the raw text if no sentence end is found."""
    end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "), text.rfind("."))
    return text[: end + 1].strip() if end > 40 else text.strip()


def _evidence(c: dict) -> str:
    typ = c.get("matched_typology", {}) or {}
    pred = "ILLICIT" if c.get("predicted_label") == 1 else "licit"
    edges = c.get("top_edges", []) or []
    # Phrase the score as a percentage band rather than a raw decimal — the small model otherwise
    # parrots/garbles the figure (e.g. inventing "0.0000"); a qualitative phrase is harder to corrupt.
    pct = 100 * float(c.get("score", 0))
    level = "very high" if pct >= 90 else "high" if pct >= 70 else "moderate"
    return (
        f"Flagged transaction #{c.get('node_id')}\n"
        f"Model's confidence it is illicit: {level} (about {pct:.0f}%); the model predicts it is {pred}.\n"
        f"Matched laundering typology: {typ.get('label', 'n/a')}\n"
        f"Main contributing factors: {_feature_phrases(c)}\n"
        f"Connectivity: {'linked to ' + str(len(edges)) + ' influential transactions' if edges else 'isolated — no connected neighbourhood'}"
    )


def template_summary(c: dict) -> str:
    """Deterministic, LLM-free summary — instant and always available."""
    typ = (c.get("matched_typology", {}) or {}).get("label", "an anomalous pattern")
    pct = round(100 * float(c.get("score", 0)), 1)
    groups, seen = [], set()
    for f in c.get("top_features", [])[:6]:
        g = f.get("group", "")
        if g and g not in seen:
            seen.add(g)
            groups.append(_GROUP_GLOSSARY.get(g, g))
        if len(groups) >= 2:
            break
    shape = (f"it matches the '{typ}' laundering typology" if c.get("top_edges")
             else "its own transaction features are anomalous (it has no connected neighbourhood)")
    feat_txt = (", driven mainly by " + " and ".join(groups)) if groups else ""
    return (f"Transaction #{c.get('node_id')} was flagged with {pct}% illicit probability because {shape}{feat_txt}. "
            f"See the highlighted features and edges below for the supporting evidence.")


def llm_summary(c: dict) -> str | None:
    """LLM rephrasing of the same grounded evidence; None if the model is disabled or unavailable."""
    if not LLM_ENABLED:
        return None
    try:
        import torch
        tok, model = _model()
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _evidence(c)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return _trim_to_sentence(text) or None
    except Exception:
        return None


def warm() -> None:
    """Preload the LLM weights (idempotent via lru_cache) so the first real summary pays only the
    generation cost, not the ~1GB load. Safe to call in a background thread at startup."""
    if LLM_ENABLED:
        try:
            _model()
        except Exception:
            pass


def narrate(contract: dict) -> str:
    """Best available summary, synchronously (LLM if enabled, else template). Kept for callers that
    want a single blocking call; the service uses template_summary + llm_summary for the async path."""
    return llm_summary(contract) or template_summary(contract)
