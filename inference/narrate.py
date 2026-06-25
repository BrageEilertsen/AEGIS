"""Plain-English narration of an explanation contract.

Two layers, both GROUNDED in the structured evidence the GNN explainer already produced (score,
predicted label, matched typology, top features, salient edges) — nothing is invented:

  * ``template_summary``  deterministic, instant, no model. Always available; rendered immediately.
  * ``llm_summary``       a proper instruct model rephrases the SAME evidence into fluent prose. By
                          default this calls a HOSTED model on Hugging Face's serverless inference
                          API (fast + reliable); if no token is set it falls back to a small LOCAL
                          transformers model (CPU, for offline/local dev). Either way the service
                          runs it in the background and the UI upgrades the template text when ready.

To keep the LLM faithful it is given a glossary of what each feature GROUP means and told to refer
to features only via those descriptions — so it can't hallucinate that, say, a Laplacian positional
encoding "indicates money laundering". If the LLM is disabled or errors, the grounded template is
used, so the summary box always renders something correct.

Config:
  AEGIS_LLM_SUMMARY=1|0   enable/disable the LLM rephrasing (default on; off -> template only)
  AEGIS_HF_TOKEN=hf_...   Hugging Face token with the "Inference Providers" permission -> use the
                          hosted API. Unset -> use the local transformers fallback.
  AEGIS_LLM_MODEL=<id>    instruct model id (default Qwen/Qwen2.5-7B-Instruct hosted; for the local
                          fallback set a small model like Qwen/Qwen2.5-0.5B-Instruct)
  AEGIS_HF_URL=<url>      OpenAI-compatible chat endpoint (default HF router)
"""
from __future__ import annotations

import functools
import os

LLM_ENABLED = os.environ.get("AEGIS_LLM_SUMMARY", "1").lower() in ("1", "true", "yes")
_HF_TOKEN = os.environ.get("AEGIS_HF_TOKEN", "").strip()
_HF_URL = os.environ.get("AEGIS_HF_URL", "https://router.huggingface.co/v1/chat/completions").strip()
_HOSTED = bool(_HF_TOKEN)
# Hosted default is a capable 7B; local fallback default stays tiny so it can run on a CPU dev box.
_MODEL = os.environ.get("AEGIS_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct" if _HOSTED
                        else "Qwen/Qwen2.5-0.5B-Instruct")

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
    pct = 100 * float(c.get("score", 0))
    level = "very high" if pct >= 90 else "high" if pct >= 70 else "moderate"
    return (
        f"Flagged transaction #{c.get('node_id')}\n"
        f"Model's confidence it is illicit: {level} (about {pct:.0f}%); the model predicts it is {pred}.\n"
        f"Matched laundering typology: {typ.get('label', 'n/a')}\n"
        f"Main contributing factors: {_feature_phrases(c)}\n"
        f"Connectivity: {'linked to ' + str(len(edges)) + ' influential transactions' if edges else 'isolated — no connected neighbourhood'}"
    )


def _hf_chat(system: str, user: str) -> str:
    """Call the hosted OpenAI-compatible chat endpoint on Hugging Face's inference router."""
    import httpx
    resp = httpx.post(
        _HF_URL, timeout=30.0,
        headers={"Authorization": f"Bearer {_HF_TOKEN}", "Content-Type": "application/json"},
        json={"model": _MODEL,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              "max_tokens": 180, "temperature": 0.2})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


@functools.lru_cache(maxsize=1)
def _local_model():
    """Small local transformers model — fallback when no hosted token is configured (local dev)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    return tok, model


def _local_summary(c: dict) -> str | None:
    import torch
    tok, model = _local_model()
    msgs = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": _evidence(c)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=96, do_sample=False,
                             repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return _trim_to_sentence(text) or None


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
    """LLM rephrasing of the same grounded evidence (hosted if a token is set, else local).
    Returns None if disabled or the model is unavailable — callers fall back to the template."""
    if not LLM_ENABLED:
        return None
    try:
        if _HOSTED:
            return _trim_to_sentence(_hf_chat(_SYSTEM, _evidence(c))) or None
        return _local_summary(c)
    except Exception:
        return None


def warm() -> None:
    """Preload the local fallback model so its first summary isn't slow. No-op for the hosted path
    (nothing to load) and safe to call in a background thread at startup."""
    if LLM_ENABLED and not _HOSTED:
        try:
            _local_model()
        except Exception:
            pass


def narrate(contract: dict) -> str:
    """Best available summary, synchronously (LLM if enabled, else template). Kept for callers that
    want a single blocking call; the service uses template_summary + llm_summary for the async path."""
    return llm_summary(contract) or template_summary(contract)
