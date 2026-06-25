# AEGIS inference service image — fully self-contained (CPU).
# Bakes the narration model + the trained checkpoint + the pre-built featurized graph, so the
# container runs identically with NO host mounts and NO 6.9M-row CSV (locally and on Azure).
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch_geometric>=2.5 networkx scipy scikit-learn pandas pyyaml \
    fastapi "uvicorn[standard]" httpx transformers huggingface_hub

# Grounded plain-English summary model (override with --build-arg AEGIS_LLM_MODEL=...).
ARG AEGIS_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${AEGIS_LLM_MODEL}'); AutoModelForCausalLM.from_pretrained('${AEGIS_LLM_MODEL}')"

# Trained checkpoint + pre-built graph from the public HF model repo (best.pt ~16KB, graph ~33MB).
ARG AEGIS_HF_REPO=bragee/AEGIS
RUN python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; \
    os.makedirs('/app/baked/gcn_lismall', exist_ok=True); \
    shutil.copy(d('${AEGIS_HF_REPO}','gcn-li-small/best.pt'),'/app/baked/gcn_lismall/best.pt'); \
    shutil.copy(d('${AEGIS_HF_REPO}','gcn-li-small/prebuilt_graph.pt'),'/app/baked/gcn_lismall/prebuilt_graph.pt')"

COPY ml/ ./ml/
COPY inference/ ./inference/
COPY infra/seed/ ./baked/seed/

ENV PYTHONUNBUFFERED=1 \
    AEGIS_CHECKPOINT=/app/baked/gcn_lismall/best.pt \
    AEGIS_PREBUILT_GRAPH=/app/baked/gcn_lismall/prebuilt_graph.pt \
    AEGIS_ADVERSARIAL_ARTIFACT=/app/baked/seed/demo_adversarial_artifact.json \
    AEGIS_FEATURE_CACHE=/app/cache/features \
    AEGIS_LLM_SUMMARY=1
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
