# AEGIS inference service image. CPU torch (the demo graph fits CPU); native scatter -> no
# torch_scatter/torch_sparse needed. The trained checkpoint + caches are volume-mounted (gitignored,
# large), not baked into the image.
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch_geometric>=2.5 networkx scipy scikit-learn pandas pyyaml \
    fastapi "uvicorn[standard]" httpx transformers
# Bake the small narration model into the image so there is no runtime download (offline-friendly,
# ~1GB). Override the repo id with --build-arg AEGIS_LLM_MODEL=... to bake a different model.
ARG AEGIS_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('${AEGIS_LLM_MODEL}'); AutoModelForCausalLM.from_pretrained('${AEGIS_LLM_MODEL}')"
COPY ml/ ./ml/
COPY inference/ ./inference/
ENV PYTHONUNBUFFERED=1 AEGIS_FEATURE_CACHE=/data/cache/features
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
