# AEGIS inference service image. CPU torch (the demo graph fits CPU); native scatter -> no
# torch_scatter/torch_sparse needed. The trained checkpoint + caches are volume-mounted (gitignored,
# large), not baked into the image.
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch_geometric>=2.5 networkx scipy scikit-learn pandas pyyaml \
    fastapi "uvicorn[standard]" httpx
COPY ml/ ./ml/
COPY inference/ ./inference/
ENV PYTHONUNBUFFERED=1 AEGIS_FEATURE_CACHE=/data/cache/features
EXPOSE 8000
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
