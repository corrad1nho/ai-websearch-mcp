FROM python:3.12-slim AS runtime

ARG RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
ARG APP_HOME="/app"
ARG HF_HOME="/opt/hf-cache"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=${HF_HOME} \
    TRANSFORMERS_CACHE=${HF_HOME} \
    SENTENCE_TRANSFORMERS_HOME=${HF_HOME} \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    RERANK_MODEL=${RERANK_MODEL} \
    RERANK_DEVICE=cpu \
    TORCH_THREADS=6 \
    OMP_NUM_THREADS=6 \
    MKL_NUM_THREADS=6 \
    MCP_TRANSPORT=streamable-http \
    PORT=8000

WORKDIR ${APP_HOME}

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${APP_HOME} ${HF_HOME}

RUN pip install --upgrade pip --root-user-action=ignore \
 && pip install --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --root-user-action=ignore \
      httpx \
      sentence-transformers \
      transformers \
      huggingface-hub \
      mcp

COPY search_mcp.py ${APP_HOME}/search_mcp.py

RUN python - <<'PY'
import os
from sentence_transformers import CrossEncoder

model = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
cache = os.environ.get("HF_HOME")

print(f"Preloading reranker: {model}")
print(f"HF_HOME: {cache}")

CrossEncoder(model, device="cpu", max_length=512)

print("Reranker baked into image.")
PY

RUN chown -R 10001:10001 ${APP_HOME} ${HF_HOME} \
 && chmod -R u=rwX,g=rX,o= ${APP_HOME} ${HF_HOME}

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/" || exit 1

CMD ["python", "/app/search_mcp.py", "mcp"]