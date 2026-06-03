FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/YOUR_USERNAME/search-mcp"
LABEL org.opencontainers.image.description="SearXNG + Crawl4AI + reranker MCP server"
LABEL org.opencontainers.image.licenses="MIT"

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/app app

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "mcp[cli]>=1.2.0" \
    "httpx>=0.27.0" \
    "sentence-transformers>=3.0.0"

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface \
 && python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('BAAI/bge-reranker-base', device='cpu')" \
 && chown -R 10001:10001 /app

COPY --chown=10001:10001 search_mcp.py /app/search_mcp.py

ENV RERANK_DEVICE=cpu \
    TORCH_THREADS=6 \
    SEARXNG_URL=http://searxng.searxng.svc.cluster.local:8080 \
    CRAWL4AI_URL=http://crawl4ai.crawl4ai.svc.cluster.local:11235 \
    VERIFY_SSL=false \
    MCP_TRANSPORT=streamable-http \
    PORT=8000 \
    HF_HUB_DISABLE_TELEMETRY=1

USER 10001
EXPOSE 8000
CMD ["python", "search_mcp.py", "mcp"]