#!/usr/bin/env python3
"""
SearXNG + Crawl4AI + reranker search pipeline for AI agents (MCP server).

Two content-returning tools:
  - quick_answer:  shallow crawl of top results, lean output, fast-ish
  - deep_research: thorough multi-source crawl, more passages

CLI test modes:
  python search_mcp.py quick    "latest qwen3 release" --time day
  python search_mcp.py quick    "important tech news"   --time day --news
  python search_mcp.py research "how does qwen3 attention work" --sources 5 --chunks 8
  python search_mcp.py search   "qwen3" --time week          # raw snippets (debug)
  python search_mcp.py bench    "qwen3 news"                 # warm-process timing

MCP server modes:
  python search_mcp.py mcp                 # transport from MCP_TRANSPORT env (default stdio)
  MCP_TRANSPORT=streamable-http python search_mcp.py mcp     # HTTP for in-cluster

Prereqs (port-forward your cluster services for local testing):
  kubectl -n searxng  port-forward svc/searxng  8080:8080
  kubectl -n crawl4ai port-forward svc/crawl4ai 11235:11235

Env:
  SEARXNG_URL    default http://localhost:8080
  CRAWL4AI_URL   default http://localhost:11235
  RERANK_MODEL   default BAAI/bge-reranker-base
  RERANK_DEVICE  default auto    (auto|cpu|cuda)
  TORCH_THREADS  default 6       (CPU thread cap)
  VERIFY_SSL     default false   (skip TLS verify for internal/self-signed)
  MCP_TRANSPORT  default stdio   (stdio|streamable-http|sse)
  PORT           default 8000    (HTTP transport port)
"""

# ---- Env caps MUST be set before torch / sentence-transformers import ----
import os
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_THREADS", "6"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_THREADS", "6"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
import time
import asyncio
import argparse
import warnings

import httpx

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ---------------- Config ----------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:11235")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "6"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("1", "true", "yes")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
PORT = int(os.getenv("PORT", "8000"))

VALID_TIME_RANGES = {"", "day", "week", "month", "year"}

# ---------------- Reranker ----------------
_reranker = None


def _resolve_device() -> str:
    if RERANK_DEVICE != "auto":
        return RERANK_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_reranker():
    """Lazy-load the cross-encoder once per process; cached thereafter."""
    global _reranker
    if _reranker is None:
        import torch
        device = _resolve_device()
        if device == "cpu":
            torch.set_num_threads(TORCH_THREADS)
        from sentence_transformers import CrossEncoder
        print(f"[init] loading reranker '{RERANK_MODEL}' on {device}...",
              file=sys.stderr, flush=True)
        t0 = time.time()
        _reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
        print(f"[init] reranker ready in {time.time()-t0:.1f}s",
              file=sys.stderr, flush=True)
    return _reranker


# ---------------- SearXNG ----------------
async def searxng_search(
    query: str,
    num: int = 8,
    time_range: str = "",
    news: bool = False,
) -> list[dict]:
    """Query SearXNG JSON API.

    time_range: '' | 'day' | 'week' | 'month' | 'year'
    news: if True, restrict to the 'news' category (recency-focused engines)
    """
    if time_range not in VALID_TIME_RANGES:
        time_range = ""

    params = {"q": query, "format": "json", "safesearch": 0}
    if time_range:
        params["time_range"] = time_range
    if news:
        params["categories"] = "news"

    async with httpx.AsyncClient(timeout=20, verify=VERIFY_SSL) as client:
        r = await client.get(f"{SEARXNG_URL}/search", params=params)
        r.raise_for_status()
        data = r.json()

    results = data.get("results", [])[:num]
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""),
         "snippet": x.get("content", "")}
        for x in results if x.get("url")
    ]


# ---------------- Crawl4AI (0.8.x /crawl schema) ----------------
def _crawl_payload(urls: list[str], page_timeout: int) -> dict:
    return {
        "urls": urls,
        "browser_config": {
            "type": "BrowserConfig",
            "params": {
                "headless": True,
                "text_mode": True,
                "light_mode": True,
            },
        },
        "crawler_config": {
            "type": "CrawlerRunConfig",
            "params": {
                "cache_mode": "bypass",
                "page_timeout": page_timeout,
                "wait_until": "domcontentloaded",
                "excluded_tags": ["nav", "footer", "header", "aside",
                                  "form", "script", "style"],
                "exclude_external_links": True,
                "exclude_all_images": True,
                "markdown_generator": {
                    "type": "DefaultMarkdownGenerator",
                    "params": {
                        "content_filter": {
                            "type": "PruningContentFilter",
                            "params": {"threshold": 0.48,
                                       "threshold_type": "fixed"},
                        }
                    },
                },
            },
        },
    }


async def fetch_pages(urls: list[str], page_timeout: int) -> dict[str, str]:
    if not urls:
        return {}
    payload = _crawl_payload(urls, page_timeout)
    async with httpx.AsyncClient(timeout=120, verify=VERIFY_SSL) as client:
        r = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload)
        r.raise_for_status()
        data = r.json()

    out: dict[str, str] = {}
    for res in data.get("results", []):
        if not res.get("success", False):
            continue
        url = res.get("url", "")
        md_field = res.get("markdown")
        if isinstance(md_field, dict):
            md = (md_field.get("fit_markdown")
                  or md_field.get("raw_markdown") or "")
        else:
            md = md_field or ""
        if url and md and md.strip():
            out[url] = md
    return out


# ---------------- chunk + rerank ----------------
def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def rerank(query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
    if not docs:
        return []
    scores = get_reranker().predict(
        [(query, d) for d in docs],
        batch_size=32,
        show_progress_bar=False,
    )
    ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
    return [(i, float(s)) for i, s in ranked[:top_k]]


# ---------------- core pipeline ----------------
async def _crawl_search_rerank(
    query: str,
    num_sources: int,
    top_chunks: int,
    page_timeout: int,
    time_range: str,
    news: bool,
    chunk_size: int,
    timing: bool,
) -> str:
    t0 = time.time()
    results = await searxng_search(query, num=num_sources,
                                   time_range=time_range, news=news)
    if not results:
        return "No results found."
    t1 = time.time()

    url_to_title = {r["url"]: r["title"] for r in results}
    pages = await fetch_pages(list(url_to_title.keys()), page_timeout)
    t2 = time.time()

    if not pages:
        snip = "\n\n".join(
            f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results
        )
        return f"(crawl returned no content — showing search snippets)\n\n{snip}"

    all_chunks, meta = [], []
    for url, md in pages.items():
        for ch in chunk_text(md, size=chunk_size):
            all_chunks.append(ch)
            meta.append(url)

    ranked = await asyncio.to_thread(rerank, query, all_chunks, top_chunks)
    t3 = time.time()

    blocks, seen = [], set()
    for idx, score in ranked:
        url = meta[idx]
        key = (url, all_chunks[idx][:80])
        if key in seen:
            continue
        seen.add(key)
        title = url_to_title.get(url, url)
        blocks.append(
            f"[{title}]({url}) (rel {score:.2f})\n{all_chunks[idx].strip()}"
        )

    sources = "\n".join(f"- [{url_to_title[u]}]({u})" for u in pages)
    body = ("## Relevant passages\n\n"
            + "\n\n---\n\n".join(blocks)
            + f"\n\n## Sources\n{sources}")

    if timing:
        body += (f"\n\n---\n[timing] search={t1-t0:.1f}s crawl={t2-t1:.1f}s "
                 f"rerank={t3-t2:.1f}s total={t3-t0:.1f}s | "
                 f"pages={len(pages)} chunks={len(all_chunks)}")
    return body


# ---------------- public functions ----------------
async def do_quick(query: str, time_range: str = "", news: bool = False,
                   timing: bool = False) -> str:
    """Fast path: shallow crawl of top 3 results, return ~4 lean passages."""
    return await _crawl_search_rerank(
        query, num_sources=3, top_chunks=4, page_timeout=10000,
        time_range=time_range, news=news, chunk_size=1000, timing=timing,
    )


async def do_research(query: str, num_sources: int = 5, top_chunks: int = 8,
                      time_range: str = "", news: bool = False,
                      timing: bool = False) -> str:
    """Deep path: thorough multi-source crawl, return more passages."""
    return await _crawl_search_rerank(
        query, num_sources=num_sources, top_chunks=top_chunks,
        page_timeout=15000, time_range=time_range, news=news,
        chunk_size=1200, timing=timing,
    )


# ---------------- MCP server ----------------
def build_mcp():
    """Construct the FastMCP server, preload the model, register tools."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    # Trusted internal cluster: disable DNS-rebinding host validation so
    # requests via the service FQDN (not just localhost) are accepted.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"],
    )

    mcp = FastMCP("web-search", transport_security=security)

    # Preload the reranker at construction (first call fast; readiness gate).
    print("[startup] warming reranker...", file=sys.stderr, flush=True)
    get_reranker()
    print("[startup] reranker warm — server ready.", file=sys.stderr, flush=True)

    @mcp.tool()
    async def quick_answer(query: str, time_range: str = "",
                           news: bool = False) -> str:
        """Fast web lookup. Crawls the top results and returns the most
        relevant content passages. Use for current facts, quick questions,
        release notes, and recent news.

        Args:
            query: The search query. Be specific.
            time_range: Optional recency filter: '' (any), 'day', 'week',
                'month', 'year'. Use 'day' or 'week' for recent news.
            news: Set True to restrict to news sources (recency-focused).
        """
        try:
            if time_range not in VALID_TIME_RANGES:
                time_range = ""
            return await do_quick(query, time_range=time_range, news=news)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return f"ERROR in quick_answer: {type(e).__name__}: {e}"

    @mcp.tool()
    async def deep_research(query: str, num_sources: int = 5,
                            top_chunks: int = 8, time_range: str = "",
                            news: bool = False) -> str:
        """Thorough web research across multiple sources. Crawls several
        pages, extracts content, and returns reranked passages with
        citations. Use for complex questions needing comprehensive coverage.

        Args:
            query: The research question. Be specific.
            num_sources: How many sources to crawl (3-8).
            top_chunks: How many passages to return (4-12).
            time_range: Optional recency filter: '', 'day', 'week', 'month', 'year'.
            news: Set True to restrict to news sources.
        """
        try:
            if time_range not in VALID_TIME_RANGES:
                time_range = ""
            num_sources = max(1, min(num_sources, 8))
            top_chunks = max(1, min(top_chunks, 12))
            return await do_research(query, num_sources=num_sources,
                                     top_chunks=top_chunks, time_range=time_range,
                                     news=news)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            return f"ERROR in deep_research: {type(e).__name__}: {e}"

    return mcp


def run_mcp():
    mcp = build_mcp()
    transport = MCP_TRANSPORT
    if transport in ("streamable-http", "sse"):
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = PORT
        print(f"[startup] MCP serving via {transport} on 0.0.0.0:{PORT}",
              file=sys.stderr, flush=True)
    else:
        print(f"[startup] MCP serving via {transport}",
              file=sys.stderr, flush=True)
    mcp.run(transport=transport)


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Search MCP pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_quick = sub.add_parser("quick", help="fast shallow search")
    p_quick.add_argument("query", nargs="+")
    p_quick.add_argument("--time", default="", choices=sorted(VALID_TIME_RANGES))
    p_quick.add_argument("--news", action="store_true")

    p_res = sub.add_parser("research", help="deep multi-source research")
    p_res.add_argument("query", nargs="+")
    p_res.add_argument("--sources", type=int, default=5)
    p_res.add_argument("--chunks", type=int, default=8)
    p_res.add_argument("--time", default="", choices=sorted(VALID_TIME_RANGES))
    p_res.add_argument("--news", action="store_true")

    p_search = sub.add_parser("search", help="raw SearXNG snippets (debug)")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--time", default="", choices=sorted(VALID_TIME_RANGES))
    p_search.add_argument("--news", action="store_true")

    p_bench = sub.add_parser("bench", help="run several queries in one warm process")
    p_bench.add_argument("query", nargs="+")
    p_bench.add_argument("--runs", type=int, default=3)

    sub.add_parser("mcp", help="run as MCP server (transport from MCP_TRANSPORT)")

    args = parser.parse_args()

    if args.mode == "mcp":
        run_mcp()
        return

    q = " ".join(args.query)

    if args.mode == "quick":
        print(asyncio.run(do_quick(q, time_range=args.time,
                                   news=args.news, timing=True)))

    elif args.mode == "research":
        print(asyncio.run(do_research(q, num_sources=args.sources,
                                      top_chunks=args.chunks,
                                      time_range=args.time, news=args.news,
                                      timing=True)))

    elif args.mode == "search":
        results = asyncio.run(searxng_search(q, time_range=args.time,
                                             news=args.news))
        if not results:
            print("No results found.")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")

    elif args.mode == "bench":
        async def _bench():
            get_reranker()  # warm once, like the server does
            for i in range(args.runs):
                t = time.time()
                await do_quick(q)
                print(f"run {i+1}: {time.time()-t:.1f}s",
                      file=sys.stderr, flush=True)
        asyncio.run(_bench())


if __name__ == "__main__":
    main()