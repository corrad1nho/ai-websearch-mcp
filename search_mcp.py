#!/usr/bin/env python3
"""
CPU-friendly SearXNG + Crawl4AI + MiniLM reranker MCP server.

Tools:
  - web_search_snippets:      SearXNG only, fastest, no crawl
  - quick_search:             crawl top 3 pages, small rerank budget
  - comprehensive_search:     crawl top 10 pages, medium rerank budget
  - deep_research:            deep crawl interesting seeds up to depth 3, guarded budget

Main design choices:
  - One CPU-friendly reranker only: cross-encoder/ms-marco-MiniLM-L-6-v2
  - quick_search is intentionally aggressive about speed
  - Crawl4AI cache is used by default except for very fresh searches
  - crawled chunks are quality-filtered before reranking and weak scores are dropped

CLI examples:
  python search_mcp snippets "qwen3 release" --freshness week --category news
  python search_mcp quick "kubernetes sidecar containers" --category documentation
  python search_mcp comprehensive "crawl4ai deep crawling" --category documentation
  python search_mcp deep "kubernetes challenges" --category documentation --depth 3
  python search_mcp bench "Ai news" --runs 3
  python search_mcp mcp

Recommended CPU-only env for Kubernetes:
  RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
  RERANK_DEVICE=cpu
  TORCH_THREADS=6
  TOKENIZERS_PARALLELISM=false
  PRELOAD_RERANKER=true

Crawl4AI note:
  This script targets the common /crawl API shape used by recent Crawl4AI
  releases. Deep-crawl schema support has changed across versions, so
  fetch_deep_pages() gracefully falls back to shallow seed crawling if the
  server rejects the deep-crawl payload.
"""

# ---- Env caps MUST be set before torch / sentence-transformers import ----
import os
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_THREADS", "6"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_THREADS", "6"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
import re
import time
import asyncio
import argparse
import warnings
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ---------------- Config ----------------
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:11235").rstrip("/")

# One CPU-friendly model. Keep the image/cache simple.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "6"))
PRELOAD_RERANKER = os.getenv("PRELOAD_RERANKER", "true").lower() in ("1", "true", "yes")

VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("1", "true", "yes")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
PORT = int(os.getenv("PORT", "8000"))

# Crawl cache: use cache for most searches, bypass only for very fresh/news-like calls.
CRAWL_CACHE_MODE = os.getenv("CRAWL_CACHE_MODE", "enabled")

# Search/crawl budgets. Tune these, not the model.
SNIPPETS_RESULTS = int(os.getenv("SNIPPETS_RESULTS", "8"))
QUICK_SEARCH_RESULTS = int(os.getenv("QUICK_SEARCH_RESULTS", "8"))
QUICK_CRAWL_TOP_N = int(os.getenv("QUICK_CRAWL_TOP_N", "3"))
QUICK_TOP_CHUNKS = int(os.getenv("QUICK_TOP_CHUNKS", "4"))
QUICK_MAX_RERANK_DOCS = int(os.getenv("QUICK_MAX_RERANK_DOCS", "18"))
QUICK_MAX_CHUNKS_PER_PAGE = int(os.getenv("QUICK_MAX_CHUNKS_PER_PAGE", "5"))
QUICK_PAGE_TIMEOUT_MS = int(os.getenv("QUICK_PAGE_TIMEOUT_MS", "6000"))

COMPREHENSIVE_SEARCH_RESULTS = int(os.getenv("COMPREHENSIVE_SEARCH_RESULTS", "16"))
COMPREHENSIVE_CRAWL_TOP_N = int(os.getenv("COMPREHENSIVE_CRAWL_TOP_N", "10"))
COMPREHENSIVE_TOP_CHUNKS = int(os.getenv("COMPREHENSIVE_TOP_CHUNKS", "10"))
COMPREHENSIVE_MAX_RERANK_DOCS = int(os.getenv("COMPREHENSIVE_MAX_RERANK_DOCS", "60"))
COMPREHENSIVE_MAX_CHUNKS_PER_PAGE = int(os.getenv("COMPREHENSIVE_MAX_CHUNKS_PER_PAGE", "8"))
COMPREHENSIVE_PAGE_TIMEOUT_MS = int(os.getenv("COMPREHENSIVE_PAGE_TIMEOUT_MS", "10000"))

DEEP_SEARCH_RESULTS = int(os.getenv("DEEP_SEARCH_RESULTS", "12"))
DEEP_SEED_TOP_N = int(os.getenv("DEEP_SEED_TOP_N", "3"))
DEEP_MAX_DEPTH = int(os.getenv("DEEP_MAX_DEPTH", "3"))
DEEP_MAX_PAGES_TOTAL = int(os.getenv("DEEP_MAX_PAGES_TOTAL", "30"))
DEEP_TOP_CHUNKS = int(os.getenv("DEEP_TOP_CHUNKS", "16"))
DEEP_MAX_RERANK_DOCS = int(os.getenv("DEEP_MAX_RERANK_DOCS", "120"))
DEEP_MAX_CHUNKS_PER_PAGE = int(os.getenv("DEEP_MAX_CHUNKS_PER_PAGE", "6"))
DEEP_PAGE_TIMEOUT_MS = int(os.getenv("DEEP_PAGE_TIMEOUT_MS", "12000"))

VALID_FRESHNESS = {"", "day", "week", "month", "year"}
VALID_CATEGORIES = {
    "general",
    "news",
    "academic",
    "science",
    "it",
    "documentation",
    "github",
    "forums",
    "reddit",
    "stackoverflow",
}

# Quality gates. CrossEncoder scores may be negative; this is a pragmatic
# threshold to avoid returning complete garbage when the crawler extracts poor text.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-6.0"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "250"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "35"))
MAX_TABLE_PIPE_RATIO = float(os.getenv("MAX_TABLE_PIPE_RATIO", "0.08"))
MAX_SHORT_OR_SYMBOLIC_LINE_RATIO = float(os.getenv("MAX_SHORT_OR_SYMBOLIC_LINE_RATIO", "0.45"))
MAX_LINK_DENSITY = float(os.getenv("MAX_LINK_DENSITY", "0.35"))
MAX_REPEAT_LINE_RATIO = float(os.getenv("MAX_REPEAT_LINE_RATIO", "0.35"))

# Conservative SearXNG mapping. Engine names only work if enabled in your SearXNG.
# If an engine/category is missing, SearXNG normally returns fewer/no results rather
# than crashing. Still, default to broad categories where possible.
CATEGORY_MAP: dict[str, dict[str, str]] = {
    "general": {},
    "news": {"categories": "news"},
    "academic": {"categories": "science"},
    "science": {"categories": "science"},
    "it": {"categories": "it"},
    "documentation": {"categories": "it"},
    "github": {"engines": "github"},
    "forums": {"engines": "reddit,stackoverflow,stackexchange"},
    "reddit": {"engines": "reddit"},
    "stackoverflow": {"engines": "stackoverflow,stackexchange"},
}

# Optional query hints for categories where SearXNG categories are too broad.
# These are intentionally mild, because source bias can hurt recall.
CATEGORY_QUERY_HINTS: dict[str, str] = {
    "documentation": "official documentation docs",
    "github": "GitHub repository issues pull requests",
    "forums": "forum discussion reddit stackoverflow",
    "reddit": "reddit discussion",
    "stackoverflow": "stackoverflow stackexchange",
}

# ---------------- Globals ----------------
_reranker = None
_http_client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class ModeConfig:
    search_results: int
    crawl_top_n: int
    top_chunks: int
    max_rerank_docs: int
    max_chunks_per_page: int
    page_timeout_ms: int
    chunk_size: int
    chunk_overlap: int


QUICK_CFG = ModeConfig(
    search_results=QUICK_SEARCH_RESULTS,
    crawl_top_n=QUICK_CRAWL_TOP_N,
    top_chunks=QUICK_TOP_CHUNKS,
    max_rerank_docs=QUICK_MAX_RERANK_DOCS,
    max_chunks_per_page=QUICK_MAX_CHUNKS_PER_PAGE,
    page_timeout_ms=QUICK_PAGE_TIMEOUT_MS,
    chunk_size=900,
    chunk_overlap=100,
)

COMPREHENSIVE_CFG = ModeConfig(
    search_results=COMPREHENSIVE_SEARCH_RESULTS,
    crawl_top_n=COMPREHENSIVE_CRAWL_TOP_N,
    top_chunks=COMPREHENSIVE_TOP_CHUNKS,
    max_rerank_docs=COMPREHENSIVE_MAX_RERANK_DOCS,
    max_chunks_per_page=COMPREHENSIVE_MAX_CHUNKS_PER_PAGE,
    page_timeout_ms=COMPREHENSIVE_PAGE_TIMEOUT_MS,
    chunk_size=1100,
    chunk_overlap=150,
)

DEEP_CFG = ModeConfig(
    search_results=DEEP_SEARCH_RESULTS,
    crawl_top_n=DEEP_SEED_TOP_N,
    top_chunks=DEEP_TOP_CHUNKS,
    max_rerank_docs=DEEP_MAX_RERANK_DOCS,
    max_chunks_per_page=DEEP_MAX_CHUNKS_PER_PAGE,
    page_timeout_ms=DEEP_PAGE_TIMEOUT_MS,
    chunk_size=1200,
    chunk_overlap=160,
)

# ---------------- Helpers ----------------
def _valid_freshness(value: str) -> str:
    return value if value in VALID_FRESHNESS else ""


def _valid_category(value: str) -> str:
    value = (value or "general").strip().lower().replace("-", "_")
    return value if value in VALID_CATEGORIES else "general"


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _dedupe_results(results: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    out: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(r)
    return out


def _query_for_category(query: str, category: str) -> str:
    """Add a light query hint for categories that benefit from it."""
    category = _valid_category(category)
    hint = CATEGORY_QUERY_HINTS.get(category, "")
    if not hint:
        return query
    # Do not add hints when the user already clearly specified source targeting.
    q_lower = query.lower()
    if any(marker in q_lower for marker in ("site:", "github", "reddit", "stackoverflow", "official docs", "documentation")):
        return query
    return f"{query} {hint}"


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            verify=VERIFY_SSL,
            limits=httpx.Limits(max_connections=30, max_keepalive_connections=15),
            follow_redirects=True,
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

# ---------------- Reranker ----------------
def get_reranker():
    """Lazy-load one CPU-friendly cross-encoder; cached per process."""
    global _reranker
    if _reranker is None:
        import torch
        if RERANK_DEVICE == "cpu":
            torch.set_num_threads(TORCH_THREADS)
        from sentence_transformers import CrossEncoder
        print(
            f"[init] loading reranker '{RERANK_MODEL}' on {RERANK_DEVICE} "
            f"with torch_threads={TORCH_THREADS}...",
            file=sys.stderr,
            flush=True,
        )
        t0 = time.time()
        _reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=384)
        print(f"[init] reranker ready in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    return _reranker


def rerank(query: str, docs: list[str], top_k: int, batch_size: int = 16) -> list[tuple[int, float]]:
    if not docs:
        return []
    scores = get_reranker().predict(
        [(query, d) for d in docs],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    ranked = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
    return [(i, float(s)) for i, s in ranked[:top_k]]


def rerank_results_by_snippet(query: str, results: list[dict], top_n: int) -> list[dict]:
    """First-stage ranking: title + snippet only, before expensive crawling."""
    if not results:
        return []
    docs = [
        _normalize_space(f"{r.get('title', '')}\n{r.get('snippet', '')}")[:1000]
        for r in results
    ]
    ranked = rerank(query, docs, min(top_n, len(docs)), batch_size=16)
    return [results[i] for i, _ in ranked]

# ---------------- SearXNG ----------------
async def searxng_search(query: str, num: int, freshness: str = "", category: str = "general") -> list[dict]:
    freshness = _valid_freshness(freshness)
    category = _valid_category(category)
    effective_query = _query_for_category(query, category)

    params: dict[str, str | int] = {
        "q": effective_query,
        "format": "json",
        "safesearch": 0,
    }
    if freshness:
        params["time_range"] = freshness
    params.update(CATEGORY_MAP.get(category, {}))

    client = await get_http_client()
    r = await client.get(f"{SEARXNG_URL}/search", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = []
    for x in data.get("results", [])[:num]:
        url = x.get("url") or ""
        if not url:
            continue
        results.append({
            "title": x.get("title") or url,
            "url": url,
            "snippet": x.get("content") or x.get("snippet") or "",
        })
    return _dedupe_results(results)

# ---------------- Crawl4AI payloads ----------------
def _cache_mode_for(freshness: str, category: str) -> str:
    # Very fresh and news/forum searches are more likely to need live pages.
    if freshness == "day" or category in {"news", "forums", "reddit", "stackoverflow"}:
        return "bypass"
    return CRAWL_CACHE_MODE


def _base_browser_config() -> dict:
    return {
        "type": "BrowserConfig",
        "params": {
            "headless": True,
            "text_mode": True,
            "light_mode": True,
            "ignore_https_errors": not VERIFY_SSL,
        },
    }


def _base_crawler_params(page_timeout_ms: int, freshness: str, category: str) -> dict:
    category = _valid_category(category)

    # Quick/default crawling: favor fast, readable content over perfect rendering.
    params = {
        "cache_mode": _cache_mode_for(freshness, category),
        "page_timeout": page_timeout_ms,
        "wait_until": "domcontentloaded",
        "scan_full_page": False,
        "process_iframes": False,
        "remove_overlay_elements": True,
        "excluded_tags": [
            "nav", "footer", "header", "aside", "form", "script", "style",
            "noscript", "svg", "canvas",
        ],
        "exclude_external_links": True,
        "exclude_social_media_links": True,
        "exclude_all_images": True,
        "only_text": False,
        "markdown_generator": {
            "type": "DefaultMarkdownGenerator",
            "params": {
                "content_filter": {
                    "type": "PruningContentFilter",
                    "params": {
                        "threshold": 0.50,
                        "threshold_type": "fixed",
                        "min_word_threshold": 20,
                    },
                },
                "options": {
                    "ignore_links": False,
                    "escape_html": False,
                    "body_width": 0,
                },
            },
        },
    }

    # Documentation pages often have useful side/main links, but still avoid external links.
    if category == "documentation":
        params["excluded_tags"] = [
            "footer", "header", "form", "script", "style", "noscript", "svg", "canvas"
        ]

    return params


def _crawl_payload(urls: list[str], page_timeout_ms: int, freshness: str, category: str) -> dict:
    return {
        "urls": urls,
        "browser_config": _base_browser_config(),
        "crawler_config": {
            "type": "CrawlerRunConfig",
            "params": _base_crawler_params(page_timeout_ms, freshness, category),
        },
    }


def _deep_crawl_payload(
    seed_url: str,
    query: str,
    max_depth: int,
    max_pages: int,
    page_timeout_ms: int,
    freshness: str,
    category: str,
) -> dict:
    """Crawl4AI deep crawling payload with a fallback in fetch_deep_pages()."""
    params = _base_crawler_params(page_timeout_ms, freshness, category)
    params.update({
        "deep_crawl_strategy": {
            "type": "BestFirstCrawlingStrategy",
            "params": {
                "max_depth": max_depth,
                "include_external": False,
                "max_pages": max_pages,
                "url_scorer": {
                    "type": "KeywordRelevanceScorer",
                    "params": {
                        "keywords": query.split()[:12],
                        "weight": 0.7,
                    },
                },
            },
        },
        "stream": False,
    })
    return {
        "urls": [seed_url],
        "browser_config": _base_browser_config(),
        "crawler_config": {
            "type": "CrawlerRunConfig",
            "params": params,
        },
    }

# ---------------- Crawl4AI parsing/fetching ----------------
def _extract_markdown(res: dict) -> str:
    md_field = res.get("markdown")
    if isinstance(md_field, dict):
        return (
            md_field.get("fit_markdown")
            or md_field.get("markdown_with_citations")
            or md_field.get("raw_markdown")
            or ""
        )
    return md_field or ""


def _extract_crawl_results(data: dict | list) -> dict[str, str]:
    """Accepts common Crawl4AI response shapes and returns url -> markdown."""
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results") or data.get("data") or []
        if not results and (data.get("url") or data.get("markdown")):
            results = [data]
    else:
        results = []

    out: dict[str, str] = {}
    for res in results:
        if not isinstance(res, dict):
            continue
        if res.get("success", True) is False:
            continue
        url = res.get("url") or res.get("final_url") or ""
        md = _extract_markdown(res).strip()
        if url and md:
            out[url] = md
    return out


async def fetch_pages(urls: list[str], page_timeout_ms: int, freshness: str, category: str) -> dict[str, str]:
    if not urls:
        return {}
    payload = _crawl_payload(urls, page_timeout_ms, freshness, category)
    client = await get_http_client()
    r = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload, timeout=120)
    r.raise_for_status()
    return _extract_crawl_results(r.json())


async def fetch_deep_pages(
    seed_urls: list[str],
    query: str,
    max_depth: int,
    max_pages_total: int,
    page_timeout_ms: int,
    freshness: str,
    category: str,
) -> dict[str, str]:
    if not seed_urls:
        return {}

    per_seed_budget = max(3, max_pages_total // max(1, len(seed_urls)))
    client = await get_http_client()

    async def _one(seed: str) -> dict[str, str]:
        payload = _deep_crawl_payload(
            seed_url=seed,
            query=query,
            max_depth=max_depth,
            max_pages=per_seed_budget,
            page_timeout_ms=page_timeout_ms,
            freshness=freshness,
            category=category,
        )
        try:
            r = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload, timeout=240)
            r.raise_for_status()
            pages = _extract_crawl_results(r.json())
            if pages:
                return pages
        except Exception as exc:
            print(f"[warn] deep crawl failed for {seed}: {exc}; falling back to shallow", file=sys.stderr)
        return await fetch_pages([seed], page_timeout_ms, freshness, category)

    batches = await asyncio.gather(*[_one(seed) for seed in seed_urls], return_exceptions=True)
    out: dict[str, str] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            print(f"[warn] deep crawl batch failed: {batch}", file=sys.stderr)
            continue
        for url, md in batch.items():
            if len(out) >= max_pages_total:
                break
            out[url] = md
    return out

# ---------------- Chunking/ranking/output ----------------
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += max(1, size - overlap)
    return chunks


def is_good_chunk(text: str) -> bool:
    """Reject crawler/table/navigation junk before expensive reranking."""
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return False

    words = re.findall(r"\w+", text)
    if len(words) < MIN_CHUNK_WORDS:
        return False

    # Markdown tables from index/reference pages often contain lots of pipes and
    # symbolic separator lines. They rerank badly but can still leak into top-k
    # if the candidate set is weak.
    pipe_ratio = text.count("|") / max(1, len(text))
    if pipe_ratio > MAX_TABLE_PIPE_RATIO:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    short_or_symbolic = sum(
        1 for line in lines
        if len(line) < 20 or re.fullmatch(r"[\|\-\s:_•·]+", line)
    )
    if short_or_symbolic / max(1, len(lines)) > MAX_SHORT_OR_SYMBOLIC_LINE_RATIO:
        return False

    # Link farms / nav pages: many markdown links and little prose.
    link_count = len(re.findall(r"\[[^\]]+\]\([^\)]+\)", text))
    if link_count / max(1, len(lines)) > MAX_LINK_DENSITY:
        return False

    normalized_lines = [_normalize_space(line).lower() for line in lines]
    if normalized_lines:
        repeated = len(normalized_lines) - len(set(normalized_lines))
        if repeated / max(1, len(normalized_lines)) > MAX_REPEAT_LINE_RATIO:
            return False

    # Mostly punctuation/symbol text is never useful to an LLM.
    alpha_num = sum(ch.isalnum() for ch in text)
    if alpha_num / max(1, len(text)) < 0.45:
        return False

    return True


def trim_chunk(text: str, max_chars: int | None = None) -> str:
    """Clean excessive whitespace while preserving markdown readability."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    text = re.sub(r"[ \t]+", " ", text)
    if max_chars and len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].strip()
        return cut + " …"
    return text


def build_chunk_corpus(pages: dict[str, str], cfg: ModeConfig) -> tuple[list[str], list[str]]:
    all_chunks: list[str] = []
    meta: list[str] = []
    # Prefer diversity: scan each page, keep only decent chunks, and cap per page.
    for url, md in pages.items():
        kept_for_page = 0
        chunks = chunk_text(md, size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        for ch in chunks:
            ch = trim_chunk(ch)
            if not is_good_chunk(ch):
                continue
            all_chunks.append(ch)
            meta.append(url)
            kept_for_page += 1
            if len(all_chunks) >= cfg.max_rerank_docs:
                return all_chunks, meta
            if kept_for_page >= cfg.max_chunks_per_page:
                break
    return all_chunks, meta


def format_snippets(results: list[dict], timing_line: str | None = None) -> str:
    if not results:
        return "No results found."
    body = "## Search results\n\n" + "\n\n".join(
        f"{i}. [{r['title']}]({r['url']})\n{r.get('snippet', '').strip()}"
        for i, r in enumerate(results, 1)
    )
    if timing_line:
        body += f"\n\n---\n{timing_line}"
    return body


def format_passages(
    ranked: list[tuple[int, float]],
    chunks: list[str],
    meta: list[str],
    url_to_title: dict[str, str],
    pages: dict[str, str],
    fallback_results: list[dict] | None = None,
    timing_line: str | None = None,
) -> str:
    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    for idx, score in ranked:
        if score < MIN_RERANK_SCORE:
            continue
        url = meta[idx]
        text = trim_chunk(chunks[idx], max_chars=1800)
        if not is_good_chunk(text):
            continue
        key = (url, text[:160])
        if key in seen:
            continue
        seen.add(key)
        title = url_to_title.get(url) or url
        blocks.append(f"[{title}]({url}) (rel {score:.2f})\n{text}")

    if not blocks:
        fallback = format_snippets(fallback_results or []) if fallback_results else "No useful search snippets available."
        body = (
            "No strong crawled passages survived the quality filter. "
            "Showing search snippets instead.\n\n" + fallback
        )
        if timing_line:
            body += f"\n\n---\n{timing_line}"
        return body

    sources = "\n".join(
        f"- [{url_to_title.get(u) or u}]({u})"
        for u in pages.keys()
    )
    body = "## Relevant passages\n\n" + "\n\n---\n\n".join(blocks) + f"\n\n## Sources\n{sources}"
    if timing_line:
        body += f"\n\n---\n{timing_line}"
    return body


async def crawl_search_rerank(
    query: str,
    cfg: ModeConfig,
    freshness: str = "",
    category: str = "general",
    timing: bool = False,
) -> str:
    freshness = _valid_freshness(freshness)
    category = _valid_category(category)

    t0 = time.time()
    results = await searxng_search(query, num=cfg.search_results, freshness=freshness, category=category)
    if not results:
        return "No results found."
    t1 = time.time()

    ranked_results = await asyncio.to_thread(rerank_results_by_snippet, query, results, cfg.crawl_top_n)
    t1b = time.time()

    urls = [r["url"] for r in ranked_results]
    url_to_title = {r["url"]: r["title"] for r in results}
    pages = await fetch_pages(urls, cfg.page_timeout_ms, freshness, category)
    t2 = time.time()

    if not pages:
        body = format_snippets(ranked_results)
        if timing:
            body += (
                f"\n\n---\n[timing] search={t1-t0:.1f}s snippet_rerank={t1b-t1:.1f}s "
                f"crawl={t2-t1b:.1f}s total={t2-t0:.1f}s | crawl returned no content"
            )
        return f"(crawl returned no content — showing search snippets)\n\n{body}"

    chunks, meta = build_chunk_corpus(pages, cfg)
    if not chunks:
        t3 = time.time()
        body = "No useful crawled chunks survived the quality filter. Showing search snippets instead.\n\n" + format_snippets(ranked_results)
        if timing:
            body += (
                f"\n\n---\n[timing] search={t1-t0:.1f}s snippet_rerank={t1b-t1:.1f}s "
                f"crawl={t2-t1b:.1f}s quality_filter={t3-t2:.1f}s total={t3-t0:.1f}s | "
                f"category={category} freshness={freshness or 'any'} pages={len(pages)} chunks=0"
            )
        return body

    ranked = await asyncio.to_thread(rerank, query, chunks, cfg.top_chunks)
    t3 = time.time()

    timing_line = None
    if timing:
        timing_line = (
            f"[timing] search={t1-t0:.1f}s snippet_rerank={t1b-t1:.1f}s "
            f"crawl={t2-t1b:.1f}s chunk_rerank={t3-t2:.1f}s total={t3-t0:.1f}s | "
            f"category={category} freshness={freshness or 'any'} pages={len(pages)} chunks={len(chunks)}"
        )
    return format_passages(ranked, chunks, meta, url_to_title, pages, fallback_results=ranked_results, timing_line=timing_line)

# ---------------- Public tool functions ----------------
async def do_snippets(query: str, freshness: str = "", category: str = "general", timing: bool = False) -> str:
    freshness = _valid_freshness(freshness)
    category = _valid_category(category)
    t0 = time.time()
    results = await searxng_search(query, num=SNIPPETS_RESULTS, freshness=freshness, category=category)
    t1 = time.time()
    line = (
        f"[timing] search={t1-t0:.1f}s total={t1-t0:.1f}s | "
        f"category={category} freshness={freshness or 'any'} results={len(results)}"
    ) if timing else None
    return format_snippets(results, line)


async def do_quick_search(query: str, freshness: str = "", category: str = "general", timing: bool = False) -> str:
    return await crawl_search_rerank(query, QUICK_CFG, freshness=freshness, category=category, timing=timing)


async def do_comprehensive_search(query: str, freshness: str = "", category: str = "general", timing: bool = False) -> str:
    return await crawl_search_rerank(query, COMPREHENSIVE_CFG, freshness=freshness, category=category, timing=timing)


async def do_deep_research(
    query: str,
    max_depth: int = DEEP_MAX_DEPTH,
    max_pages: int = DEEP_MAX_PAGES_TOTAL,
    freshness: str = "",
    category: str = "general",
    timing: bool = False,
) -> str:
    freshness = _valid_freshness(freshness)
    category = _valid_category(category)
    max_depth = max(1, min(max_depth, 3))
    max_pages = max(5, min(max_pages, 50))

    t0 = time.time()
    results = await searxng_search(query, num=DEEP_SEARCH_RESULTS, freshness=freshness, category=category)
    if not results:
        return "No results found."
    t1 = time.time()

    seeds = await asyncio.to_thread(rerank_results_by_snippet, query, results, DEEP_SEED_TOP_N)
    t1b = time.time()
    seed_urls = [r["url"] for r in seeds]

    pages = await fetch_deep_pages(
        seed_urls=seed_urls,
        query=query,
        max_depth=max_depth,
        max_pages_total=max_pages,
        page_timeout_ms=DEEP_PAGE_TIMEOUT_MS,
        freshness=freshness,
        category=category,
    )
    t2 = time.time()

    if not pages:
        body = format_snippets(seeds)
        if timing:
            body += (
                f"\n\n---\n[timing] search={t1-t0:.1f}s seed_rerank={t1b-t1:.1f}s "
                f"deep_crawl={t2-t1b:.1f}s total={t2-t0:.1f}s | deep crawl returned no content"
            )
        return f"(deep crawl returned no content — showing seed snippets)\n\n{body}"

    url_to_title = {r["url"]: r["title"] for r in results}
    chunks, meta = build_chunk_corpus(pages, DEEP_CFG)
    if not chunks:
        t3 = time.time()
        body = "No useful deep-crawled chunks survived the quality filter. Showing seed snippets instead.\n\n" + format_snippets(seeds)
        if timing:
            body += (
                f"\n\n---\n[timing] search={t1-t0:.1f}s seed_rerank={t1b-t1:.1f}s "
                f"deep_crawl={t2-t1b:.1f}s quality_filter={t3-t2:.1f}s total={t3-t0:.1f}s | "
                f"category={category} freshness={freshness or 'any'} seeds={len(seed_urls)} "
                f"pages={len(pages)} chunks=0 depth={max_depth} max_pages={max_pages}"
            )
        return body

    ranked = await asyncio.to_thread(rerank, query, chunks, DEEP_TOP_CHUNKS)
    t3 = time.time()

    timing_line = None
    if timing:
        timing_line = (
            f"[timing] search={t1-t0:.1f}s seed_rerank={t1b-t1:.1f}s "
            f"deep_crawl={t2-t1b:.1f}s chunk_rerank={t3-t2:.1f}s total={t3-t0:.1f}s | "
            f"category={category} freshness={freshness or 'any'} seeds={len(seed_urls)} "
            f"pages={len(pages)} chunks={len(chunks)} depth={max_depth} max_pages={max_pages}"
        )
    return format_passages(ranked, chunks, meta, url_to_title, pages, fallback_results=seeds, timing_line=timing_line)

# ---------------- MCP server ----------------
def build_mcp():
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"],
    )
    mcp = FastMCP("web-search", transport_security=security)

    if PRELOAD_RERANKER:
        print("[startup] warming reranker...", file=sys.stderr, flush=True)
        get_reranker()
        print("[startup] reranker warm — server ready.", file=sys.stderr, flush=True)
    else:
        print("[startup] reranker lazy-load enabled.", file=sys.stderr, flush=True)

    @mcp.tool()
    async def web_search_snippets(query: str, category: str = "general", freshness: str = "") -> str:
        """Fastest web lookup. Uses SearXNG snippets only; no browser crawling.

        Use first for broad discovery, simple current facts, or when snippets
        are likely enough.

        IMPORTANT FOR THE CALLING LLM:
        - Always cite the source URLs when using information from this tool.
        - Do not present claims from this tool without attribution.
        - Prefer citing the specific source that supports each statement.

        Args:
            query: Specific web search query.
            category: One of general, news, academic, science, it,
                documentation, github, forums, reddit, stackoverflow.
            freshness: Optional recency filter: '', 'day', 'week', 'month', 'year'.
        """
        return await do_snippets(query, freshness=freshness, category=category)

    @mcp.tool()
    async def quick_search(query: str, category: str = "general", freshness: str = "") -> str:
        """Quick web search with shallow crawling of the top 3 results.

        Best default for readily available information. Searches wider, reranks
        snippets, crawls only the top candidates, and returns a few relevant
        passages. Optimized for speed.

        IMPORTANT FOR THE CALLING LLM:
        - Always cite the source URLs when using information from this tool.
        - Do not present claims from this tool without attribution.
        - Prefer citing the specific source that supports each statement.

        Args:
            query: Specific search query.
            category: One of general, news, academic, science, it,
                documentation, github, forums, reddit, stackoverflow.
            freshness: Optional recency filter: '', 'day', 'week', 'month', 'year'.
        """
        return await do_quick_search(query, freshness=freshness, category=category)

    @mcp.tool()
    async def comprehensive_search(query: str, category: str = "general", freshness: str = "") -> str:
        """Comprehensive multi-source search with shallow crawling of top 10 results.

        Use when quick_search is too thin or when comparing multiple sources matters.
        More complete than quick_search, but slower.

        IMPORTANT FOR THE CALLING LLM:
        - Always cite the source URLs when using information from this tool.
        - Do not present claims from this tool without attribution.
        - Prefer citing the specific source that supports each statement.

        Args:
            query: Specific search query.
            category: One of general, news, academic, science, it,
                documentation, github, forums, reddit, stackoverflow.
            freshness: Optional recency filter: '', 'day', 'week', 'month', 'year'.
        """
        return await do_comprehensive_search(query, freshness=freshness, category=category)

    @mcp.tool()
    async def deep_research(
        query: str,
        category: str = "general",
        freshness: str = "",
        max_depth: int = DEEP_MAX_DEPTH,
        max_pages: int = DEEP_MAX_PAGES_TOTAL,
    ) -> str:
        """Deep research using Crawl4AI deep crawling from relevant seed URLs.

        Use for complex topics where important information may live below the
        initial search result pages: documentation, release notes, multi-page
        guides, APIs, and technical investigations.

        Guardrails: max_depth is capped at 3 and max_pages at 50.

        IMPORTANT FOR THE CALLING LLM:
        - Always cite the source URLs when using information from this tool.
        - Do not present claims from this tool without attribution.
        - Prefer citing the specific source that supports each statement.

        Args:
            query: Research question or topic.
            category: One of general, news, academic, science, it,
                documentation, github, forums, reddit, stackoverflow.
            freshness: Optional recency filter: '', 'day', 'week', 'month', 'year'.
            max_depth: Crawl depth, 1-3. Default 3.
            max_pages: Total page budget, 5-50. Default from env.
        """
        return await do_deep_research(
            query,
            max_depth=max_depth,
            max_pages=max_pages,
            freshness=freshness,
            category=category,
        )

    return mcp


def run_mcp():
    mcp = build_mcp()
    transport = MCP_TRANSPORT
    if transport in ("streamable-http", "sse"):
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = PORT
        print(f"[startup] MCP serving via {transport} on 0.0.0.0:{PORT}", file=sys.stderr, flush=True)
    else:
        print(f"[startup] MCP serving via {transport}", file=sys.stderr, flush=True)
    mcp.run(transport=transport)

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="CPU-friendly Search MCP pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("query", nargs="+")
        p.add_argument("--freshness", "--time", dest="freshness", default="", choices=sorted(VALID_FRESHNESS))
        p.add_argument("--category", default="general", choices=sorted(VALID_CATEGORIES))

    p_snip = sub.add_parser("snippets", help="SearXNG snippets only")
    add_common(p_snip)

    p_quick = sub.add_parser("quick", help="quick crawl top 3")
    add_common(p_quick)

    p_comp = sub.add_parser("comprehensive", help="crawl top 10")
    add_common(p_comp)

    p_deep = sub.add_parser("deep", help="deep crawl relevant seeds")
    add_common(p_deep)
    p_deep.add_argument("--depth", type=int, default=DEEP_MAX_DEPTH)
    p_deep.add_argument("--max-pages", type=int, default=DEEP_MAX_PAGES_TOTAL)

    p_bench = sub.add_parser("bench", help="run several quick searches in one warm process")
    p_bench.add_argument("query", nargs="+")
    p_bench.add_argument("--runs", type=int, default=3)
    p_bench.add_argument("--freshness", "--time", dest="freshness", default="", choices=sorted(VALID_FRESHNESS))
    p_bench.add_argument("--category", default="general", choices=sorted(VALID_CATEGORIES))

    sub.add_parser("mcp", help="run as MCP server")

    args = parser.parse_args()

    if args.mode == "mcp":
        run_mcp()
        return

    q = " ".join(getattr(args, "query", []))

    try:
        if args.mode == "snippets":
            print(asyncio.run(do_snippets(q, freshness=args.freshness, category=args.category, timing=True)))
        elif args.mode == "quick":
            print(asyncio.run(do_quick_search(q, freshness=args.freshness, category=args.category, timing=True)))
        elif args.mode == "comprehensive":
            print(asyncio.run(do_comprehensive_search(q, freshness=args.freshness, category=args.category, timing=True)))
        elif args.mode == "deep":
            print(asyncio.run(do_deep_research(
                q,
                max_depth=args.depth,
                max_pages=args.max_pages,
                freshness=args.freshness,
                category=args.category,
                timing=True,
            )))
        elif args.mode == "bench":
            async def _bench():
                get_reranker()
                for i in range(args.runs):
                    t = time.time()
                    await do_quick_search(q, freshness=args.freshness, category=args.category)
                    print(f"run {i+1}: {time.time()-t:.1f}s", file=sys.stderr, flush=True)
            asyncio.run(_bench())
    finally:
        try:
            asyncio.run(close_http_client())
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
