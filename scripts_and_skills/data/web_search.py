"""
web_search.py — DuckDuckGo search wrapper for Claude Code.

Local models can't browse the web. This script bridges that gap:
Claude Code calls this script, gets results back as text, and can
reason over them.

Usage (from CLI):
    python -m scripts_and_skills.data.web_search "your query" --top 5
    python -m scripts_and_skills.data.web_search "your query" --json

Usage (from Python):
    from scripts_and_skills.data.web_search import search
    results = search("attention mechanism transformer", top=5)
    for r in results:
        print(r["title"], r["url"])
        print(r["snippet"])
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    # Package was renamed from duckduckgo_search to ddgs
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False


def search(query: str,
           top: int = 5,
           region: str = "wt-wt",
           safesearch: str = "moderate") -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return structured results.

    Returns list of dicts with keys:
        title, url, snippet
    """
    if not HAS_DDG:
        return [{"error": "pip install ddgs", "title": "", "url": "", "snippet": ""}]

    try:
        with DDGS() as ddg:
            raw = list(ddg.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=top,
            ))
        results = []
        for r in raw:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", r.get("url", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            })
        return results
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return [{"error": str(e), "title": "", "url": "", "snippet": ""}]


def format_results_text(results: List[Dict[str, str]]) -> str:
    """Format results as readable text for Claude to process."""
    lines = []
    for i, r in enumerate(results, 1):
        if "error" in r:
            lines.append(f"[ERROR] {r['error']}")
            continue
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def fetch_url_text(url: str, timeout: int = 15, max_chars: int = 8000) -> str:
    """
    Fetch a web page and return its readable text content.
    Strips HTML tags and collapses whitespace.
    Returns empty string on failure.
    """
    try:
        import requests as _req
        import re
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        r = _req.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        html = r.text
        # Remove script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        logger.warning(f"Could not fetch {url}: {e}")
        return ""


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DuckDuckGo search CLI")
    parser.add_argument("query",          help="Search query")
    parser.add_argument("--top",   "-n",  type=int, default=5,  help="Number of results")
    parser.add_argument("--json",         action="store_true",  help="Output as JSON")
    parser.add_argument("--region",       default="wt-wt")
    args = parser.parse_args()

    results = search(args.query, top=args.top, region=args.region)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results_text(results))
