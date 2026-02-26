"""
arxiv_crawler.py — ArXiv paper search and retrieval for Claude Code data layer.

Uses the ArXiv Atom/XML API directly (no oarc-crawlers dependency).
Fetches paper metadata and abstracts as training data sources.

Usage (CLI):
    python -m scripts_and_skills.data.arxiv_crawler "large language models" --top 5
    python -m scripts_and_skills.data.arxiv_crawler "transformer attention" --top 3 --json

Usage (Python):
    from scripts_and_skills.data.arxiv_crawler import ArxivCrawler
    crawler = ArxivCrawler()
    papers = crawler.search_papers("quantum computing", max_results=5)
    text = crawler.fetch_paper_text(papers[0])
"""

import os
import sys
import json
import time
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS  = "http://www.w3.org/2005/Atom"

# ArXiv namespace map for ElementTree
NS = {
    "atom":    "http://www.w3.org/2005/Atom",
    "arxiv":   "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Keyword → arXiv category (from agent-chef)
CATEGORY_MAP = {
    "neural network":        "cs.LG",
    "machine learning":      "cs.LG",
    "deep learning":         "cs.LG",
    "artificial intelligence": "cs.AI",
    "computer vision":       "cs.CV",
    "natural language":      "cs.CL",
    "nlp":                   "cs.CL",
    "large language":        "cs.CL",
    "transformer":           "cs.CL",
    "robotics":              "cs.RO",
    "quantum":               "quant-ph",
    "physics":               "physics",
    "mathematics":           "math",
    "statistics":            "stat",
    "biology":               "q-bio",
    "economics":             "econ",
    "reinforcement learning": "cs.LG",
    "graph neural":          "cs.LG",
    "diffusion":             "cs.CV",
    "generative":            "cs.LG",
}


class ArxivCrawler:
    """
    Search ArXiv and fetch paper text via the public Atom API.
    No external dependencies beyond `requests` (already in project).
    """

    def __init__(self, rate_limit_delay: float = 3.0):
        try:
            import requests as _req
            self._requests = _req
        except ImportError:
            raise ImportError("pip install requests")
        self.rate_limit_delay = rate_limit_delay

    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search ArXiv and return structured paper metadata.

        Returns list of dicts:
            title, authors, abstract, categories,
            arxiv_url, pdf_link, published, arxiv_id
        """
        params = {
            "search_query": f"all:{query}",
            "max_results":  max_results,
            "sortBy":       "relevance",
        }
        logger.info(f"ArXiv search: '{query}' (max_results={max_results})")

        try:
            r = self._requests.get(ARXIV_API, params=params, timeout=30)
            r.raise_for_status()
        except Exception as e:
            logger.error(f"ArXiv API request failed: {e}")
            return []

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")
            return []

        papers = []
        for entry in root.findall("atom:entry", NS):
            paper = self._parse_entry(entry)
            if paper.get("title"):
                papers.append(paper)

        logger.info(f"ArXiv: {len(papers)} papers found for '{query}'")

        # Rate-limit: ArXiv TOS asks for 3s between bulk requests
        if papers:
            time.sleep(self.rate_limit_delay)

        return papers

    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        """Parse a single Atom <entry> element into a paper dict."""

        def text(tag: str) -> str:
            el = entry.find(f"atom:{tag}", NS)
            return (el.text or "").strip() if el is not None else ""

        title   = text("title").replace("\n", " ")
        abstract = text("summary").replace("\n", " ")
        published = text("published")[:10]  # YYYY-MM-DD

        # Authors
        authors = [
            (a.find("atom:name", NS).text or "").strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]

        # Links: abs page + PDF
        arxiv_url = ""
        pdf_link  = ""
        for link in entry.findall("atom:link", NS):
            rel   = link.get("rel", "")
            title_ = link.get("title", "")
            href  = link.get("href", "")
            if rel == "alternate" or title_ == "":
                if "abs" in href or rel == "alternate":
                    arxiv_url = href
            if title_ == "pdf" or link.get("type", "") == "application/pdf":
                pdf_link = href

        # arxiv_id from <id> tag (URL form)
        id_el = entry.find("atom:id", NS)
        full_id = (id_el.text or "").strip() if id_el is not None else ""
        arxiv_id = full_id.split("/abs/")[-1] if "/abs/" in full_id else full_id

        if not arxiv_url and full_id:
            arxiv_url = full_id  # <id> IS the abs URL

        # Categories
        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        return {
            "title":      title,
            "authors":    authors,
            "abstract":   abstract,
            "categories": categories,
            "arxiv_url":  arxiv_url,
            "pdf_link":   pdf_link,
            "published":  published,
            "arxiv_id":   arxiv_id,
        }

    def fetch_paper_text(self, paper: Dict[str, Any], max_chars: int = 8000) -> str:
        """
        Return the best available text for a paper.
        Uses abstract directly; optionally enriches with arxiv page text.
        Falls back gracefully — never raises.
        """
        # Start with abstract — always available and clean
        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()
        authors  = ", ".join(paper.get("authors", []))
        published = paper.get("published", "")

        header = f"Title: {title}\nAuthors: {authors}\nPublished: {published}\n\n"
        base   = header + abstract

        if len(base) >= max_chars or not paper.get("arxiv_url"):
            return base[:max_chars]

        # Try to fetch the arxiv abstract page for extra context
        try:
            from .web_search import fetch_url_text
            page_text = fetch_url_text(paper["arxiv_url"], max_chars=max_chars - len(base))
            if page_text:
                return (base + "\n\n" + page_text)[:max_chars]
        except Exception as e:
            logger.debug(f"Could not fetch arxiv page text: {e}")

        return base[:max_chars]

    def _guess_category(self, query: str) -> Optional[str]:
        """Keyword → arXiv category hint (informational only)."""
        q = query.lower()
        for keyword, cat in CATEGORY_MAP.items():
            if keyword in q:
                return cat
        return "cs.LG"  # Default: machine learning


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ArXiv paper search CLI")
    parser.add_argument("query",            help="Search query")
    parser.add_argument("--top",   "-n",    type=int, default=5,
                        help="Max papers to return")
    parser.add_argument("--json",           action="store_true",
                        help="Output as JSON")
    parser.add_argument("--fetch-text",     action="store_true",
                        help="Also fetch full text for each paper")
    args = parser.parse_args()

    crawler = ArxivCrawler()
    papers  = crawler.search_papers(args.query, max_results=args.top)

    if not papers:
        print("No papers found.")
        sys.exit(0)

    if args.fetch_text:
        for p in papers:
            p["text"] = crawler.fetch_paper_text(p)

    if args.json:
        print(json.dumps(papers, indent=2, ensure_ascii=False))
    else:
        for i, p in enumerate(papers, 1):
            print(f"\n[{i}] {p['title']}")
            print(f"    Authors:    {', '.join(p['authors'][:3])}"
                  + (" et al." if len(p['authors']) > 3 else ""))
            print(f"    Published:  {p['published']}")
            print(f"    Categories: {', '.join(p['categories'][:3])}")
            print(f"    URL:        {p['arxiv_url']}")
            if p.get("abstract"):
                snippet = p["abstract"][:200].replace("\n", " ")
                print(f"    Abstract:   {snippet}...")
            if args.fetch_text and p.get("text"):
                print(f"    Text chars: {len(p['text'])}")
