"""
embeddings.py — nomic-embed-text via Ollama REST API.

Stores embeddings in a separate parquet index alongside the prompt store.
Provides cosine similarity search over any dataset.

Model: nomic-embed-text (requires Ollama 0.1.26+)
  ollama pull nomic-embed-text

Usage:
    emb = EmbeddingStore()
    emb.embed_dataset("my-dataset")          # embed all rows missing embeddings
    results = emb.search("my-dataset", "what is attention?", top_k=5)
    for r in results:
        print(r["score"], r["input"])
"""

import os
import json
import logging
import struct
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(os.getenv(
    "CLAUDE_DATA_ROOT",
    str(Path(__file__).parent.parent.parent / "local_data")
))

OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM     = 768   # nomic-embed-text output dimension

EMB_SCHEMA = pa.schema([
    pa.field("id",           pa.string()),
    pa.field("dataset_name", pa.string()),
    pa.field("text",         pa.string()),
    pa.field("embedding",    pa.binary()),   # numpy float32 array packed as bytes
    pa.field("created_at",   pa.string()),
])


def _vec_to_bytes(vec: List[float]) -> bytes:
    arr = [float(v) for v in vec]
    return struct.pack(f"{len(arr)}f", *arr)


def _bytes_to_vec(b: bytes) -> "np.ndarray":
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingStore:
    """
    Manages embedding vectors for prompt store rows.
    Uses nomic-embed-text via Ollama REST API.
    Stores vectors in local_data/embeddings/<dataset_name>.parquet
    """

    def __init__(self, data_root: Optional[Path] = None):
        if not HAS_REQUESTS:
            raise ImportError("pip install requests")
        if not HAS_NUMPY:
            raise ImportError("pip install pandas pyarrow numpy")
        self.root = Path(data_root or DEFAULT_DATA_ROOT) / "embeddings"
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Ollama REST call ──────────────────────────────────────────────────────

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Call Ollama nomic-embed-text and return embedding vector.

        Tries the current /api/embed endpoint (Ollama 0.4+) first;
        falls back to the legacy /api/embeddings endpoint.
        /api/embed  : payload={model, input},  response={embeddings: [[...]]}
        /api/embeddings: payload={model, prompt}, response={embedding: [...]}
        """
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            # /api/embed returns {"embeddings": [[float, ...]]}
            embeddings = data.get("embeddings")
            if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
                return embeddings[0]
            # Fallback: legacy {"embedding": [float, ...]}
            return data.get("embedding")
        except requests.RequestException:
            pass
        # Legacy fallback for older Ollama builds
        try:
            r = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("embedding")
        except requests.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts. Returns list with None on failure."""
        return [self.embed_text(t) for t in texts]

    # ── Index management ──────────────────────────────────────────────────────

    def _path(self, dataset_name: str) -> Path:
        safe = dataset_name.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.parquet"

    def _load_index(self, dataset_name: str) -> "pd.DataFrame":
        p = self._path(dataset_name)
        if not p.exists():
            return pd.DataFrame(columns=[f.name for f in EMB_SCHEMA])
        return pd.read_parquet(p)

    def _save_index(self, df: "pd.DataFrame", dataset_name: str):
        p = self._path(dataset_name)
        table = pa.Table.from_pandas(df, schema=EMB_SCHEMA, preserve_index=False)
        pq.write_table(table, p, compression="snappy")

    def embed_dataset(self, dataset_name: str,
                      text_column: str = "input",
                      force: bool = False) -> int:
        """
        Embed all rows in a prompt store dataset that don't have embeddings yet.
        Loads the dataset from PromptStore, embeds the text_column, saves index.
        Returns number of new embeddings added.
        """
        from .prompt_store import PromptStore
        store = PromptStore(self.root.parent)   # self.root = local_data/embeddings → parent = local_data/
        df = store.load(dataset_name)
        idx = self._load_index(dataset_name)
        existing_ids = set(idx["id"].tolist()) if not idx.empty else set()

        to_embed = df if force else df[~df["id"].isin(existing_ids)]
        if to_embed.empty:
            logger.info(f"All rows already embedded in dataset '{dataset_name}'")
            return 0

        new_rows = []
        for _, row in to_embed.iterrows():
            text = str(row.get(text_column, "") or row.get("conversations", ""))[:2048]
            vec = self.embed_text(text)
            if vec:
                new_rows.append({
                    "id":           row["id"],
                    "dataset_name": dataset_name,
                    "text":         text,
                    "embedding":    _vec_to_bytes(vec),
                    "created_at":   datetime.now(timezone.utc).isoformat(),
                })
                logger.debug(f"Embedded row {row['id']}")

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined = pd.concat([idx, new_df], ignore_index=True) if not idx.empty else new_df
            self._save_index(combined, dataset_name)
            # Update has_embedding flag in prompt store
            embedded_ids = {r["id"] for r in new_rows}
            df.loc[df["id"].isin(embedded_ids), "has_embedding"] = True
            store._save(df, dataset_name)

        logger.info(f"Added {len(new_rows)} embeddings for '{dataset_name}'")
        return len(new_rows)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, dataset_name: str, query: str,
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over a dataset's embedding index.
        Returns top_k results sorted by cosine similarity.
        """
        query_vec = self.embed_text(query)
        if query_vec is None:
            logger.error("Could not embed query — is Ollama running with nomic-embed-text?")
            return []

        idx = self._load_index(dataset_name)
        if idx.empty:
            logger.warning(f"No embeddings found for dataset '{dataset_name}'. Run embed_dataset() first.")
            return []

        q = np.array(query_vec, dtype=np.float32)
        scores = []
        for _, row in idx.iterrows():
            vec = _bytes_to_vec(row["embedding"])
            scores.append((cosine_similarity(q, vec), row["id"], row["text"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        # Fetch full rows from prompt store
        from .prompt_store import PromptStore
        store = PromptStore(self.root.parent)   # self.root = local_data/embeddings → parent = local_data/
        df = store.load(dataset_name)

        results = []
        for score, row_id, text in top:
            row = df[df["id"] == row_id]
            if not row.empty:
                r = row.iloc[0].to_dict()
                r["score"] = round(score, 4)
                results.append(r)

        return results

    def search_all(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search across ALL embedded datasets."""
        all_results = []
        for p in sorted(self.root.glob("*.parquet")):
            dataset_name = p.stem
            results = self.search(dataset_name, query, top_k=top_k)
            for r in results:
                r["_dataset"] = dataset_name
            all_results.extend(results)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]

    def list_indexed(self) -> List[Dict[str, Any]]:
        """List all indexed datasets."""
        results = []
        for p in sorted(self.root.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                results.append({
                    "name":  p.stem,
                    "count": len(df),
                    "size":  f"{p.stat().st_size // 1024} KB",
                })
            except Exception as e:
                results.append({"name": p.stem, "error": str(e)})
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EmbeddingStore CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_embed = sub.add_parser("embed", help="Embed a dataset")
    p_embed.add_argument("dataset")
    p_embed.add_argument("--column", default="input")
    p_embed.add_argument("--force", action="store_true")

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("dataset")
    p_search.add_argument("query")
    p_search.add_argument("--top", type=int, default=5)

    p_search_all = sub.add_parser("search-all", help="Search across all datasets")
    p_search_all.add_argument("query")
    p_search_all.add_argument("--top", type=int, default=5)

    sub.add_parser("list", help="List indexed datasets")

    args = parser.parse_args()
    store = EmbeddingStore()

    if args.cmd == "embed":
        n = store.embed_dataset(args.dataset, args.column, args.force)
        print(f"Added {n} embeddings")
    elif args.cmd == "search":
        results = store.search(args.dataset, args.query, args.top)
        for r in results:
            print(f"  [{r['score']}] {r.get('input','')[:80]}")
    elif args.cmd == "search-all":
        results = store.search_all(args.query, args.top)
        for r in results:
            print(f"  [{r['score']}] ({r.get('_dataset','')}) {r.get('input','')[:80]}")
    elif args.cmd == "list":
        for ds in store.list_indexed():
            print(f"  {ds['name']:30} {ds.get('count',0):>6} vectors  {ds.get('size','')}")
    else:
        parser.print_help()
