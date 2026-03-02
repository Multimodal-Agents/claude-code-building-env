"""
prompt_store.py — Parquet-based prompt and conversation database.

Schema is Unsloth/ShareGPT compatible. Each row can be:
  - A full conversation  (conversations column, JSON array)
  - A simple prompt pair (input + output columns)
  - An input-only entry  (output left null — for future labelling)

Usage:
    store = PromptStore()
    store.add_conversation("my-dataset", [{"from":"human","value":"hi"},{"from":"gpt","value":"hello"}])
    store.add_prompt("my-dataset", input="What is X?", output="X is ...")
    df = store.load("my-dataset")
    store.search("my-dataset", "question about X")   # keyword search
    store.list_datasets()
    store.export_unsloth("my-dataset", "output.jsonl")
"""

import os
import re
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)

# ── Default data root (gitignored) ──────────────────────────────────────────
DEFAULT_DATA_ROOT = Path(os.getenv(
    "CLAUDE_DATA_ROOT",
    str(Path(__file__).parent.parent.parent / "local_data")
))

# ── Parquet schema ────────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("id",           pa.string()),          # uuid
    pa.field("dataset_name", pa.string()),          # logical dataset name
    pa.field("split",        pa.string()),          # train / test / val
    pa.field("conversations",pa.string()),          # JSON: [{from,value},...] ShareGPT format
    pa.field("messages",     pa.string()),          # JSON: [{role,content,thinking},...] OpenAI/gpt-oss format
    pa.field("input",        pa.string()),          # raw input text (for simple pairs)
    pa.field("output",       pa.string()),          # raw output text (nullable)
    pa.field("description",  pa.string()),          # human-readable description (nullable)
    pa.field("source",       pa.string()),          # origin: file path, url, manual, generated
    pa.field("tags",         pa.string()),          # JSON array of tag strings
    pa.field("created_at",   pa.string()),          # ISO 8601 timestamp
    pa.field("has_embedding",pa.bool_()),           # whether embedding exists in EmbeddingStore
])


# ── Format conversion helpers ─────────────────────────────────────────────────

def _is_openai_format(turns: List[Dict]) -> bool:
    """Return True if turns uses OpenAI format (role/content), False for ShareGPT (from/value)."""
    return bool(turns) and "role" in turns[0]


def _openai_to_sharegpt(turns: List[Dict]) -> List[Dict]:
    """Convert OpenAI messages [{role,content,thinking}] → ShareGPT [{from,value}].

    For assistant messages with thinking, we embed the thinking inside <think> tags
    so the content is recoverable from ShareGPT format if needed.
    """
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    result = []
    for msg in turns:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        thinking = msg.get("thinking") or ""
        sgpt_role = role_map.get(role, role)
        if role == "assistant" and thinking:
            value = f"<think>\n{thinking}\n</think>\n\n{content}"
        else:
            value = content
        result.append({"from": sgpt_role, "value": value})
    return result


def _sharegpt_to_openai(turns: List[Dict]) -> List[Dict]:
    """Convert ShareGPT [{from,value}] → OpenAI [{role,content,thinking}].

    Extracts <think>...</think> blocks into the thinking field if present.
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system", "assistant": "assistant"}
    result = []
    for msg in turns:
        sgpt_role = msg.get("from", "")
        value     = msg.get("value", "")
        role      = role_map.get(sgpt_role, sgpt_role)
        thinking  = None
        content   = value
        if role == "assistant":
            m = re.match(r"<think>\s*(.*?)\s*</think>\s*(.*)", value, re.DOTALL)
            if m:
                thinking = m.group(1).strip() or None
                content  = m.group(2).strip()
        result.append({"role": role, "content": content, "thinking": thinking})
    return result


class PromptStore:
    """
    Manages a collection of parquet files under local_data/prompts/.
    Each dataset_name maps to one parquet file.
    Multiple datasets can coexist.
    """

    def __init__(self, data_root: Optional[Path] = None):
        if not HAS_PANDAS:
            raise ImportError("pip install pandas pyarrow")
        self.root = Path(data_root or DEFAULT_DATA_ROOT) / "prompts"
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _path(self, dataset_name: str) -> Path:
        safe = dataset_name.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe}.parquet"

    def _load_raw(self, dataset_name: str) -> "pd.DataFrame":
        p = self._path(dataset_name)
        if not p.exists():
            return pd.DataFrame(columns=[f.name for f in SCHEMA])
        df = pd.read_parquet(p)
        # Back-fill any columns added after the file was written (schema evolution)
        for field in SCHEMA:
            if field.name not in df.columns:
                df[field.name] = False if field.type == pa.bool_() else ""
        return df

    def _save(self, df: "pd.DataFrame", dataset_name: str):
        p = self._path(dataset_name)
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        pq.write_table(table, p, compression="snappy")

    def _row(self, dataset_name: str, split: str, conversations: Optional[List[Dict]],
             input_text: str, output_text: str, description: str,
             source: str, tags: List[str],
             openai_messages: Optional[List[Dict]] = None) -> Dict:
        return {
            "id":            str(uuid.uuid4()),
            "dataset_name":  dataset_name,
            "split":         split,
            "conversations": json.dumps(conversations) if conversations else "",
            "messages":      json.dumps(openai_messages) if openai_messages else "",
            "input":         input_text or "",
            "output":        output_text or "",
            "description":   description or "",
            "source":        source or "manual",
            "tags":          json.dumps(tags or []),
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "has_embedding": False,
        }

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_conversation(self, dataset_name: str,
                         messages: List[Dict],
                         split: str = "train",
                         description: str = "",
                         source: str = "generated",
                         tags: Optional[List[str]] = None) -> str:
        """
        Add a conversation in either OpenAI or ShareGPT format.

        OpenAI format (gpt-oss/Multilingual-Thinking compatible):
          messages = [
            {"role": "system",    "content": "...", "thinking": None},
            {"role": "user",      "content": "...", "thinking": None},
            {"role": "assistant", "content": "...", "thinking": "reasoning..."},
          ]

        ShareGPT format (legacy):
          messages = [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

        Both are stored; the correct format is detected automatically.
        Returns the new row id.
        """
        if _is_openai_format(messages):
            openai_msgs  = messages
            sharegpt     = _openai_to_sharegpt(messages)
        else:
            sharegpt     = messages
            openai_msgs  = _sharegpt_to_openai(messages)

        # Flatten first user/assistant pair for quick search/display
        inp = next((m["value"] for m in sharegpt if m.get("from") == "human"), "")
        out = next((m["value"] for m in sharegpt if m.get("from") in ("gpt", "assistant")), "")

        row = self._row(dataset_name, split, sharegpt, inp, out, description, source, tags,
                        openai_messages=openai_msgs)
        df = self._load_raw(dataset_name)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save(df, dataset_name)
        return row["id"]

    def add_prompt(self, dataset_name: str,
                   input_text: str,
                   output_text: str = "",
                   split: str = "train",
                   description: str = "",
                   source: str = "manual",
                   tags: Optional[List[str]] = None) -> str:
        """
        Add a simple input/output pair without a full conversation structure.
        output_text can be empty — useful for inputs awaiting labelling.
        Returns the new row id.
        """
        messages = []
        if input_text:
            messages.append({"from": "human", "value": input_text})
        if output_text:
            messages.append({"from": "gpt", "value": output_text})
        row = self._row(dataset_name, split, messages or None,
                        input_text, output_text, description, source, tags)
        df = self._load_raw(dataset_name)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save(df, dataset_name)
        return row["id"]

    def add_batch(self, dataset_name: str, rows: List[Dict[str, Any]],
                  split: str = "train", source: str = "batch") -> List[str]:
        """Bulk insert. Each dict in rows should have at minimum 'input' key."""
        ids = []
        records = []
        for r in rows:
            msgs = r.get("conversations")
            inp = r.get("input", "")
            out = r.get("output", "")
            if not msgs and inp:
                msgs = [{"from": "human", "value": inp}]
                if out:
                    msgs.append({"from": "gpt", "value": out})
            row = self._row(dataset_name, r.get("split", split), msgs, inp, out,
                            r.get("description", ""), r.get("source", source),
                            r.get("tags"))
            records.append(row)
            ids.append(row["id"])
        df = self._load_raw(dataset_name)
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
        self._save(df, dataset_name)
        return ids

    def upsert_batch(self, dataset_name: str, rows: List[Dict[str, Any]],
                     split: str = "train", source: str = "batch") -> List[str]:
        """
        Upsert rows by matching on the 'input' field.
        Existing rows with the same input are replaced; new inputs are appended.
        Safe to call multiple times — idempotent.
        """
        ids = []
        records = []
        new_inputs = set()
        for r in rows:
            msgs = r.get("conversations")
            inp = r.get("input", "")
            out = r.get("output", "")
            if not msgs and inp:
                msgs = [{"from": "human", "value": inp}]
                if out:
                    msgs.append({"from": "gpt", "value": out})
            row = self._row(dataset_name, r.get("split", split), msgs, inp, out,
                            r.get("description", ""), r.get("source", source),
                            r.get("tags"))
            records.append(row)
            ids.append(row["id"])
            new_inputs.add(inp)
        df = self._load_raw(dataset_name)
        # Drop any existing rows whose input matches a row we're upserting
        if not df.empty and new_inputs:
            df = df[~df["input"].isin(new_inputs)]
        df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
        self._save(df, dataset_name)
        return ids

    def upsert_prompt(self, dataset_name: str,
                      input_text: str,
                      output_text: str = "",
                      split: str = "train",
                      description: str = "",
                      source: str = "manual",
                      tags: Optional[List[str]] = None) -> str:
        """
        Upsert a single prompt by input text.
        Replaces any existing row with the same input; otherwise appends.
        """
        messages = []
        if input_text:
            messages.append({"from": "human", "value": input_text})
        if output_text:
            messages.append({"from": "gpt", "value": output_text})
        row = self._row(dataset_name, split, messages or None,
                        input_text, output_text, description, source, tags)
        df = self._load_raw(dataset_name)
        if not df.empty:
            df = df[df["input"] != input_text]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save(df, dataset_name)
        return row["id"]

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, dataset_name: str,
             split: Optional[str] = None) -> "pd.DataFrame":
        """Load a dataset, optionally filtered by split."""
        df = self._load_raw(dataset_name)
        if split:
            df = df[df["split"] == split]
        return df.reset_index(drop=True)

    def get(self, dataset_name: str, row_id: str) -> Optional[Dict]:
        """Fetch a single row by id."""
        df = self._load_raw(dataset_name)
        row = df[df["id"] == row_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def search(self, dataset_name: str, query: str,
               columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Simple case-insensitive keyword search across text columns. Returns list of dicts."""
        df = self._load_raw(dataset_name)
        cols = columns or ["input", "output", "description", "conversations", "tags"]
        cols = [c for c in cols if c in df.columns]
        q = query.lower()
        mask = df[cols].apply(
            lambda col: col.astype(str).str.lower().str.contains(q, na=False)
        ).any(axis=1)
        return df[mask].reset_index(drop=True).to_dict("records")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets with row counts and splits."""
        results = []
        for p in sorted(self.root.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                results.append({
                    "name":   p.stem,
                    "rows":   len(df),
                    "splits": df["split"].value_counts().to_dict() if "split" in df else {},
                    "size":   f"{p.stat().st_size // 1024} KB",
                })
            except Exception as e:
                results.append({"name": p.stem, "error": str(e)})
        return results

    def stats(self, dataset_name: str) -> Dict[str, Any]:
        df = self._load_raw(dataset_name)
        return {
            "dataset":        dataset_name,
            "total_rows":     len(df),
            "splits":         df["split"].value_counts().to_dict() if not df.empty else {},
            "has_output":     int((df["output"] != "").sum()) if not df.empty else 0,
            "has_embedding":  int(df["has_embedding"].sum()) if not df.empty else 0,
            "sources":        df["source"].value_counts().to_dict() if not df.empty else {},
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def export_unsloth(self, dataset_name: str, output_path: str,
                       split: Optional[str] = "train") -> str:
        """
        Export to Unsloth/ShareGPT JSONL format.
        Each line: {"conversations": [{from, value}, ...]}
        """
        df = self.load(dataset_name, split=split)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                convs = row.get("conversations", "")
                if convs and convs != "null":
                    try:
                        parsed = json.loads(convs)
                        f.write(json.dumps({"conversations": parsed}) + "\n")
                        written += 1
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Exported {written} rows to {out}")
        return str(out)

    def _get_openai_messages(self, row: Dict) -> Optional[List[Dict]]:
        """Extract OpenAI messages from a row, preferring the messages column."""
        msgs_json = row.get("messages", "")
        if msgs_json:
            try:
                return json.loads(msgs_json)
            except Exception:
                pass
        convs_json = row.get("conversations", "")
        if convs_json and convs_json != "null":
            try:
                return _sharegpt_to_openai(json.loads(convs_json))
            except Exception:
                pass
        return None

    def export_gpt_oss_parquet(self, dataset_name: str, output_path: str,
                                split: Optional[str] = "train",
                                reasoning_language: str = "English") -> str:
        """
        Export in HuggingFaceH4/Multilingual-Thinking compatible parquet format.

        Produces columns: reasoning_language, developer, user, analysis, final, messages
        This is directly loadable by Unsloth's standardize_sharegpt + apply_chat_template
        pipeline for gpt-oss fine-tuning.

        reasoning_language — injected into system message as "reasoning language: {lang}"
                             so the model learns to reason in that language.
        """
        df = self.load(dataset_name, split=split)
        rows = []

        for _, row in df.iterrows():
            msgs = self._get_openai_messages(row)
            if not msgs:
                continue

            sys_msg  = next((m for m in msgs if m.get("role") == "system"),    None)
            user_msg = next((m for m in msgs if m.get("role") == "user"),      None)
            asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)

            if not (user_msg and asst_msg):
                continue

            # Rebuild system content with reasoning language prefix (matches HF dataset style)
            base_system = (sys_msg or {}).get("content", "You are a helpful AI assistant.")
            full_system  = f"reasoning language: {reasoning_language}\n\n{base_system}"

            analysis = (asst_msg.get("thinking") or "").strip()
            final    = (asst_msg.get("content")  or "").strip()

            # messages in OpenAI format with thinking — exactly as HF dataset
            hf_messages = [
                {"role": "system",    "content": full_system,         "thinking": None},
                {"role": "user",      "content": user_msg["content"], "thinking": None},
                {"role": "assistant", "content": final,               "thinking": analysis or None},
            ]

            rows.append({
                "reasoning_language": reasoning_language,
                "developer":          base_system,
                "user":               user_msg["content"],
                "analysis":           analysis,
                "final":              final,
                "messages":           hf_messages,
            })

        if not rows:
            logger.warning(f"No exportable rows found in '{dataset_name}' split='{split}'")
            return ""

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_df = pd.DataFrame(rows)
        # messages column must be stored as JSON string for cross-tool compat
        out_df["messages"] = out_df["messages"].apply(json.dumps)

        out_df.to_parquet(str(out_path), index=False, compression="snappy")
        logger.info(f"Exported {len(rows)} rows to {out_path} (gpt-oss parquet format)")
        return str(out_path)

    def delete_dataset(self, dataset_name: str):
        p = self._path(dataset_name)
        if p.exists():
            p.unlink()
            logger.info(f"Deleted dataset: {dataset_name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="PromptStore CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all datasets")
    p_stats = sub.add_parser("stats", help="Show stats for a dataset")
    p_stats.add_argument("dataset")

    p_search = sub.add_parser("search", help="Search a dataset")
    p_search.add_argument("dataset")
    p_search.add_argument("query")

    p_export = sub.add_parser("export", help="Export dataset to ShareGPT JSONL (Unsloth)")
    p_export.add_argument("dataset")
    p_export.add_argument("output")
    p_export.add_argument("--split", default="train")

    p_export_hf = sub.add_parser("export-gpt-oss", help="Export dataset to gpt-oss parquet (HF/Unsloth compatible)")
    p_export_hf.add_argument("dataset")
    p_export_hf.add_argument("output")
    p_export_hf.add_argument("--split", default="train")
    p_export_hf.add_argument("--lang", default="English", help="Reasoning language injected into system prompt")

    args = parser.parse_args()
    store = PromptStore()

    if args.cmd == "list":
        for ds in store.list_datasets():
            print(f"  {ds['name']:30} {ds.get('rows',0):>6} rows  {ds.get('size','')}")
    elif args.cmd == "stats":
        import pprint
        pprint.pprint(store.stats(args.dataset))
    elif args.cmd == "search":
        df = store.search(args.dataset, args.query)
        print(df[["id", "input", "output"]].to_string())
    elif args.cmd == "export":
        path = store.export_unsloth(args.dataset, args.output, args.split)
        print(f"Exported to: {path}")
    elif args.cmd == "export-gpt-oss":
        path = store.export_gpt_oss_parquet(args.dataset, args.output, args.split, args.lang)
        print(f"Exported to: {path}")
    else:
        parser.print_help()
