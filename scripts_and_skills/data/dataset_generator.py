"""
dataset_generator.py — Lite agent-chef for Claude Code.

Digests documents (code, LaTeX, Markdown, text) into ShareGPT training
conversations stored in the PromptStore parquet database.

Pipeline:
  1. Load document (auto-detect type: code / latex / markdown / plain)
  2. Chunk content intelligently (respects function/section boundaries)
  3. For each chunk, call Ollama to generate Q&A conversation turns
  4. For code/math: optionally generate test/experiment conversations
  5. Save all conversations to PromptStore

Generation uses the Ollama REST API directly (same model Claude Code uses).
Default model: gpt-oss:20b — override with OLLAMA_GEN_MODEL env var.

Usage:
    python -m scripts_and_skills.data.dataset_generator \\
        --file my_paper.tex \\
        --dataset my-training-data \\
        --turns 3 \\
        --chunks 10
"""

import os
import re
import json
import logging
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
GEN_MODEL    = os.getenv("OLLAMA_GEN_MODEL", "gpt-oss:20b")
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE",  "1500"))   # chars per chunk
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP", "200"))


# ── Document type detection ───────────────────────────────────────────────────

def detect_doc_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".py", ".ts", ".js", ".rs", ".go", ".java", ".cpp", ".c", ".cs"):
        return "code"
    if ext in (".tex", ".bib"):
        return "latex"
    if ext in (".md", ".mdx"):
        return "markdown"
    return "plain"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_code(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split code at function/class boundaries where possible."""
    boundary = re.compile(r"(?m)^(?:def |class |async def |function |fn |pub fn )")
    positions = [m.start() for m in boundary.finditer(text)]
    if not positions:
        return chunk_plain(text, size, overlap)
    chunks, current_start = [], 0
    for i, pos in enumerate(positions[1:], 1):
        segment = text[current_start:pos]
        if len(segment) > size:
            chunks.extend(chunk_plain(segment, size, overlap))
        else:
            chunks.append(segment.strip())
        current_start = max(current_start, pos - overlap)
    chunks.append(text[current_start:].strip())
    return [c for c in chunks if c.strip()]


def chunk_latex(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split LaTeX at section/subsection boundaries."""
    boundary = re.compile(r"(?m)^\\(?:section|subsection|subsubsection|paragraph)\{")
    positions = [m.start() for m in boundary.finditer(text)]
    if not positions:
        return chunk_plain(text, size, overlap)
    chunks = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segment = text[pos:end].strip()
        if len(segment) > size:
            chunks.extend(chunk_plain(segment, size, overlap))
        else:
            chunks.append(segment)
    return [c for c in chunks if c.strip()]


def chunk_plain(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding window chunking on paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        plen = len(para)
        if current_len + plen > size and current:
            chunks.append("\n\n".join(current))
            # Keep overlap
            while current and current_len > overlap:
                removed = current.pop(0)
                current_len -= len(removed)
        current.append(para)
        current_len += plen
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def chunk_document(path: Path, doc_type: Optional[str] = None) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dtype = doc_type or detect_doc_type(path)
    if dtype == "code":
        return chunk_code(text)
    if dtype == "latex":
        return chunk_latex(text)
    return chunk_plain(text)


# ── Ollama generation ─────────────────────────────────────────────────────────

GEN_CTX = int(os.getenv("OLLAMA_GEN_CTX", "4096"))   # keep small — KV cache is the VRAM killer


def _ollama_chat(messages: List[Dict[str, str]], model: str = GEN_MODEL) -> str:
    """Call Ollama /api/chat and return the assistant message content."""
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.7, "num_ctx": GEN_CTX},
    }
    timeout = int(os.getenv("OLLAMA_GEN_TIMEOUT", "300"))
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def generate_qa_conversation(chunk: str, doc_type: str,
                              num_turns: int = 3,
                              model: str = GEN_MODEL) -> List[Dict[str, str]]:
    """
    Given a content chunk, generate a multi-turn Q&A conversation.
    Returns ShareGPT format: [{"from": "human", "value": "..."}, ...]
    """
    type_hints = {
        "code":     "a software developer asking about the code",
        "latex":    "a researcher asking about the academic content",
        "markdown": "a technical reader asking about the documentation",
        "plain":    "a curious reader asking questions",
    }
    persona = type_hints.get(doc_type, type_hints["plain"])

    system = textwrap.dedent(f"""
        You are generating training data. Your task is to produce a realistic
        multi-turn conversation between {persona} and a knowledgeable AI assistant.
        
        Rules:
        - Generate EXACTLY {num_turns} question-answer pairs
        - Questions must be grounded in the provided content
        - Answers must be accurate, detailed, and directly answerable from the content
        - Output ONLY valid JSON: an array of objects with "from" and "value" keys
        - "from" is either "human" or "gpt"
        - No markdown, no explanation, just the JSON array
    """).strip()

    prompt = f"Content:\n```\n{chunk[:2000]}\n```\n\nGenerate {num_turns} Q&A turns as JSON array:"

    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": prompt},
    ]

    raw = _ollama_chat(messages, model)

    # Parse JSON from response
    try:
        # Strip markdown code blocks if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            # Normalize: ensure from/value keys
            normalized = []
            for item in parsed:
                if "from" in item and "value" in item:
                    normalized.append({"from": item["from"], "value": item["value"]})
            return normalized if normalized else []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse conversation JSON: {e}\nRaw: {raw[:200]}")

    return []


def generate_code_tests_conversation(chunk: str,
                                     model: str = GEN_MODEL) -> List[Dict[str, str]]:
    """
    For code chunks: generate a conversation about writing/understanding tests.
    """
    system = textwrap.dedent("""
        Generate a training conversation between a developer and an AI assistant
        about testing the provided code. Include: what to test, how to test it,
        and show example test code. Output ONLY a JSON array of {from, value} objects.
    """).strip()
    prompt = f"Code:\n```\n{chunk[:2000]}\n```\n\nGenerate test discussion as JSON array:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    raw = _ollama_chat(messages, model)
    try:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [{"from": i["from"], "value": i["value"]}
                    for i in parsed if "from" in i and "value" in i]
    except Exception:
        pass
    return []


def generate_math_experiment_conversation(chunk: str,
                                          model: str = GEN_MODEL) -> List[Dict[str, str]]:
    """
    For LaTeX math: generate a conversation that walks through the math step-by-step,
    including derivations and numerical examples where appropriate.
    """
    system = textwrap.dedent("""
        Generate a detailed mathematical tutoring conversation between a student
        and an expert AI. Walk through the mathematics step by step, verify results,
        and provide a concrete numerical example. Include LaTeX notation where appropriate.
        Output ONLY a JSON array of {from, value} objects.
    """).strip()
    prompt = f"Mathematical content:\n```\n{chunk[:2000]}\n```\n\nGenerate math walkthrough as JSON array:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    raw = _ollama_chat(messages, model)
    try:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [{"from": i["from"], "value": i["value"]}
                    for i in parsed if "from" in i and "value" in i]
    except Exception:
        pass
    return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

class DatasetGenerator:
    """
    Orchestrates document → conversation pipeline.
    Saves results to PromptStore.
    """

    def __init__(self, data_root: Optional[Path] = None,
                 model: str = GEN_MODEL):
        from .prompt_store import PromptStore
        self.store = PromptStore(data_root)
        self.model = model

    def process_file(self,
                     file_path: str,
                     dataset_name: str,
                     num_turns: int = 3,
                     max_chunks: int = 20,
                     generate_tests: bool = True,
                     split: str = "train") -> Dict:
        """
        Process a single document file into training conversations.
        Returns summary of what was generated.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_type = detect_doc_type(path)
        chunks = chunk_document(path, doc_type)[:max_chunks]
        logger.info(f"Processing {path.name} ({doc_type}): {len(chunks)} chunks")

        stats = {"file": str(path), "doc_type": doc_type,
                 "chunks": len(chunks), "conversations": 0,
                 "test_conversations": 0, "failed": 0}

        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i+1}/{len(chunks)}...")

            # Main Q&A conversation
            convs = generate_qa_conversation(chunk, doc_type, num_turns, self.model)
            if convs:
                self.store.add_conversation(
                    dataset_name, convs,
                    split=split,
                    description=f"QA from {path.name} chunk {i+1}",
                    source=str(path),
                    tags=[doc_type, "qa", path.stem],
                )
                stats["conversations"] += 1
            else:
                stats["failed"] += 1

            # Supplementary conversations
            if generate_tests:
                extra = []
                if doc_type == "code":
                    extra = generate_code_tests_conversation(chunk, self.model)
                elif doc_type == "latex" and any(k in chunk for k in ["equation", "theorem", "proof"]):
                    extra = generate_math_experiment_conversation(chunk, self.model)
                if extra:
                    self.store.add_conversation(
                        dataset_name, extra,
                        split=split,
                        description=f"{'Tests' if doc_type=='code' else 'Math walkthrough'} from {path.name} chunk {i+1}",
                        source=str(path),
                        tags=[doc_type, "tests" if doc_type=="code" else "math", path.stem],
                    )
                    stats["test_conversations"] += 1

        logger.info(f"Done: {stats['conversations']} conversations, "
                    f"{stats['test_conversations']} supplementary, "
                    f"{stats['failed']} failed")
        return stats

    def process_directory(self, dir_path: str, dataset_name: str,
                          extensions: Optional[List[str]] = None,
                          **kwargs) -> List[Dict]:
        """Process all matching files in a directory."""
        exts = set(extensions or [".py", ".ts", ".js", ".md", ".tex", ".txt"])
        results = []
        for f in sorted(Path(dir_path).rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                try:
                    r = self.process_file(str(f), dataset_name, **kwargs)
                    results.append(r)
                except Exception as e:
                    logger.error(f"Failed to process {f}: {e}")
                    results.append({"file": str(f), "error": str(e)})
        return results

    def process_topic(self, topic: str, dataset_name: str,
                      search_top: int = 5,
                      context_file: Optional[str] = None,
                      num_turns: int = 3,
                      max_chunks_per_page: int = 5,
                      split: str = "train") -> Dict:
        """
        Research a topic via web search, fetch each page's text, and generate
        training conversations from the content.

        context_file — optional path to a .py / .tex / .md file.
            When provided, a snippet of that file is included in every generation
            prompt so the model generates questions that relate the topic to that
            specific code or document.  Useful for building targeted datasets like
            "how does the Ollama API work, in the context of ollama_interface.py".
        """
        from .web_search import search, fetch_url_text

        # Build a context hint from the reference file if provided
        context_hint = ""
        if context_file:
            ctx_path = Path(context_file)
            if ctx_path.exists():
                ctx_text = ctx_path.read_text(encoding="utf-8", errors="replace")[:1500]
                context_hint = (
                    f"The research relates to the following file ({ctx_path.name}):\n\n"
                    f"{ctx_text}\n\n"
                    f"Generate questions that explore how the topic applies to or "
                    f"interacts with this code/document."
                )
                logger.info(f"Context file loaded: {ctx_path.name}")
            else:
                logger.warning(f"Context file not found: {context_file}")

        logger.info(f"Searching: '{topic}' (top {search_top})")
        results = search(topic, top=search_top)
        results = [r for r in results if not r.get("error") and r.get("url")]

        stats = {
            "topic":            topic,
            "pages_found":      len(results),
            "pages_processed":  0,
            "conversations":    0,
            "failed":           0,
        }

        for r in results:
            url   = r["url"]
            title = r.get("title", url)
            logger.info(f"  Fetching: {title[:70]}")

            page_text = fetch_url_text(url)
            if not page_text:
                stats["failed"] += 1
                continue

            chunks = chunk_plain(page_text)[:max_chunks_per_page]
            stats["pages_processed"] += 1

            for i, chunk in enumerate(chunks):
                guided = f"{context_hint}\n\n---\n\n{chunk}" if context_hint else chunk
                convs = generate_qa_conversation(guided, "plain", num_turns, self.model)
                if convs:
                    self.store.add_conversation(
                        dataset_name, convs,
                        split=split,
                        description=f"Web: {title[:80]} chunk {i+1}",
                        source=url,
                        tags=["web-research", "qa", topic[:30]],
                    )
                    stats["conversations"] += 1
                else:
                    stats["failed"] += 1

        logger.info(f"Topic done: {stats['pages_processed']} pages, "
                    f"{stats['conversations']} conversations, "
                    f"{stats['failed']} failed")
        return stats


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="DatasetGenerator CLI — file/dir mode or web research mode",
        epilog=(
            "File mode:  python -m ... my_file.py --dataset my-ds\n"
            "Topic mode: python -m ... --topic 'Ollama API' --dataset my-ds\n"
            "Guided:     python -m ... --topic 'Ollama API' --context-file ollama_interface.py --dataset my-ds"
        ),
    )
    parser.add_argument("file",           nargs="?",           help="File or directory to process (omit when using --topic)")
    parser.add_argument("--dataset",      required=True,       help="Dataset name to save to")
    parser.add_argument("--turns",        type=int, default=3, help="QA turns per chunk")
    parser.add_argument("--chunks",       type=int, default=20,help="Max chunks per file / per page in topic mode")
    parser.add_argument("--model",        default=GEN_MODEL,   help="Ollama model")
    parser.add_argument("--split",        default="train")
    parser.add_argument("--no-tests",     action="store_true", help="Skip test/math conversation generation")
    parser.add_argument("--extensions",   nargs="*",           help="File extensions for dir mode")
    # Web research mode
    parser.add_argument("--topic",        default=None,        help="Search topic — enables web research mode instead of file mode")
    parser.add_argument("--search-top",   type=int, default=5, help="Number of search results to fetch in topic mode")
    parser.add_argument("--context-file", default=None,        help="Reference file (.py/.tex/.md) to guide question generation in topic mode")

    args = parser.parse_args()

    if not args.file and not args.topic:
        parser.error("Provide either a file/directory argument or --topic for web research mode.")

    gen = DatasetGenerator(model=args.model)

    if args.topic:
        result = gen.process_topic(
            args.topic, args.dataset,
            search_top=args.search_top,
            context_file=args.context_file,
            num_turns=args.turns,
            max_chunks_per_page=args.chunks,
            split=args.split,
        )
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        p = Path(args.file)
        if p.is_dir():
            results = gen.process_directory(
                str(p), args.dataset,
                extensions=args.extensions,
                num_turns=args.turns,
                max_chunks=args.chunks,
                generate_tests=not args.no_tests,
                split=args.split,
            )
            total_convs = sum(r.get("conversations", 0) for r in results)
            print(f"\nProcessed {len(results)} files → {total_convs} conversations in '{args.dataset}'")
        else:
            result = gen.process_file(
                str(p), args.dataset,
                num_turns=args.turns,
                max_chunks=args.chunks,
                generate_tests=not args.no_tests,
                split=args.split,
            )
            print(f"\nResult: {json.dumps(result, indent=2)}")
