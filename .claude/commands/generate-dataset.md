# /generate-dataset — Generate training conversations from a document, web research, or ArXiv

Converts a file (Python, LaTeX, Markdown, code, text), a **web-researched topic**, or
**ArXiv academic papers** into ShareGPT-format training conversations stored in the local
parquet database. Optionally filters generated output through content moderation.

## Usage

**File / directory mode:**
```bash
python -m scripts_and_skills.data.dataset_generator \
    <file_or_directory> \
    --dataset <dataset-name> \
    --turns 3 \
    --chunks 20
```

**Web research mode:**
```bash
python -m scripts_and_skills.data.dataset_generator \
    --topic "Ollama Python API" \
    --dataset <dataset-name> \
    --search-top 5 \
    --turns 3
```

**Guided web research (topic + reference file):**
```bash
python -m scripts_and_skills.data.dataset_generator \
    --topic "Ollama embeddings API" \
    --context-file ./path/to/my_code.py \
    --dataset <dataset-name>
```
The `--context-file` steers every generated question toward how the topic
relates to that specific code/document.

**ArXiv paper mode:**
```bash
python -m scripts_and_skills.data.dataset_generator \
    --arxiv "transformer architecture" \
    --dataset <dataset-name> \
    --arxiv-top 5 \
    --turns 3
```

**Any mode with content moderation:**
```bash
# Works with any mode — just add --moderate to the command
python -m scripts_and_skills.data.dataset_generator my_file.py --dataset my-ds --moderate
python -m scripts_and_skills.data.dataset_generator --topic "ML" --dataset my-ds --moderate
python -m scripts_and_skills.data.dataset_generator --arxiv "ML" --dataset my-ds --moderate
```
`--moderate` passes every generated GPT response through `granite3-guardian:8b`
harm detection. Flagged conversations are skipped and counted as failed.
Requires `ollama pull granite3-guardian:8b`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Name to save conversations under |
| `--turns` | 3 | Q&A turns per content chunk |
| `--chunks` | 20 | Max chunks per file / max chunks per fetched page |
| `--model` | gpt-oss:20b | Ollama model for generation |
| `--split` | train | Dataset split (train/test/val) |
| `--no-tests` | off | Skip test/experiment conversations (file mode) |
| `--extensions` | varies | File extensions when processing a directory |
| `--topic` | — | Enable web research mode: search this topic |
| `--search-top` | 5 | Number of web pages to fetch and process |
| `--context-file` | — | Reference file to guide topic-mode question generation |
| `--arxiv` | — | Enable ArXiv mode: search academic papers for this query |
| `--arxiv-top` | 5 | Number of ArXiv papers to fetch and process |
| `--moderate` | off | Filter output through `granite3-guardian:8b` harm detection |

## Instructions

1. Ask whether the user wants **file mode**, **topic/research mode**, or **ArXiv mode**.
2. For file mode: ask for path if not given; confirm dataset name, chunks, turns.
3. For topic mode: ask for topic string and optional context file.
4. For ArXiv mode: ask for search query and number of papers (`--arxiv-top`).
5. Ask if they want moderation (`--moderate`) — note it requires `granite3-guardian:8b`.
6. Confirm all settings before running.
7. Run the command and report progress.
8. After completion, show stats:
   ```
   python -m scripts_and_skills.data.prompt_store stats <dataset-name>
   ```
9. Offer to run embeddings:
   ```
   python -m scripts_and_skills.data.embeddings embed <dataset-name>
   ```

## Doc type behavior (file mode)

- **Code files** (.py, .ts, .js etc): generates Q&A + test conversations
- **LaTeX** (.tex): generates Q&A + math walkthrough conversations
- **Markdown/Text**: generates Q&A conversations only

## Web research behavior (topic mode)

1. Searches DuckDuckGo for the topic (`--search-top` results)
2. Fetches and strips HTML from each page URL
3. Chunks the page text and generates Q&A conversations
4. If `--context-file` provided, each generation prompt includes a snippet of
   that file so questions are grounded in how the topic applies to your code

## ArXiv behavior (--arxiv mode)

1. Queries the ArXiv Atom API for `--arxiv-top` papers sorted by relevance
2. For each paper, uses the abstract + optional arxiv page text as content
3. Chunks the text and generates Q&A conversations (same pipeline as file/topic mode)
4. Tags conversations with `arxiv`, `qa`, the query string, and paper categories
5. ArXiv TOS: a 3-second rate-limit delay is applied between requests

## Moderation behavior (--moderate)

- After generating each conversation, the joined GPT responses are sent to
  `granite3-guardian:8b` with the `harm` category
- Flagged conversations are skipped and counted as `failed` in the result stats
- On model error or timeout: fail-safe → conversation is blocked
- Works with all three modes (file, topic, arxiv)
- Pull the model first: `ollama pull granite3-guardian:8b`

## Notes

- Requires Ollama running with the generation model loaded
- `OLLAMA_GEN_CTX=4096` is the default context — safe for 16 GB VRAM
- `OLLAMA_GEN_TIMEOUT=300` is the default timeout per Ollama call
- Large files are automatically chunked — no need to split manually
- Drop a whole `src/` directory to process an entire codebase
