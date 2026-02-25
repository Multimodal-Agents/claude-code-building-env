# /generate-dataset — Generate training conversations from a document or web research

Converts a file (Python, LaTeX, Markdown, code, text) **or a web-researched topic**
into ShareGPT-format training conversations stored in the local parquet database.

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

## Instructions

1. Ask whether the user wants **file mode** or **topic/research mode**.
2. For file mode: ask for path if not given; confirm dataset name, chunks, turns.
3. For topic mode: ask for topic string and optional context file.
4. Confirm all settings before running.
5. Run the command and report progress.
6. After completion, show stats:
   ```
   python -m scripts_and_skills.data.prompt_store stats <dataset-name>
   ```
7. Offer to run embeddings:
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

## Notes

- Requires Ollama running with the generation model loaded
- `OLLAMA_GEN_CTX=4096` is the default context — safe for 16 GB VRAM
- `OLLAMA_GEN_TIMEOUT=300` is the default timeout per Ollama call
- Large files are automatically chunked — no need to split manually
- Drop a whole `src/` directory to process an entire codebase
