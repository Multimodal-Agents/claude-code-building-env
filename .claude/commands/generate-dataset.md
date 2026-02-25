# /generate-dataset — Generate training conversations from a document

Converts a file (Python, LaTeX, Markdown, code, text) into ShareGPT-format
training conversations stored in the local parquet database.

## Usage

The user will provide a file path or directory. Run:

```bash
python -m scripts_and_skills.data.dataset_generator \
    <file_or_directory> \
    --dataset <dataset-name> \
    --turns 3 \
    --chunks 20
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Name to save conversations under |
| `--turns` | 3 | Q&A turns per content chunk |
| `--chunks` | 20 | Max chunks to process per file |
| `--model` | gpt-oss:20b | Ollama model for generation |
| `--split` | train | Dataset split (train/test/val) |
| `--no-tests` | off | Skip test/experiment conversations |
| `--extensions` | varies | File extensions when processing a directory |

## Instructions

1. Ask the user for the file path if not provided: "What file do you want to process?"
2. Ask for a dataset name if not provided: "What should this dataset be called?"
3. Confirm the settings before running (file, chunks, turns, dataset name).
4. Run the command and report progress.
5. After completion, show stats:
   ```
   python -m scripts_and_skills.data.prompt_store stats <dataset-name>
   ```
6. Offer to also run embeddings:
   ```
   python -m scripts_and_skills.data.embeddings embed <dataset-name>
   ```
   (Only if nomic-embed-text is available in Ollama.)

## Doc type behavior

- **Code files** (.py, .ts, .js etc): generates Q&A + test conversations
- **LaTeX** (.tex): generates Q&A + math walkthrough conversations  
- **Markdown/Text**: generates Q&A conversations only

## Notes

- Requires Ollama running with the generation model loaded
- Large files are automatically chunked — no need to split manually
- Drop a whole `src/` directory to process an entire codebase
