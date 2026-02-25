---
name: prompt-manager
description: Manage local parquet-based prompt and training data databases. Use this skill when the user asks to store, search, retrieve, or export prompts and conversations. Also activates when the user mentions datasets, training data, Unsloth, or fine-tuning data.
---

# Prompt Manager Skill

You have access to a local database of prompts and training conversations stored as parquet files.

## Data Location
`M:\claude_code_building_env\local_data\prompts\` (or `$CLAUDE_DATA_ROOT/prompts/`)

## Core Operations

### List all datasets
```bash
python -m scripts_and_skills.data.prompt_store list
```

### Add a prompt pair
```python
from scripts_and_skills.data.prompt_store import PromptStore
store = PromptStore()
store.add_prompt(
    dataset_name="my-dataset",
    input_text="What is X?",
    output_text="X is...",  # leave empty string for unlabelled entries
    tags=["topic", "category"],
    source="manual"
)
```

### Add a full conversation (ShareGPT format)
```python
store.add_conversation(
    dataset_name="my-dataset",
    messages=[
        {"from": "human",  "value": "How does attention work?"},
        {"from": "gpt",    "value": "Attention allows the model to..."},
    ],
    tags=["transformers", "nlp"]
)
```

### Search (keyword)
```bash
python -m scripts_and_skills.data.prompt_store search <dataset> "<query>"
```

### Search (semantic — requires nomic-embed-text)
```bash
python -m scripts_and_skills.data.embeddings search <dataset> "<query>" --top 5
python -m scripts_and_skills.data.embeddings search-all "<query>" --top 5
```

### Embed a dataset for semantic search
```bash
python -m scripts_and_skills.data.embeddings embed <dataset>
```
Requires Ollama running with `nomic-embed-text` loaded:
```bash
ollama pull nomic-embed-text
```

### Export to Unsloth JSONL
```bash
python -m scripts_and_skills.data.prompt_store export <dataset> output.jsonl --split train
```

## Dataset Schema

Each row contains:
- `id` — uuid
- `dataset_name` — logical name
- `split` — train / test / val
- `conversations` — JSON array of `{from, value}` objects (ShareGPT format)
- `input` / `output` — raw text for simple pairs
- `description` — human annotation
- `source` — origin (file path, url, manual, generated)
- `tags` — JSON array
- `created_at` — ISO timestamp
- `has_embedding` — bool

## When to use this skill

- User says "save this prompt" → `add_prompt()`
- User says "show me my datasets" → `list`
- User asks "find prompts about X" → `search`
- User wants training data → `export_unsloth()`
- User wants to fine-tune → export + point at the JSONL
