# /prompt-list — Browse and interact with the prompt/conversation database

You have access to a local parquet-based prompt database stored in `local_data/prompts/`.

## What you can do

**List all datasets:**
```
python -m scripts_and_skills.data.prompt_store list
```

**Show stats for a dataset:**
```
python -m scripts_and_skills.data.prompt_store stats <dataset-name>
```

**Search a dataset by keyword:**
```
python -m scripts_and_skills.data.prompt_store search <dataset-name> "<query>"
```

**Semantic search (requires nomic-embed-text running in Ollama):**
```
python -m scripts_and_skills.data.embeddings search <dataset-name> "<query>" --top 5
python -m scripts_and_skills.data.embeddings search-all "<query>" --top 5
```

**Export a dataset to Unsloth JSONL:**
```
python -m scripts_and_skills.data.prompt_store export <dataset-name> output.jsonl --split train
```

## Instructions

1. Run the list command first to show the user what datasets exist.
2. If the user gives a topic or question, run semantic search across all datasets.
3. If asked to load a prompt, show the user the top results and ask which to use.
4. If asked to add a prompt, use the Python API:
   ```python
   from scripts_and_skills.data.prompt_store import PromptStore
   store = PromptStore()
   store.add_prompt("dataset-name", input_text="...", output_text="...", tags=["tag1"])
   ```
5. Always confirm the operation and show the result count.
6. If no datasets exist yet, inform the user and suggest running `/generate-dataset`.
