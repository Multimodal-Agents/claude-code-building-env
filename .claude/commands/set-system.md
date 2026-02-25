# /set-system — Change the system prompt of any Ollama model interactively

This command lets you swap the system prompt of any Ollama model, either by typing
one, or by selecting from a saved prompt set in the local parquet database.

---

## Instructions

### Step 1 — Show available prompt datasets

Run this to list what's in the database:
```
python -m scripts_and_skills.data.prompt_store list
```

If the user says "use CoreCoder" or mentions a dataset by name, skip to Step 3.
If no datasets exist yet: suggest running `python -m scripts_and_skills.data.seeds.corecoder_prompts` first.

### Step 2 — Load and show system prompts from a dataset

Find entries tagged as system prompts:
```python
from scripts_and_skills.data.prompt_store import PromptStore
store = PromptStore()
rows = store.search("corecoder-vscode-copilot", "system prompt")
for i, r in enumerate(rows[:10]):
    print(f"[{i}] {r.get('description','')} — {r.get('output','')[:80]}...")
```

Show the numbered list to the user and ask which one to use, OR if they say
"type my own", skip to Step 4.

### Step 3 — Let user pick or type a system prompt

Ask:
> "Which system prompt do you want? Enter a number from the list, or type your own text."

Wait for the user's response. Then:
- If they give a number → use `rows[n]['output']` as the system prompt text
- If they type text directly → use that text as the system prompt

### Step 4 — Choose base model and target name

Ask:
> "Which base model should this be built on? (default: gpt-oss:20b)"
> "What should the new model be called? (e.g. corecoder:latest)"

Reasonable defaults:
- base model: `gpt-oss:20b`
- new name: `corecoder:latest` (if using CoreCoder prompt) or `custom:latest`

### Step 5 — Preview the Modelfile before creating

```python
from scripts_and_skills.model_manager import ModelfileBuilder

b = ModelfileBuilder.from_existing("<base_model>")
b.set_system("""<system_prompt>""")
b.set_parameter("temperature", 0.2)
print(b.build())
```

Show the user the rendered Modelfile and ask: "Create this model? (yes/no)"

### Step 6 — Create the model

If yes:
```python
b.create_model("<new_model_name>")
```

After creation, tell the user:
> "Model '<name>' is ready. To use it, update your launcher or run:
>  `claude --model <name>`
>  Or restart this session with: `ollama run <name>`"

### Optionally — save the Modelfile for later

```python
b.save(name="<new_model_name>")
```
Saves to `local_data/modelfiles/<name>.modelfile`

---

## Quick shortcuts the user might say

- "set system prompt to X" → skip directly to Step 5 with the provided text
- "use the corecoder prompt" → auto-select the CoreCoder system prompt, ask only for model name
- "show me my prompts" → run Step 1 + Step 2 only
- "reset to defaults" → rebuild from base without a custom system prompt
