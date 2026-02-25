# Quickstart Guide

> Get from zero to a custom Ollama model with a CoreCoder system prompt in ~10 minutes.

---

## Prerequisites

| Requirement | Check |
|-------------|-------|
| Python 3.10+ | `python --version` |
| Ollama installed + running | `ollama list` |
| `gpt-oss:20b` pulled in Ollama | `ollama list \| findstr gpt-oss` |
| Git | `git --version` |

---

## Step 0 — Create & activate the virtual environment

A top-level `venv/` lives at the repo root. Create it once:

```powershell
python -m venv venv
```

Activate it **every time** before running any Python commands:

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your prompt. All `python` commands below assume this is active.

> **Already done?** If `venv/` exists and packages are installed, skip straight to Step 2.

---

## Step 1 — Install Python dependencies

> Activate the venv first (Step 0).

```powershell
# Data layer (prompts, embeddings, dataset generator, web search)
pip install -r scripts_and_skills/data/requirements.txt

# Model manager (Ollama API, Modelfile, GGUF, LoRA)
pip install -r scripts_and_skills/model_manager/requirements.txt

# For HuggingFace downloads
pip install huggingface_hub
```

Or install everything in one shot:

```powershell
pip install pandas pyarrow numpy requests ddgs huggingface_hub
```

---

## Step 2 — Pull the embedding model

The semantic search system uses `nomic-embed-text`:

```powershell
ollama pull nomic-embed-text
```

---

## Step 3 — Seed the CoreCoder prompt dataset

```powershell
cd M:\claude_code_building_env
.\venv\Scripts\python -m scripts_and_skills.data.seeds.corecoder_prompts
```

The seed script is **idempotent** — safe to run multiple times. Rows are upserted, not duplicated.

Expected output:
```
[corecoder seed] dataset='corecoder-vscode-copilot' rows=19
Done. Run embeddings next:
  python -m scripts_and_skills.data.embeddings embed corecoder-vscode-copilot
```

---

## Step 4 — Index the dataset for semantic search

```powershell
.\venv\Scripts\python -m scripts_and_skills.data.embeddings embed corecoder-vscode-copilot
```

---

## Step 5 — Verify everything is working

```powershell
# List all datasets
.\venv\Scripts\python -m scripts_and_skills.data.prompt_store list

# Check row count
.\venv\Scripts\python -m scripts_and_skills.data.prompt_store stats corecoder-vscode-copilot

# Semantic search test (after Step 4)
.\venv\Scripts\python -m scripts_and_skills.data.embeddings search corecoder-vscode-copilot "investigate terminal" --top 3

# List Ollama models
.\venv\Scripts\python -m scripts_and_skills.model_manager.ollama_api list

# Check Ollama version
.\venv\Scripts\python -m scripts_and_skills.model_manager.ollama_api version
```

---

## Step 6 — Create a CoreCoder model (system prompt embedded)

> **Already done?** `corecoder:latest` is already in Ollama (`ollama list | findstr corecoder`). Skip to Step 7.

This bakes the CoreCoder system prompt directly into the Ollama model:

```python
# Run this in a Python terminal
from scripts_and_skills.model_manager import ModelfileBuilder
from scripts_and_skills.data.prompt_store import PromptStore

# Load the system prompt from the database
store = PromptStore()
rows = store.search("corecoder-vscode-copilot", "system prompt")
system_prompt = rows[0]["output"]

# Build and deploy
b = ModelfileBuilder.from_existing("gpt-oss:20b")
b.set_system(system_prompt)
b.set_parameter("temperature", 0.2)
b.set_parameter("num_ctx", 8192)
b.create_model("corecoder:latest")
```

Or via CLI:
```powershell
.\venv\Scripts\python -m scripts_and_skills.model_manager.modelfile create corecoder:latest `
    --from gpt-oss:20b `
    --system "You are CoreCoder..." `
    --temperature 0.2 `
    --ctx 8192
```

> Uses the Ollama 0.17+ structured create API (no `modelfile` string field).

---

## Step 7 — Use the model

```powershell
# Test it
ollama run corecoder:latest "List the files in the current directory"

# Launch Claude Code with it
claude --model corecoder:latest

# Or update the launcher permanently
# Edit scripts_and_skills/claude_scripts/run_claude.ps1:
#   claude --model corecoder:latest
```

---

## Step 8 — (Optional) Download newer GGUFs from HuggingFace

See what quantizations are available for the gpt-oss-20b GGUF repo:

```powershell
.\venv\Scripts\python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF --list
```

Pick the right quant for your VRAM:

| VRAM | Recommended quant |
|------|------------------|
| 12 GB (Titan XP) | `Q4_K_M` |
| 16 GB | `Q5_K_M` |
| 24 GB | `Q8_0` |

Download and deploy in one command:
```powershell
.\venv\Scripts\python -m scripts_and_skills.model_manager.hf_download unsloth/gpt-oss-20b-GGUF `
    --file "gpt-oss-20b-Q4_K_M.gguf" `
    --out "M:/models/gguf" `
    --deploy --name "gpt-oss:20b-q4"
```

---

## Step 9 — (Optional) Setup llama.cpp for conversion + quantization

Only needed if you want to convert HuggingFace safetensors models to GGUF yourself,
or quantize existing GGUFs with `llama-quantize`.

```powershell
# Python tools only (no C++ build needed for conversion)
.\setup_llamacpp.ps1 -SkipBuild

# Full build with CUDA (Titan XP — takes ~5 min, requires cmake)
.\setup_llamacpp.ps1 -CudaBuild

# After setup, convert a HF model:
.\venv\Scripts\python -m scripts_and_skills.model_manager.gguf_manager convert `
    "C:\models\hf\my-model" `
    "M:\models\gguf\my-model-f16.gguf" `
    --script "M:\llama.cpp\convert_hf_to_gguf.py"
```

---

## Slash commands (use inside Claude Code sessions)

| Command | What it does |
|---------|-------------|
| `/prompt-list` | Browse and search the prompt database |
| `/set-system` | Change the system prompt of any Ollama model interactively |
| `/generate-dataset` | Convert docs/code files into ShareGPT training conversations |
| `/parse-codebase` | Smart codebase exploration (minimal token usage) |
| `/web-search` | DuckDuckGo search (useful for local model sessions) |
| `/model-manager` | Manage models, GGUFs, Modelfiles, LoRA adapters |

---

## What's working right now

| Component | Status | Notes |
|-----------|--------|-------|
| `PromptStore` | ✅ Ready | Run the seed script first |
| `EmbeddingStore` | ✅ Ready | Requires `nomic-embed-text` pulled in Ollama |
| `DatasetGenerator` | ✅ Ready | Requires `gpt-oss:20b` running |
| `WebSearch` | ✅ Ready | Requires `pip install duckduckgo-search` |
| `OllamaAPI` | ✅ Ready | Requires Ollama running |
| `ModelfileBuilder` | ✅ Ready | Works immediately |
| `GGUFManager` | ✅ Ready | Registry works; conversion needs llama.cpp |
| `LoRAManager` | ✅ Ready | Deployment needs a GGUF adapter file |
| `hf_download` | ✅ Ready | Requires `pip install huggingface_hub` |
| llama.cpp | ⚙️ Optional | Run `setup_llamacpp.ps1` to enable conversion |

---

## Troubleshooting

**`ollama: command not found`** — Ollama isn't in PATH. Launch Ollama Desktop or add `C:\Users\<you>\AppData\Local\Programs\Ollama` to PATH.

**`Connection refused` on port 11434** — Ollama isn't running. Start it: `ollama serve` or launch Ollama Desktop.

**`ModuleNotFoundError: pandas`** — Activate the venv first (`.\venv\Scripts\Activate.ps1`), then run `pip install -r scripts_and_skills/data/requirements.txt`

**Web search returns no results or import error** — The package was renamed: `pip install ddgs`

**`nomic-embed-text not found`** — Run `ollama pull nomic-embed-text`

**GGUF conversion fails** — Make sure you ran `setup_llamacpp.ps1` and the convert script path is correct.
