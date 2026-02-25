# claude-code-building-environment

> Personal AI development abstraction layer built on top of [Claude Code CLI](https://github.com/anthropics/claude-code).

Run a local LLM (Ollama `gpt-oss:20b` - 14 GB) through Claude Code's full agentic tooling —
with your own skills, plugins, and hooks that survive every upstream update.

---

## Quick Start

### 1. Clone this repo
```powershell
git clone https://github.com/Multimodal-Agents/claude-code-building-env.git
cd claude-code-building-env
```

### 2. Run setup
```powershell
.\setup.ps1
```

This clones two Anthropic reference repos into their expected locations:
- `scripts_and_skills/claude_skills/skills/` ← [anthropics/skills](https://github.com/anthropics/skills)
- `claude_code_custom/claude-code/` ← [anthropics/claude-code](https://github.com/anthropics/claude-code)

**If you already have `claude-code` cloned elsewhere:**
```powershell
.\setup.ps1 -ClaudeCodeDir "C:\your\existing\claude-code"
```

**Skip either clone:**
```powershell
.\setup.ps1 -SkipClaudeCode   # skip claude-code ref
.\setup.ps1 -SkipSkills        # skip skills ref
```

### 3. Create Python venv and install dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas pyarrow numpy requests ddgs huggingface_hub
```

### 4. Launch
```powershell
.\scripts_and_skills\claude_scripts\run_claude.ps1
```
Or point your desktop shortcut to:
```
powershell.exe -ExecutionPolicy Bypass -File "M:\claude_code_building_env\scripts_and_skills\claude_scripts\run_claude.ps1"
```

See [QUICKSTART.md](QUICKSTART.md) for the full step-by-step guide including seeding the CoreCoder dataset and creating the `corecoder:latest` model.

---

## Repository Structure

```
.
├── assets/                              # Icons and static media
├── CLAUDE.md                            # Global context injected at every session
├── QUICKSTART.md                        # Step-by-step setup guide
├── setup.ps1                            # One-time repo setup
├── setup_llamacpp.ps1                   # Optional: llama.cpp build + Python tools
├── venv/                                # Top-level Python virtual environment (git-ignored)
├── .claude/
│   └── commands/                        # Slash commands available in every session
├── scripts_and_skills/
│   ├── claude_scripts/
│   │   └── run_claude.ps1               # Main entry point / launcher
│   ├── data/                            # Data layer
│   │   ├── prompt_store.py              # Parquet-backed prompt + conversation CRUD
│   │   ├── embeddings.py                # nomic-embed-text semantic search
│   │   ├── dataset_generator.py         # Docs/code → ShareGPT training conversations
│   │   ├── web_search.py                # DuckDuckGo search wrapper
│   │   └── seeds/
│   │       └── corecoder_prompts.py     # Seeds 19-row CoreCoder prompt dataset
│   ├── model_manager/                   # Ollama model lifecycle
│   │   ├── ollama_api.py                # REST wrapper: list/create/delete/pull/embed
│   │   ├── modelfile.py                 # Modelfile builder (system prompt, params, adapters)
│   │   ├── gguf_manager.py              # GGUF registry + HF→GGUF conversion + quantize
│   │   ├── lora_manager.py              # LoRA adapter registry + Ollama deployment
│   │   └── hf_download.py               # Download GGUFs from HuggingFace Hub
│   └── claude_skills/
│       ├── skills/                      # Official Anthropic skills (reference, git-ignored)
│       ├── blues_skills/                # Custom domain skills
│       └── template_skills/             # Scaffold for new skills
├── local_data/                          # Runtime data (git-ignored, machine-local)
│   ├── prompts/                         # Prompt + conversation parquet datasets
│   ├── embeddings/                      # Embedding vectors for semantic search
│   ├── modelfiles/                      # Saved Modelfile blueprints
│   ├── gguf/                            # GGUF file registry
│   └── lora/                            # LoRA adapter registry
└── claude_code_custom/
    └── claude-code/                     # Upstream reference clone (git-ignored)
        └── plugins/                     # Official plugin examples
```

**Excluded from git** (see `.gitignore`):
- `venv/` — Python virtual environment
- `local_data/` — runtime parquet + embedding files
- `claude_custom_projects_1/` — active work product
- `git_clones/` — third-party clones
- `claude_code_custom/claude-code/` — upstream reference
- `scripts_and_skills/claude_skills/skills/` — Anthropic skills reference clone
- `basic_reference_documentation_library/` — large reference dumps

---

## Slash Commands

Available inside any `claude` session:

| Command | What it does |
|---------|-------------|
| `/prompt-list` | Browse, search, and export the prompt dataset |
| `/set-system` | Change any Ollama model's system prompt interactively |
| `/generate-dataset` | Convert files/dirs into ShareGPT training conversations |
| `/parse-codebase` | Smart codebase exploration with minimal token usage |
| `/web-search` | DuckDuckGo search (useful for local model sessions) |
| `/model-manager` | Manage models, GGUFs, Modelfiles, and LoRA adapters |

---

## Philosophy

The Claude Code CLI is the **engine**. This repo is the **configuration and extension layer**.

```
[Desktop Shortcut]
       │
       ▼
[run_claude.ps1]  ← env setup, model routing, Ollama health check
       │
       ▼
[claude CLI]  ← reads CLAUDE.md, loads skills & plugins automatically
       │
       ▼
[Ollama gpt-oss:20b]  ← local 14 GB model, Titan XP
```

Benefits of this approach:
- **Update claude-code anytime** — extensions are outside the CLI package
- **Skills are just markdown** — no build step, no breaking changes
- **Hooks intercept everything** — pre/post tool use without CLI modification
- **Works offline** — full stack runs locally

---

## Roadmap

- [x] Phase 1 — Git repo, launcher, CLAUDE.md, gitignore
- [x] Phase 2 — Data layer: PromptStore, EmbeddingStore, DatasetGenerator, WebSearch
- [x] Phase 2 — Model manager: OllamaAPI, ModelfileBuilder, GGUFManager, LoRAManager, HFDownload
- [x] Phase 2 — CoreCoder prompt dataset (19 rows) + `corecoder:latest` model in Ollama
- [x] Phase 2 — 6 slash commands + 2 custom skills
- [ ] Phase 3 — Custom plugins (commands + agents for project archetypes)
- [ ] Phase 4 — Hook system (automated code review, safety checks)
- [ ] Phase 5 — MCP server integration

---

## Credits

Built on top of [Claude Code](https://github.com/anthropics/claude-code) by Anthropic.
Skills reference from the [official skills repo](https://github.com/anthropics/claude-code).
All customization in this repo is original work.
