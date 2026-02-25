# CLAUDE.md — Global Context for This Workspace

This file is automatically read by Claude Code at session start.
It describes the structure and intent of this local AI development environment.

---

## What This Repo Is

This is a **personal AI development abstraction layer** built on top of the Claude Code CLI.
The goal is to stay perpetually compatible with future Claude Code updates while keeping all
customization owned and versioned here — plugins, skills, hooks, launchers, and docs.

The engine (Claude Code CLI) is treated as an upgradeable dependency. This repo is the code.

---

## Local Model Stack

- **Runtime**: Ollama (`http://localhost:11434`)
- **Primary model**: `gpt-oss:20b`  — 14 GB, runs on Titan XP, fast local inference
- **Entry point**: `scripts_and_skills/claude_scripts/run_claude.ps1` (desktop shortcut)

---

## Directory Map

```
claude_code_building_env/
├── assets/                         # Icons and static media
├── scripts_and_skills/
│   ├── claude_scripts/
│   │   └── run_claude.ps1          # ← Main launcher / entry point
│   ├── data/                       # Data layer (parquet DB, embeddings, generators)
│   │   ├── prompt_store.py         # CRUD for prompts + conversations (Unsloth/ShareGPT)
│   │   ├── embeddings.py           # nomic-embed-text via Ollama + cosine search
│   │   ├── dataset_generator.py    # Lite agent-chef: docs → training conversations
│   │   ├── web_search.py           # DuckDuckGo search wrapper
│   │   └── seeds/                  # One-time dataset seed scripts
│   └── model_manager/              # Ollama model + GGUF + LoRA lifecycle
│       ├── ollama_api.py           # REST wrapper: list/create/delete/pull/embed
│       ├── modelfile.py            # Modelfile builder (system prompt, params, adapters)
│       ├── gguf_manager.py         # GGUF registry + HF→GGUF conversion + quantize
│       ├── lora_manager.py         # LoRA adapter registry + Ollama deployment
│       └── hf_download.py          # Download GGUFs from HuggingFace Hub
│   └── claude_skills/
│       ├── skills/                 # Anthropic official skills reference (git-ignored)
│       ├── blues_skills/           # Custom domain-specific skills
│       └── template_skills/        # Skill scaffolding templates
├── .claude/
│   └── commands/                   # Slash commands: /prompt-list /generate-dataset /show-dataset /parse-codebase /web-search
├── local_data/                     # Runtime parquet files (git-ignored, machine-local)
│   ├── prompts/                    # Prompt + conversation datasets
│   ├── embeddings/                 # Embedding vectors for semantic search
│   ├── modelfiles/                 # Saved Modelfile blueprints
│   ├── gguf/                       # GGUF file registry (registry.parquet)
│   └── lora/                       # LoRA adapter registry (registry.parquet)
├── claude_code_custom/
│   └── claude-code/                # Upstream Anthropic repo (reference, ignored by git)
│       └── plugins/                # Official plugin examples & patterns
└── claude_custom_projects_1/       # Active projects (excluded from git)
    ├── learning_projects/
    ├── enterprise_projects/
    └── personal_projects/
```

---

## Architecture: Abstraction Layer Philosophy

Rather than forking claude-code and fighting merges forever, we extend it through its
official extension surface:

| Extension Point | Location | Purpose |
|----------------|----------|---------|
| **Skills**     | `scripts_and_skills/claude_skills/` | Teach Claude repeatable workflows |
| **Plugins**    | `claude_code_custom/claude-code/plugins/` | Commands, agents, hooks |
| **Hooks**      | Plugin `hooks.json` files | Pre/post tool intercepts |
| **CLAUDE.md**  | Here + per-project | Session-level context injection |
| **Launcher**   | `run_claude.ps1` | Environment setup, model routing |
| **Data Layer** | `scripts_and_skills/data/` | Parquet prompt DB, embeddings, dataset generator, web search |
| **Commands**   | `.claude/commands/` | Slash commands usable in any session |

This layer works with **any future version** of the Claude Code CLI.

---

## Active Skills

| Skill | Path | What it does |
|-------|------|-------------|
| <!-- blues-terminal-execution | `scripts_and_skills/claude_skills/blues_skills/skills/blues-terminal-execution/` | Safe terminal command patterns (disabled — gpt-oss:20b handles this natively; re-enable if small models are added) --> |
| **prompt-manager** | `scripts_and_skills/claude_skills/blues_skills/skills/prompt-manager/` | Store/search/export prompts and conversations via PromptStore + EmbeddingStore |
| **code-parser** | `scripts_and_skills/claude_skills/blues_skills/skills/code-parser/` | Surgical codebase reading protocol — single tree scan, mental map, targeted reads only |

---

## Slash Commands

| Command | File | What it does |
|---------|------|-------------|
| `/prompt-list` | `.claude/commands/prompt-list.md` | Browse, search, and export prompt datasets |
| `/generate-dataset` | `.claude/commands/generate-dataset.md` | Convert files/dirs into ShareGPT training conversations |
| `/parse-codebase` | `.claude/commands/parse-codebase.md` | Intelligent codebase understanding with minimal token usage |
| `/web-search` | `.claude/commands/web-search.md` | DuckDuckGo search for local model sessions |
| `/model-manager` | `.claude/commands/model-manager.md` | Manage Ollama models, Modelfiles, GGUFs, and LoRA adapters |
| `/set-system` | `.claude/commands/set-system.md` | Interactively change a model's system prompt from a prompt set or free text |
| `/show-dataset` | `.claude/commands/show-dataset.md` | View rows from any saved dataset — runs directly, zero-token overhead |

---

## Key Rules for This Session

1. Default model is `gpt-oss:20b` via Ollama unless specified otherwise.
2. New skills go in `scripts_and_skills/claude_skills/` under an appropriate sub-collection.
3. New plugins go in `claude_code_custom/claude-code/plugins/` for now; may move to own dir later.
4. Project files are never committed to this repo — they live in `claude_custom_projects_1/`.
5. The `claude_code_custom/claude-code/` directory is a reference clone — treat as read-only.
6. Static media / icons live in `assets/`.
