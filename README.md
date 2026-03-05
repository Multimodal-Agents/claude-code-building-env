# claude-code-building-environment

> Personal AI development abstraction layer built on top of [Claude Code CLI](https://github.com/anthropics/claude-code).

Extend Claude Code with your own skills, plugins, slash commands, and hooks —
all outside the CLI package so they survive every upstream update.
Works with **Anthropic Pro** (cloud) or a **local Ollama model** (offline).

---

## Prerequisites

| Requirement | How to check | Required? |
|-------------|-------------|-----------|
| **Claude Code CLI** | `claude --version` | **Yes** |
| Python 3.10+ | `python --version` | Yes |
| Git | `git --version` | Yes |
| Ollama | `ollama list` | Only for local models |

### Installing Claude Code (if you don't have it yet)

```bash
npm install -g @anthropic-ai/claude-code
```

> Need Node.js? Get it from [nodejs.org](https://nodejs.org/).
> Full install docs: [docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code/overview)

---

## Already have Claude Code installed?

If you already have `claude` working, you can start using this repo's extensions immediately:

```powershell
git clone https://github.com/Multimodal-Agents/claude-code-building-env.git
cd claude-code-building-env
.\setup.ps1            # clones Anthropic reference repos (skills + claude-code examples)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas pyarrow numpy requests ddgs huggingface_hub
```

Then just launch Claude from this directory:
```powershell
claude                 # uses Anthropic Pro — CLAUDE.md, skills, and slash commands load automatically
```

That's it. Claude Code reads `CLAUDE.md` and discovers the skills/commands in this repo on startup. No Ollama required.

**Want the interactive launcher instead?** (lets you pick Anthropic Pro, Ollama, or Speech mode)
```powershell
.\scripts_and_skills\claude_scripts\run_claude.ps1
```

---

## Full Setup (new users or local Ollama models)

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
├── CLAUDE.md                            # Global context injected at every session
├── QUICKSTART.md                        # Step-by-step setup guide
├── TESTING.md                           # Test procedures
├── setup.ps1                            # One-time repo setup
├── setup_llamacpp.ps1                   # Optional: llama.cpp build + Python tools
├── show_dataset.py                      # Quick dataset viewer
├── assets/                              # Icons, static media, video scripts
├── .claude/
│   └── commands/                        # 11 slash commands (see table below)
├── scripts_and_skills/
│   ├── claude_scripts/
│   │   ├── run_claude.ps1               # Windows launcher (interactive menu)
│   │   └── run_claude.sh               # WSL/Linux launcher
│   ├── data/                            # Data layer
│   │   ├── prompt_store.py              # Parquet-backed prompt + conversation CRUD
│   │   ├── embeddings.py                # nomic-embed-text semantic search
│   │   ├── dataset_generator.py         # Docs/code → ShareGPT training conversations
│   │   ├── web_search.py                # DuckDuckGo search wrapper
│   │   ├── arxiv_crawler.py             # ArXiv Atom API search + paper text fetching
│   │   ├── classifier.py               # Content moderation via granite3-guardian
│   │   ├── science_validator.py         # Scientific claim validation
│   │   └── seeds/
│   │       └── corecoder_prompts.py     # Seeds 19-row CoreCoder prompt dataset
│   ├── model_manager/                   # Ollama model lifecycle
│   │   ├── ollama_api.py                # REST wrapper: list/create/delete/pull/embed
│   │   ├── modelfile.py                 # Modelfile builder (system prompt, params, adapters)
│   │   ├── gguf_manager.py              # GGUF registry + HF→GGUF conversion + quantize
│   │   ├── lora_manager.py              # LoRA adapter registry + Ollama deployment
│   │   └── hf_download.py              # Download GGUFs from HuggingFace Hub
│   ├── speech/                          # Voice pipeline
│   │   ├── voice_pipeline.py            # Real-time speech-to-speech loop
│   │   ├── stt.py                       # Whisper speech-to-text
│   │   ├── tts.py                       # edge-tts text-to-speech
│   │   ├── script_narrator.py           # Script → narrated audio
│   │   └── audio_utils.py              # Audio helpers
│   ├── code_graph/                      # Live code graph visualization
│   │   ├── server.py                    # FastAPI graph server
│   │   ├── code_parser.py              # Import/dependency parser
│   │   ├── hook_notify.py              # Claude hook listener
│   │   ├── index.html                  # Cytoscape.js terminal UI
│   │   ├── start.ps1                   # Windows launcher
│   │   └── start.sh                    # WSL launcher
│   └── claude_skills/
│       ├── skills/                      # Official Anthropic skills (reference, git-ignored)
│       ├── blues_skills/                # 6 custom domain skills (see table below)
│       └── template_skills/             # Scaffold for new skills
├── agent-chef-claude-code/              # Dataset generation agent
│   ├── ragchef.py                       # RAG-powered conversation chef
│   ├── conversation_generator.py        # Conversation synthesis
│   ├── crawlers_module.py               # Web/doc crawlers
│   ├── dataset_expander.py             # Dataset augmentation
│   └── classification.py               # Quality classification
├── dataset_kitchen/                     # Test datasets and samples
├── local_data/                          # Runtime data (git-ignored, machine-local)
│   ├── prompts/                         # Prompt + conversation parquet datasets
│   ├── embeddings/                      # Embedding vectors for semantic search
│   ├── exports/                         # Exported datasets
│   └── modelfiles/                      # Saved Modelfile blueprints
├── claude_custom_projects_1/            # Active projects (git-ignored)
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

## Plugins

This repo uses the official [Playground plugin](https://claude.com/plugins/playground) for interactive HTML/JS prototyping. The **cytoscape-playground** skill (below) extends it with Cytoscape.js graph visualization.

Reference plugins from the upstream `claude-code` repo are cloned into `claude_code_custom/claude-code/plugins/` by `setup.ps1` for study — they include examples like `code-review`, `feature-dev`, `hookify`, `pr-review-toolkit`, and more.

---

## Skills

| Skill | Path | What it does |
|-------|------|-------------|
| **prompt-manager** | `blues_skills/skills/prompt-manager/` | Store/search/export prompts and conversations via PromptStore + EmbeddingStore |
| **code-parser** | `blues_skills/skills/code-parser/` | Surgical codebase reading protocol — single tree scan, mental map, targeted reads only |
| **speech-to-speech** | `blues_skills/skills/speech-to-speech/` | Real-time voice pipeline: Mic → VAD → Whisper → LLM → edge-tts → speaker |
| **cytoscape-playground** | `blues_skills/skills/cytoscape-playground/` | Extends Playground plugin with Cytoscape.js graph/network visualization |
| **job-list** | `blues_skills/skills/job-list/` | Sequential project build queue — processes `*_1.md` → `*_2.md` in order, gates on ≥95% completion |
| ~~blues-terminal-execution~~ | `blues_skills/skills/blues-terminal-execution/` | Safe terminal patterns *(disabled — larger models handle this natively)* |

All skills live under `scripts_and_skills/claude_skills/blues_skills/skills/`.

---

## Slash Commands

Available in any `claude` session launched from this repo:

| Command | What it does |
|---------|-------------|
| `/prompt-list` | Browse, search, and export the prompt dataset |
| `/set-system` | Change any Ollama model's system prompt interactively |
| `/generate-dataset` | Convert files/dirs into ShareGPT training conversations |
| `/show-dataset` | View rows from any saved dataset (zero-token, raw output) |
| `/parse-codebase` | Smart codebase exploration with minimal token usage |
| `/web-search` | DuckDuckGo search (useful for local model sessions) |
| `/model-manager` | Manage models, GGUFs, Modelfiles, and LoRA adapters |
| `/script-to-audio` | Parse a video script, strip director notes, render to .wav via edge-tts |
| `/speech` | Start real-time voice-to-voice mode (say "goodbye claude" to exit) |
| `/code-graph` | Live terminal CRT code graph — parse imports, watch edits, animate changes |
| `/job-list` | Run a sequential queue of project builds from numbered markdown specs |

---

## Philosophy

The Claude Code CLI is the **engine**. This repo is the **configuration and extension layer**.

```
[Desktop Shortcut / Terminal]
       │
       ▼
[run_claude.ps1 / run_claude.sh]  ← interactive menu: Anthropic Pro, Ollama, or Speech
       │
       ├─── Anthropic Pro ──► claude          (cloud, no local model needed)
       │
       └─── Ollama ────────► claude --model gpt-oss:20b   (local, offline)
                                │
                                ▼
                         [claude CLI]  ← reads CLAUDE.md, loads skills & commands automatically
```

Benefits of this approach:
- **Update claude-code anytime** — extensions are outside the CLI package
- **Skills are just markdown** — no build step, no breaking changes
- **Hooks intercept everything** — pre/post tool use without CLI modification
- **Works with cloud or local** — Anthropic Pro for power, Ollama for offline/free
- **Playground plugin** — interactive HTML prototyping built in

---

## Roadmap

- [x] Phase 1 — Git repo, launcher, CLAUDE.md, gitignore
- [x] Phase 2 — Data layer: PromptStore, EmbeddingStore, DatasetGenerator, WebSearch, ArXiv, Classifier
- [x] Phase 2 — Model manager: OllamaAPI, ModelfileBuilder, GGUFManager, LoRAManager, HFDownload
- [x] Phase 2 — CoreCoder prompt dataset (19 rows) + `corecoder:latest` model
- [x] Phase 2 — 11 slash commands + 6 custom skills
- [x] Phase 2 — Speech pipeline (Whisper STT → LLM → edge-tts)
- [x] Phase 2 — Code graph server (FastAPI + Cytoscape.js)
- [x] Phase 2 — Agent-chef dataset generation pipeline
- [x] Phase 2 — Playground plugin integration
- [ ] Phase 3 — Custom plugins (commands + agents for project archetypes)
- [ ] Phase 4 — Hook system (automated code review, safety checks)
- [ ] Phase 5 — MCP server integration

---

## Credits

Built on top of [Claude Code](https://github.com/anthropics/claude-code) by Anthropic.
Skills reference from the [official skills repo](https://github.com/anthropics/skills).
All customization in this repo is original work.
