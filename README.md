# claude-local-layer

> Personal AI development abstraction layer built on top of [Claude Code CLI](https://github.com/anthropics/claude-code).

Run a local LLM (Ollama `gpt-oss:20b` on Titan XP) through Claude Code's full agentic tooling —
with your own skills, plugins, and hooks that survive every upstream update.

---

## Quick Start

Double-click the desktop shortcut **or** run directly:

```powershell
powershell.exe -ExecutionPolicy Bypass -File "M:\claude_code_building_env\scripts_and_skills\claude_scripts\run_claude.ps1"
```

| Flag | Default | Description |
|------|---------|-------------|
| `-Model` | `gpt-oss:20b` | Any Ollama model or `claude-*` for API |
| `-Project` | *(workspace root)* | Jump straight into a project sub-folder |
| `-ListModels` | — | Print known models and exit |
| `-Debug` | — | Pass `--debug` to claude CLI |

### Examples

```powershell
# Default (local model, workspace root)
.\run_claude.ps1

# Specific project
.\run_claude.ps1 -Project learning_projects\claude_maple_first_database

# Different model
.\run_claude.ps1 -Model "gpt-oss:7b"

# Check what models are configured
.\run_claude.ps1 -ListModels
```

---

## Repository Structure

```
.
├── assets/                              # Icons and static media
├── CLAUDE.md                            # Global context injected at every session
├── scripts_and_skills/
│   ├── claude_scripts/
│   │   └── run_claude.ps1               # Main entry point
│   └── claude_skills/
│       ├── skills/                      # Official Anthropic skills (reference, git-ignored)
│       ├── blues_skills/                # Custom domain skills
│       └── template_skills/             # Scaffold for new skills
└── claude_code_custom/
    └── claude-code/                     # Upstream reference clone (git-ignored)
        └── plugins/                     # Official plugin examples
```

**Excluded from git** (see `.gitignore`):
- `claude_custom_projects_1/` — active work product
- `git_clones/` — third-party clones
- `claude_code_custom/claude-code/` — upstream reference
- `scripts_and_skills/claude_skills/skills/` — Anthropic skills reference clone
- `basic_reference_documentation_library/` — large reference dumps

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
- [ ] Phase 2 — Custom skills library (domain-specific workflows)
- [ ] Phase 3 — Custom plugins (commands + agents for project archetypes)
- [ ] Phase 4 — Hook system (automated code review, safety checks)
- [ ] Phase 5 — MCP server integration

---

## Credits

Built on top of [Claude Code](https://github.com/anthropics/claude-code) by Anthropic.
Skills reference from the [official skills repo](https://github.com/anthropics/claude-code).
All customization in this repo is original work.
