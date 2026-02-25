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
- **Entry point**: `powershell_scripts/claude_scripts/run_claude.ps1` (desktop shortcut)

---

## Directory Map

```
claude_code_building_env/
├── powershell_scripts/
│   ├── claude_scripts/
│   │   └── run_claude.ps1          # ← Main launcher / entry point
│   └── claude_skills/
│       ├── skills/                 # Anthropic official skills reference
│       ├── blues_skills/           # Custom domain-specific skills
│       └── template_skills/        # Skill scaffolding templates
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
| **Skills**     | `powershell_scripts/claude_skills/` | Teach Claude repeatable workflows |
| **Plugins**    | `claude_code_custom/claude-code/plugins/` | Commands, agents, hooks |
| **Hooks**      | Plugin `hooks.json` files | Pre/post tool intercepts |
| **CLAUDE.md**  | Here + per-project | Session-level context injection |
| **Launcher**   | `run_claude.ps1` | Environment setup, model routing |

This layer works with **any future version** of the Claude Code CLI.

---

## Active Skills

| Skill | Path | What it does |
|-------|------|-------------|
| blues-terminal-execution | `claude_skills/blues_skills/skills/blues-terminal-execution/` | Safe terminal command patterns |
| (add more as built) | | |

---

## Key Rules for This Session

1. Default model is `gpt-oss:20b` via Ollama unless specified otherwise.
2. New skills go in `powershell_scripts/claude_skills/` under an appropriate sub-collection.
3. New plugins go in `claude_code_custom/claude-code/plugins/` for now; may move to own dir later.
4. Project files are never committed to this repo — they live in `claude_custom_projects_1/`.
5. The `claude_code_custom/claude-code/` directory is a reference clone — treat as read-only.
