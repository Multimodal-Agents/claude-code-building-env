# blues-skills

> Custom Claude Code skill collection by [Leo Borcherding](https://github.com/LeoBorcherding) /
> [Multimodal-Agents](https://github.com/Multimodal-Agents)

A plugin for [Claude Code](https://github.com/anthropics/claude-code) that adds five skills for
terminal power-users, AI/ML engineers, and voice-interface builders.

---

## Skills

| Skill | Trigger phrase examples | What it does |
|---|---|---|
| `blues-terminal-execution` | "ls", "run command", "git status" | Executes PowerShell / shell commands directly without asking for permission on safe ops |
| `code-parser` | "parse this codebase", "explore the repo" | Surgical codebase exploration — reads the minimum files needed to build a full mental model |
| `prompt-manager` | "save this prompt", "search my datasets", "export for Unsloth" | Manages a local Parquet-based prompt/training-data store with semantic search via Ollama |
| `cytoscape-playground` | "graph playground", "dependency graph", "knowledge graph" | Generates interactive Cytoscape.js graph explorers as single self-contained HTML files |
| `speech-to-speech` | "voice pipeline", "speak", "talk to Claude", "S2S" | Real-time voice conversation: Mic → Whisper STT → Claude → edge-tts → speaker |
| `job-list` | "start job list", "run jobs", "job queue", `/job-list` | Sequential project build queue — runs `*_1.md` → `*_2.md` → ..., gates each job at ≥95% objective completion before advancing |

---

## Installation

### Via local directory (current)

```bash
claude plugin add blues-core@blues-skills \
  --from directory:/path/to/blues_skills
```

### Via GitHub (once published)

```bash
claude plugin add blues-core@blues-skills \
  --from github:Multimodal-Agents/claude-code-building-env
```

---

## Skill Details

### blues-terminal-execution

Guides Claude to execute terminal commands immediately rather than describing them.

- Covers: `ls`, `cd`, file creation, git, npm/pip, process inspection, network checks
- Safety rules built in: destructive ops (delete) always ask first; reads/navigation execute directly
- Optimised for PowerShell on Windows; Unix commands also included

No extra dependencies.

---

### code-parser

A structured codebase-exploration protocol that minimises token waste.

**Protocol:**
1. Single recursive file tree capture
2. Priority read order: config → README → entry point → core module → types
3. Build an internal mental map before touching anything
4. Optionally query a semantic embedding index (requires `nomic-embed-text` via Ollama)

No extra dependencies for basic use.
Semantic search requires: `ollama pull nomic-embed-text`

---

### prompt-manager

Local prompt and conversation database backed by Parquet files.

- Add individual prompt pairs or full ShareGPT conversations
- Keyword search and semantic search (Ollama embeddings)
- Export to Unsloth JSONL for fine-tuning

**Requirements:**

```bash
pip install pandas pyarrow
# For semantic search:
ollama pull nomic-embed-text
```

Data stored at `M:\claude_code_building_env\local_data\prompts\` (configurable via `CLAUDE_DATA_ROOT`).

---

### cytoscape-playground

Produces interactive network/graph visualisations as single HTML files using Cytoscape.js.

Best for: dependency graphs, knowledge graphs, call graphs, DAGs.
Outputs a dark-themed playground with layout selector (dagre/cola/cose/breadthfirst), group
filters, neighbour highlighting, and a Claude prompt panel.

**Modes:**
- **CDN / playground** — zero install, runs immediately in any browser
- **npm project** — `npm install cytoscape cytoscape-dagre dagre`

No server-side dependencies.

---

### speech-to-speech

Real-time voice conversation pipeline.

```
Mic → VAD → faster-whisper STT → Claude → edge-tts → speaker
                                                  ↕ (optional)
                                         whisper-vits-svc voice conversion
```

**Quick start:**

```bash
pip install -r scripts_and_skills/speech/requirements.txt
python -m scripts_and_skills.speech.voice_pipeline
```

**Backends:**

| Flag | LLM backend |
|---|---|
| _(default)_ | Anthropic API (`ANTHROPIC_API_KEY` required) |
| `--ollama` | Local Ollama (no API key) |
| `--claude-code` | Claude Code CLI subprocess (inside Claude Code sessions) |

**Requirements:**
- `faster-whisper` (STT)
- `edge-tts` + `ffmpeg` (TTS)
- `sounddevice`, `webrtcvad` (audio I/O)
- Optional: [whisper-vits-svc](https://github.com/PlayVoice/whisper-vits-svc) for custom voice conversion

Windows note: `pip install webrtcvad-wheels` instead of `webrtcvad` if the build fails.

---

### job-list

Sequential build queue — processes a set of numbered project spec files one by one,
verifying completion before advancing.

**Naming convention:** `<anything>_1.md`, `<anything>_2.md`, ...

**Protocol:**
1. Auto-discover or accept explicit file list; sort numerically by suffix
2. Parse each spec: objectives, stack, output path, success criteria
3. Build, then run verification per objective
4. Score completion (PASS/PARTIAL/FAIL per objective → numeric %)
5. Gate: advance only when score ≥ 95%; defer cosmetic/minor items explicitly
6. Print a completion banner per job and a final run summary

**Unattended mode:** pass `--unattended` to the invocation, or launch the session with
`claude --dangerously-skip-permissions` to run all jobs hands-free.

No extra dependencies.

---

## Project structure

```
blues_skills/
├── .claude-plugin/
│   └── marketplace.json      ← plugin registry
├── skills/
│   ├── blues-terminal-execution/
│   │   └── SKILL.md
│   ├── code-parser/
│   │   └── SKILL.md
│   ├── prompt-manager/
│   │   └── SKILL.md
│   ├── cytoscape-playground/
│   │   ├── SKILL.md
│   │   └── templates/
│   │       └── graph-explorer.md
│   ├── speech-to-speech/
│   │   └── SKILL.md
│   └── job-list/
│       ├── SKILL.md
│       └── references/
│           └── completion-rubric.md
└── README.md
```

---

## Part of claude-code-building-env

These skills live inside the larger
[claude-code-building-env](https://github.com/Multimodal-Agents/claude-code-building-env)
monorepo alongside PowerShell utilities, embedding pipelines, and dataset tooling.

---

## Author

**Leo Borcherding** — [github.com/LeoBorcherding](https://github.com/LeoBorcherding) —
borchborchmail@gmail.com
