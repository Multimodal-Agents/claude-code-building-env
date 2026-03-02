---
name: job-list
description: Execute a sequential queue of project build specifications from numbered markdown files (*_1.md, *_2.md, ...). Triggers when the user mentions "job list", "job run", "job queue", "build jobs", "start jobs", or references files ending in _N.md. Processes each spec fully and verifies major objectives before advancing — only moves to the next job when ≥95% of major objectives are met.
---

# Job List Skill

You are a methodical, autonomous software builder. When running a job list you work through
project specifications one by one — completely and carefully — never advancing until the
current project is verified done.

---

## Overview

A **job list** is a set of numbered markdown files, each describing a complete project to
build. They are ordered by a numeric suffix:

```
<anything>_1.md   →  job 1
<anything>_2.md   →  job 2
<anything>_3.md   →  job 3
...
```

You execute them in ascending numeric order, gate-checking completion before each transition.

---

## Phase 0 — Discovery

### If files were passed explicitly
Use them in the order given.

### If no files were given — auto-discover

```bash
# Unix / WSL
ls *_[0-9]*.md 2>/dev/null | sort -t_ -k2 -V

# PowerShell (Windows)
Get-ChildItem -Filter "*_*.md" |
  Where-Object { $_.BaseName -match "_\d+$" } |
  Sort-Object { [int]($_.BaseName -replace '.*_(\d+)$','$1') } |
  Select-Object Name
```

Print the discovered queue before starting:

```
Job queue (N jobs):
  [1] my_api_project_1.md
  [2] my_api_project_2.md
  ...
```

---

## Phase 1 — Spec Parsing

Read each job file and extract:

| Field              | Where to look                                              |
|--------------------|-------------------------------------------------------------|
| **Project name**   | H1 heading or `Project:` field                             |
| **Output path**    | `Output:`, `Directory:`, or derive from project name       |
| **Tech stack**     | `Stack:`, `Dependencies:`, or infer from context           |
| **Major objectives** | Numbered/bulleted goal list, `Objectives:` section, or H2s |
| **Success criteria** | `Done when:`, `Tests:`, `Acceptance criteria:` — or derive |
| **Resources**      | Links, embedded code snippets, referenced files            |
| **Unattended flag**| `--unattended` or `--dangerously-skip-permissions` anywhere in the spec or invocation |

Print a **job card** before any work begins:

```
╔══════════════════════════════════════════════════╗
║  Job 1 of N: <project name>                      ║
╠══════════════════════════════════════════════════╣
║  Output: <path>                                  ║
║  Stack:  <tech>                                  ║
║  Objectives (M total):                           ║
║    1. <objective>                                ║
║    2. <objective>                                ║
║  Success criteria:                               ║
║    - <criterion>                                 ║
╚══════════════════════════════════════════════════╝
```

---

## Phase 2 — Build

### 2a. Setup
- Create the output directory if it does not exist
- Install dependencies (`npm install`, `pip install -r requirements.txt`, etc.)
- Note required environment variables or config; warn if missing

### 2b. Implement
- Build the project feature-by-feature following the spec
- Use the **code-parser** skill protocol when entering an existing codebase
- Choose an architecture early and commit to it; avoid mid-stream pivots
- When the spec is ambiguous, choose the most reasonable interpretation and note it

### 2c. Verify
After each major objective is implemented, run the appropriate check immediately —
do not wait until the end to discover failures.

Derived verification (when no explicit tests are given):

| Check           | Command                                  |
|-----------------|------------------------------------------|
| Syntax / import | `python -c "import <module>"` or `node -e "require('./...')"` |
| Entry point     | Run the main entry point with no args    |
| Functional      | Exercise each stated feature             |
| Smoke           | Hit the key path end-to-end              |

Capture the result of every check (PASS / FAIL / PARTIAL).

---

## Phase 3 — Completion Scoring

After all build and verify work for a job is done, score against the objectives:

```
Objective 1: <description>    → PASS
Objective 2: <description>    → PARTIAL  (core works; edge case X not handled)
Objective 3: <description>    → FAIL     (not implemented)
──────────────────────────────────────────────────────
Score: (PASS + 0.5×PARTIAL) / total × 100  =  XX%
```

**Scoring rules:**
- `PASS`    = fully implemented and verified               → 1.0 point
- `PARTIAL` = core logic works, polish or edge case missing → 0.5 points
- `FAIL`    = not implemented or broken                    → 0.0 points

**Advancement gate: score ≥ 95%**

- Score ≥ 95 → advance to next job
- Score < 95 → keep working; do NOT advance

The 5% tolerance is reserved for: cosmetic polish, unlikely edge cases, items that
require external infrastructure (cloud accounts, API keys not provided) — document
these deferred items explicitly.

Consult `references/completion-rubric.md` for edge-case scoring guidance.

---

## Phase 4 — Job Transition

When a job passes the gate, print a **completion banner**:

```
══════════════════════════════════════════════════════
  Job 1: <name>   COMPLETE  (XX%)
  Deferred (minor):
    - <deferred item 1>
    - <deferred item 2>
══════════════════════════════════════════════════════
```

Then immediately start Phase 1 for Job N+1, or print the Final Summary if done.

---

## Final Run Summary

When all jobs are complete:

```
╔══════════════════════════════════════════════════╗
║   JOB LIST COMPLETE                              ║
╠══════════════════════════════════════════════════╣
║  Job 1: <name>         XX%  ✓                    ║
║  Job 2: <name>         XX%  ✓                    ║
║  Job 3: <name>      BLOCKED  ✗                   ║
╚══════════════════════════════════════════════════╝
Deferred items:
  [Job 1] <item>
  [Job 2] <item>
Blocked jobs:
  [Job 3] <reason>
```

---

## Unattended Mode

When the user wants to walk away while all jobs run without confirmation prompts.

### Enabling before the session (recommended)

```bash
# Bash / WSL
claude --dangerously-skip-permissions

# PowerShell launcher
.\scripts_and_skills\claude_scripts\run_claude.ps1 -DangerouslySkipPermissions
```

### Enabling via the invocation

If the user includes `--unattended` or `--dangerously-skip-permissions` anywhere in
the job-list request (or in any job file's top-level flags), acknowledge it once
and proceed with all tool calls without pausing for human approval:

```
⚠  UNATTENDED MODE  — all tool calls will execute without confirmation.
   Jobs will run autonomously until complete or a hard error is hit.
   Re-run with confirmations enabled to review any blocked step.
```

Do not ask clarifying questions during unattended runs. If genuinely blocked
(missing dependency, inaccessible resource), log it and skip to the next job.

---

## When Stuck

If a job is blocked (missing dependency, broken resource, spec contradiction):

1. Log clearly:  `BLOCKED: <reason>`
2. Mark the job `BLOCKED` in the queue summary
3. Move on to Job N+1 if one exists — do NOT loop indefinitely
4. Report all blocked jobs in the Final Summary with their reasons

---

## Efficiency Rules

| Rule | Reason |
|------|--------|
| Parse the full spec before writing any code | Avoids mid-build pivots |
| Run verification per objective, not only at the end | Catch failures early |
| Defer cosmetic items explicitly in scoring | Prevents perfectionism blocking completion |
| Never re-read a file already in context | Token waste |
| When ambiguous, decide and document — don't pause to ask (unattended) | Keeps the run moving |
