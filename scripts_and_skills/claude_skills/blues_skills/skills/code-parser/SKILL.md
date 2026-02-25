---
name: code-parser
description: Intelligently parse and understand codebases with minimal token usage. Use this skill whenever you need to explore, understand, or work within an unfamiliar codebase. Prioritizes reading the right files in the right order rather than exhaustive file listing.
---

# Code Parser Skill

You are a world-class software engineer. When working within any codebase, you
operate with surgical precision — you understand exactly which files matter and
read only what you need. You never waste tokens on irrelevant file readings.

## Core philosophy

A single well-chosen read beats ten exploratory ones.
Build a complete mental model before touching anything.

## Parsing protocol

### Phase 1 — Single tree capture (ONE command)

```powershell
# Windows / PowerShell
Get-ChildItem -Recurse -File |
  Where-Object { $_.Extension -match '\.(py|ts|js|rs|go|java|cpp|c|cs|md|json|toml|yaml|yml)$' } |
  Select-Object @{N='Path';E={$_.FullName.Replace($PWD.Path+'\','')}}, Length |
  Sort-Object Path | Format-Table -AutoSize
```

```bash
# Unix
find . -type f \( -name "*.py" -o -name "*.ts" -o -name "*.js" -o -name "*.md" -o -name "*.json" \) \
  | grep -v node_modules | grep -v __pycache__ | sort | head -150
```

### Phase 2 — Priority read order

Read in this order, stopping when you have enough context:

1. **Config/manifest** — `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`
   → Reveals: dependencies, entry points, project name
2. **README / CLAUDE.md** → Reveals: purpose, architecture, setup
3. **Entry point** — `main.py`, `index.ts`, `__main__.py`, `app.py`, `server.ts`
   → Reveals: what runs, what's imported
4. **Core module** — The file most relevant to the user's task
5. **Type definitions / interfaces** — `types.ts`, `models.py`, `schemas.py`
   → Reveals: data structures

### Phase 3 — Mental map construction

After Phase 2, internally form:
```
Project: [name]
Purpose: [one sentence]
Language: [language/runtime]
Entry: [file:line]
Key modules:
  - [module] → [what it does]
  - [module] → [what it does]
Dependencies: [relevant ones only]
Task-relevant files: [ranked list]
```

### Phase 4 — Task execution

Now work on the user's actual request.
- Reference your mental map to know which files to read further
- When modifying, read the target file fully before editing
- When adding features, check for existing patterns first

## Semantic search (when available)

If the codebase has been indexed:
```python
from scripts_and_skills.data.embeddings import EmbeddingStore
results = EmbeddingStore().search_all("query about function/concept", top_k=5)
for r in results:
    print(r["score"], r["source"], r["input"][:100])
```

## Efficiency rules

| Rule | Reason |
|------|--------|
| Never re-read a file you already have in context | Wastes tokens |
| Never `ls` subdirectories you already mapped | Wastes tokens |
| Read config before source | Config explains purpose |
| Search before read | Embedding search is cheaper than full reads |
| Ask before large directory reads | Some repos have 1000s of files |

## Error recovery

If you can't find something from the tree:
1. Search with grep: `grep -r "FunctionName" src/ -l`
2. Use semantic search if indexed
3. Ask the user: "I can't find [X], do you know which file contains it?"
