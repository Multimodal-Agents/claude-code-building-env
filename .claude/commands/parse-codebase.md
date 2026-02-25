# /parse-codebase — Intelligent file tree parser for coding tasks

Build a structured understanding of a codebase with minimal token usage.
Instead of repeatedly running `ls` and reading files one by one, this command
builds a complete mental map in one pass so you can code efficiently.

## Instructions

When the user asks you to understand, explore, or work with a codebase:

### Step 1 — Single tree scan
Run ONE command to get the full structure:
```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.Extension -match '\.(py|ts|js|rs|go|java|cpp|c|cs|md|json|toml|yaml|yml|txt)$' } | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
```

Or for Unix:
```bash
find . -type f \( -name "*.py" -o -name "*.ts" -o -name "*.js" -o -name "*.md" -o -name "*.json" \) | head -100
```

### Step 2 — Identify entry points
From the file list, identify (in priority order):
1. `main.py`, `index.ts`, `app.py`, `server.py`, `__main__.py`, `Program.cs`
2. `package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod` (for dependencies)
3. `README.md`, `CLAUDE.md` (for intent/context)
4. Any file matching the user's stated task

Read these files FIRST before any others.

### Step 3 — Build mental map
After reading entry points, construct your understanding:
```
<project-name>
├── Purpose: [what this project does]
├── Entry: [main entry point]
├── Key modules: [list the important ones]
├── Dependencies: [from lock/config files]
└── Task relevance: [which files matter for the user's ask]
```

### Step 4 — Targeted reads only
Only read additional files if they are directly relevant to the task.
Never read a file solely out of curiosity. If uncertain, ask the user.

### Step 5 — Code with context
Now proceed with the coding task, referencing your mental map.
When you need to look up a specific function or class, search within the
already-read content before opening more files.

## Anti-patterns to avoid

- ❌ Running `ls` multiple times to explore subdirectories one by one
- ❌ Reading every file before starting work
- ❌ Re-reading files you've already processed
- ❌ Using `cat` on large files when you only need a specific function

## Semantic search integration

If the prompt database has embeddings for this project's files, search them:
```python
from scripts_and_skills.data.embeddings import EmbeddingStore
store = EmbeddingStore()
results = store.search_all("your function/concept query", top_k=5)
```
This surfaces the most relevant code chunks without reading every file.
