# Running /job-list Overnight — Complete Guide

How to queue multiple projects across different directories, launch in full-auto mode,
and wake up to finished builds.

---

## The Core Idea

You write one markdown spec file per project. Each file describes what to build, where to
put it, and what "done" means. You launch Claude in full-auto mode, type one command, and
leave. Claude works through every job in order, scoring each one before moving to the next.

---

## Step 1 — Write Your Job Specs

### Naming convention

Files must end with `_N.md` where N is the order you want them built:

```
my_api_1.md
dashboard_2.md
cli_tool_3.md
```

The name before `_N` can be anything. You can put each file in its own project folder,
or collect them all in one place — both work.

### Spec template

```markdown
# <Project Name>

## Objectives
1. <What must be built — be specific>
2. <Another concrete goal>
3. <Another concrete goal>

## Stack
- Language / runtime
- Key libraries / frameworks
- Any version constraints

## Output
./path/to/output/directory

## Resources
- Any reference links, file paths, or embedded code blocks Claude should use

## Done when
- Entry point runs without error
- All objectives verified (describe how)
- [nice to have] optional polish items that won't block completion
```

**Tips for good specs:**
- Make objectives concrete and testable. "Add an API endpoint" is weak. "POST /api/items returns 201 and persists to SQLite" is strong.
- Put the output path in `Output:` — Claude creates the directory if it doesn't exist.
- Mark optional items `[nice to have]` so Claude defers them instead of blocking on them.
- If a project needs an API key or external service, note it — Claude will PARTIAL the
  objective if it can't reach it rather than hanging forever.

---

## Step 2 — Multi-Project Layout

### Option A — All specs in one folder (simplest)

Put all `_N.md` files in the same directory. Claude auto-discovers them when you run
`/job-list` with no arguments.

```
overnight_run/
├── scraper_1.md        → output: ./projects/scraper/
├── api_server_2.md     → output: ./projects/api_server/
└── dashboard_3.md      → output: ./projects/dashboard/
```

Each spec's `Output:` path points to wherever that project should live.
Claude builds each project in its own output directory.

### Option B — Specs inside project folders (explicit paths)

Each project lives in its own folder with a spec next to it:

```
claude_custom_projects_1/
├── scraper/
│   └── job_1.md
├── api_server/
│   └── job_2.md
└── dashboard/
    └── job_3.md
```

Pass the paths explicitly when you invoke:
```
/job-list scraper/job_1.md api_server/job_2.md dashboard/job_3.md
```

### Option C — Mixed locations, absolute paths

When projects live in completely different locations, use absolute paths:

```
/job-list \
  M:/claude_custom_projects_1/learning_projects/scraper_1.md \
  M:/claude_custom_projects_1/enterprise_projects/api_2.md \
  C:/some_other_place/tool_3.md
```

---

## Step 3 — Launch in Full-Auto Mode

### PowerShell (Windows desktop shortcut)

1. Run `run_claude.ps1`
2. Select **option 7** — `Claude - Anthropic Pro [full auto]`
   - Uses Anthropic Pro API
   - All tool confirmations suppressed
3. At the Claude prompt, type your job-list command (see Step 4)

Or select **option 8** — `Claude - Ollama gpt-oss:20b [full auto]`
- Same but routes through your local model
- Better for long overnight runs (no API cost, no rate limits)
- Requires Ollama running with gpt-oss:20b loaded

### WSL / bash

```bash
bash /mnt/m/claude_code_building_env/scripts_and_skills/claude_scripts/run_claude.sh
# select 7 (Anthropic Pro full auto) or 8 (Ollama full auto)
```

---

## Step 4 — Start the Job Queue

### Auto-discover (specs in current directory)

```
/job-list
```

Claude scans the current directory for `*_N.md` files, sorts them numerically, prints
the queue, and starts building.

### Explicit file list

```
/job-list project_1.md project_2.md project_3.md
```

### Explicit list + unattended flag (belt-and-suspenders)

Adding `--unattended` is redundant if you launched with option 7/8, but it makes the
intent explicit in the transcript:

```
/job-list project_1.md project_2.md --unattended
```

### Absolute paths across different drives/directories

```
/job-list M:/claude_custom_projects_1/learning_projects/scraper_1.md M:/claude_custom_projects_1/enterprise_projects/api_2.md
```

---

## Step 5 — Before You Walk Away

**Quick checklist:**

- [ ] Ollama is running (if using option 8): check the dashboard panel in the launcher
- [ ] All `_N.md` spec files are written and saved
- [ ] Output directories either exist or specs say `Output: ./new_dir` (Claude creates them)
- [ ] Any required env vars are set in the current shell before launching Claude
      (e.g. `$env:OPENAI_API_KEY = "..."` before running `run_claude.ps1`)
- [ ] Your machine won't sleep — disable sleep in Windows power settings for the session
- [ ] You've typed the `/job-list ...` command and Claude has printed the job queue

Once Claude prints the job queue and starts the first job card, it's running. Leave.

---

## Step 6 — Reading the Results in the Morning

Claude prints a **Final Run Summary** when all jobs are done:

```
╔══════════════════════════════════════════════════╗
║   JOB LIST COMPLETE                              ║
╠══════════════════════════════════════════════════╣
║  Job 1: Scraper          98%  ✓                  ║
║  Job 2: API Server      100%  ✓                  ║
║  Job 3: Dashboard     BLOCKED ✗                  ║
╚══════════════════════════════════════════════════╝

Deferred items:
  [Job 1] Dark mode styling — marked [nice to have], skipped
  [Job 2] Rate limiting — requires Redis, not in spec stack

Blocked jobs:
  [Job 3] BLOCKED: spec references ./shared/tokens.json which does not exist
```

**What each status means:**

| Status | Meaning | Action |
|--------|---------|--------|
| `✓` with score | Built and verified | Check the output directory, it's done |
| `✓` with deferred items | Done but some polish skipped | Review deferred list, decide if they matter |
| `BLOCKED` | Claude couldn't proceed | Read the reason, fix the issue, re-run that job alone |

### Re-running a blocked job

```
/job-list M:/path/to/failed_job_3.md
```

Fix whatever was blocking (missing file, missing API key, etc.), then run just that one.

---

## Full Example — Three Projects Overnight

### File structure

```
M:/claude_custom_projects_1/overnight/
├── rss_scraper_1.md
├── rest_api_2.md
└── admin_ui_3.md
```

### rss_scraper_1.md

```markdown
# RSS Feed Scraper

## Objectives
1. Fetch and parse RSS/Atom feeds from a list of URLs in feeds.txt
2. Store articles (title, link, published_date, summary) in SQLite at ./data/articles.db
3. CLI: `python main.py --fetch` runs a full scrape, `python main.py --list` prints recent articles

## Stack
- Python 3.11+
- feedparser, sqlite3 (stdlib)

## Output
M:/claude_custom_projects_1/overnight/rss_scraper/

## Done when
- `python main.py --fetch` runs without error and populates the database
- `python main.py --list` prints at least the article titles
- [nice to have] deduplication by link
```

### rest_api_2.md

```markdown
# REST API for Article Database

## Objectives
1. FastAPI app that reads from the SQLite DB built in job 1
2. GET /articles — paginated list (default 20 per page)
3. GET /articles/{id} — single article
4. POST /articles/search — full-text search over title and summary

## Stack
- Python 3.11+, FastAPI, uvicorn, sqlite3

## Output
M:/claude_custom_projects_1/overnight/rest_api/

## Resources
- DB path: M:/claude_custom_projects_1/overnight/rss_scraper/data/articles.db

## Done when
- `uvicorn main:app` starts without error
- All three endpoints respond correctly (verify with httpx or curl)
```

### admin_ui_3.md

```markdown
# Admin Dashboard

## Objectives
1. Single-file HTML dashboard (no build step) that hits the REST API from job 2
2. Table view of articles with pagination controls
3. Search box that calls POST /articles/search and updates the table live

## Stack
- Vanilla HTML + CSS + JS (no framework)
- Fetches from http://localhost:8000

## Output
M:/claude_custom_projects_1/overnight/admin_ui/

## Done when
- Opening index.html in a browser shows the article table
- Search box filters results via the API
```

### Launch sequence

1. Start `run_claude.ps1` → select **7** (Anthropic Pro full auto)
2. At the prompt:

```
/job-list M:/claude_custom_projects_1/overnight/rss_scraper_1.md M:/claude_custom_projects_1/overnight/rest_api_2.md M:/claude_custom_projects_1/overnight/admin_ui_3.md
```

3. Claude prints the queue, starts Job 1. Walk away.

---

## Tips for Reliable Overnight Runs

**Write specs that don't need decisions.** If the spec is ambiguous Claude will make a
reasonable choice and document it — but the more explicit you are, the closer the result
is to what you actually wanted.

**Local model (option 8) for cost-free long runs.** Anthropic Pro has usage limits that
can slow down a long overnight queue. gpt-oss:20b on Ollama has no API cost and no rate
limits — ideal for multi-hour autonomous builds.

**Chain jobs that depend on each other.** Job 2 in the example above uses the DB from
Job 1. This works because Claude processes jobs sequentially and the output of Job 1
exists on disk by the time Job 2 starts. Just make sure the paths in later specs are
absolute and correct.

**Mark optional features `[nice to have]`.** Without this tag, Claude will keep working
to reach 95% and won't advance if something minor is failing. With it, Claude explicitly
defers the item and moves on.

**Don't put API keys in spec files.** Set env vars before launching, or put a note in
the spec like `Uses OPENAI_API_KEY from environment` — Claude will warn in the summary
if the key wasn't found rather than silently fail.
