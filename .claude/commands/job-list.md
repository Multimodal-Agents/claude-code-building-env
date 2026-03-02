# /job-list — Run a sequential queue of project build jobs from numbered markdown specs

Executes a series of project specification files (`*_1.md`, `*_2.md`, ...) in order.
Each spec describes a complete project to build. Claude builds, tests, and scores each
project before advancing — only moving on when ≥ 95% of major objectives are verified.

---

## Usage

```
/job-list [file ...] [--unattended]
```

### Examples

```bash
# Auto-discover all *_N.md files in the current directory
/job-list

# Explicit file list (processed in this order)
/job-list backend_api_1.md frontend_app_2.md cli_tool_3.md

# Walk away — no confirmation prompts (requires --dangerously-skip-permissions at launch)
/job-list --unattended

# Explicit files + unattended
/job-list project_1.md project_2.md --unattended
```

---

## What Claude does

1. **Discovers** job files (auto or explicit) and prints the ordered queue
2. **Parses** each spec: project name, output path, tech stack, objectives, success criteria
3. **Builds** the project fully, running verification after each major objective
4. **Scores** completion: PASS / PARTIAL / FAIL per objective → numeric percentage
5. **Gates**: only advances when score ≥ 95%; keeps working otherwise
6. **Transitions** to the next job with a completion banner
7. **Summarises** the full run at the end

---

## Job Spec Format

Each `*_N.md` file should contain at minimum:

```markdown
# <Project Name>

## Objectives
1. <major goal>
2. <major goal>
3. <major goal>

## Stack
- Language / framework
- Key dependencies

## Output
./path/to/output/directory

## Resources
- [Link or file reference]
- Embedded code blocks are fine

## Done when
- All objectives pass tests
- Entry point runs without error
```

The spec can be as detailed or as loose as needed — Claude will derive missing
fields from context. The more explicit the objectives, the more accurate the scoring.

---

## Flags

| Flag | Effect |
|------|--------|
| `--unattended` | Suppresses all confirmation pauses; Claude makes decisions autonomously |
| `--dangerously-skip-permissions` | Same as `--unattended` |

> For true no-prompt mode (tool-level confirmations also skipped), launch Claude Code
> with `claude --dangerously-skip-permissions` before starting the session.

---

## Completion Scoring (95% gate)

| Grade   | Meaning | Points |
|---------|---------|--------|
| PASS    | Implemented and verified | 1.0 |
| PARTIAL | Core works, edge case / polish missing | 0.5 |
| FAIL    | Not implemented or broken | 0.0 |

`score = (PASS + 0.5 × PARTIAL) / total_objectives × 100`

Score ≥ 95% → job complete, advance.
Score < 95% → keep building.

The 5% tolerance covers: cosmetic polish, unlikely edge cases, features that need
external API keys not provided, and items explicitly marked `[nice to have]`.
These are documented in the deferred list and reported in the final summary.

---

## Blocked jobs

If a job cannot proceed (missing resource, unresolvable spec contradiction), Claude marks
it `BLOCKED`, logs the reason, skips to the next job, and reports all blocked jobs in the
final summary.

---

## Instructions

1. Check for `--unattended` or `--dangerously-skip-permissions` in the invocation. If present,
   print the unattended-mode banner and proceed without pausing for confirmation anywhere.

2. Discover or validate the job file list (Phase 0 of the skill).

3. Print the full queue before starting work.

4. Execute each job in order using the full job-list skill protocol
   (Phases 1–4: parse → build → verify → score → gate → transition).

5. After the last job, print the Final Run Summary with:
   - Each job name, score, and status (✓ / ✗ / BLOCKED)
   - All deferred items grouped by job
   - All blocked jobs with their reasons
