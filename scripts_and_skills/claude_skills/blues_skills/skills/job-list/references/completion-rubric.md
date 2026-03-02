# Completion Rubric — Job List Skill

Reference for the Phase 3 scoring step. Consult when objective classification is unclear.

---

## Scoring Scale

| Grade   | Score | Definition |
|---------|-------|------------|
| PASS    | 1.0   | The objective is fully implemented **and** verified to work correctly in the normal case. |
| PARTIAL | 0.5   | The core logic is in place and the happy path works, but at least one of the following is missing: edge-case handling, error messages, optional enhancements, or cosmetic polish. |
| FAIL    | 0.0   | Not implemented, or implemented but broken in the primary use case. |

**Score = (PASS + 0.5 × PARTIAL) / total objectives × 100**

---

## Classification Guide

### PASS — all of these must be true
- The feature exists and is accessible
- The primary use case executes without error
- Output / behavior matches the spec description
- At least one verification step (test, smoke run, or manual check) confirms it

### PARTIAL — apply when the core works but any of these are true
- Edge cases explicitly listed in the spec are not handled
- Error handling is absent but the spec mentioned it
- The feature works in isolation but is not wired into the full flow yet
- A secondary mode or option described in the spec is missing
- Output formatting differs from spec but the data is correct
- Required logging or metrics are absent

### FAIL — apply when any of these are true
- The feature does not exist in the output
- Import / syntax error prevents the module from loading
- The primary use case crashes or returns wrong results
- A hard dependency is missing and was not installed

---

## The 5% Tolerance — What Counts as Deferrable

The advancement gate is ≥ 95%. The 5% buffer exists for items that are real but
genuinely minor. Something is deferrable **only if all of the following are true**:

1. It does not break the primary user flow
2. It does not cause data loss or security issues
3. It could be fixed by a developer in under an hour
4. The spec does not label it a blocker or "must have"

**Always deferrable:**
- Cosmetic: typos in CLI output, minor layout issues, unstyled error messages
- Unlikely edge cases: e.g., file > 2 GB when spec targets normal-sized files
- Optional enhancements explicitly marked `[nice to have]` in the spec
- External infrastructure not provided: cloud accounts, third-party API keys not in scope

**Never deferrable (must be PASS before advancing):**
- The project cannot be started / run
- A feature listed as a primary deliverable is absent
- Data corruption or security vulnerability
- Installation fails on the target platform

---

## Scoring Examples

### Example A — 4 objectives, score = 87.5% (does NOT advance)

| Objective                           | Grade   | Points |
|-------------------------------------|---------|--------|
| REST API serves /health endpoint    | PASS    | 1.0    |
| /items CRUD with SQLite backend     | PASS    | 1.0    |
| JWT authentication on all routes    | PARTIAL | 0.5    |
| Rate limiting (100 req/min)         | FAIL    | 0.0    |
| **Total**                           |         | **2.5 / 4 = 62.5%** |

Score 62.5% → keep working.

### Example B — 5 objectives, score = 95% (advances)

| Objective                           | Grade   | Points |
|-------------------------------------|---------|--------|
| CLI parses --input and --output     | PASS    | 1.0    |
| Converts Markdown → HTML            | PASS    | 1.0    |
| Preserves code blocks with syntax   | PASS    | 1.0    |
| Handles nested lists correctly      | PASS    | 1.0    |
| Generates table of contents         | PARTIAL | 0.5    |
| **Total**                           |         | **4.5 / 5 = 90%** |

Score 90% → keep working.

(If "generates table of contents" were PASS → 5/5 = 100% → advances.)

### Example C — 3 objectives, score = 100% (advances)

| Objective                           | Grade   | Points |
|-------------------------------------|---------|--------|
| Loads model from GGUF path          | PASS    | 1.0    |
| Streams tokens to stdout            | PASS    | 1.0    |
| Accepts --stop-token flag           | PASS    | 1.0    |
| **Total**                           |         | **3.0 / 3 = 100%** |

Deferred (minor): help text does not show default value for --stop-token.
Advances immediately.

---

## Disputed Cases

**"The spec says X but X requires an API key that wasn't provided."**
→ PARTIAL if you implemented the integration and it would work with a key.
→ FAIL if no implementation exists at all.
→ Document in deferred: "Requires API_KEY env var — not provided in this run."

**"The feature works but tests are failing due to a test config issue, not the code."**
→ Investigate the test config. If the bug is in the test harness (not the feature), fix the harness and re-score.
→ If you cannot resolve the harness issue, score the feature on manual verification and note the test config problem in deferred items.

**"The spec is contradictory."**
→ Pick the interpretation that makes the most technical sense, document your choice in the job card, and score against your interpretation.
