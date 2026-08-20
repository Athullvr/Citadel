# Project Checkpoint — Pre-Execution AI Agent Cost Predictor

**Last updated:** 2026-08-21
**Status:** Phase 1 (data collection pipeline) is BUILT and SMOKE-TESTED, not yet run at full scale.

## What this project is

A narrow, focused tool that predicts AI agent LLM-token cost RANGES *before* an
agent runs, given a task description + tool list. Differentiator vs. the
crowded "agent cost governance" space (Portkey, LiteLLM, Datadog, Langfuse,
etc.): those are all reactive (measure during/after a run); this is
pre-execution / predictive. Explicitly NOT building: budget enforcement,
observability dashboards, or multi-framework support (see full spec below).

Full original spec/scope was given by the user in the first message of this
session — re-read it if available in conversation history. Key structure:

- **Phase 1** — data collection pipeline (agent runner + task set + logging)
- **Phase 2** — feature extraction + baseline regression model with prediction intervals
- **Phase 3** — simple web UI (paste task -> predicted range + "why" breakdown)
- **Phase 4** — validation/demo polish (real-example comparisons, README with honest limitations)

Work through phases in order; confirm with user after each phase before moving on.

## Current state (Phase 1)

Location: `data_collection/` (in repo root, alongside this file).

Files that exist and are working:
- `data_collection/tools.py` — 7 synthetic/mock tools (web_search, fetch_url,
  read_document, list_files, calculator, send_email, draft_document). No real
  network calls — deterministic cost, no flakiness, but variable-length
  outputs to realistically drive context growth across turns.
- `data_collection/tasks.py` — 20 hand-designed tasks spanning 0-5 tools and
  3 categories (single_shot, narrow_multi_step, open_ended). Categories are
  for our own stratification only, not fed to the model or used as a
  ground-truth feature.
- `data_collection/agent_runner.py` — manual ReAct loop (NOT the SDK tool
  runner, for full per-turn token logging) against **Claude Sonnet 5**
  (`claude-sonnet-5` — user explicitly chose this model over Haiku to save
  cost; keep it on Sonnet 5 unless told otherwise). 15-turn safety cap.
  Logs per-turn input/output/cache tokens, tool calls, stop reasons.
- `data_collection/run_collection.py` — CLI driver: runs each task N times
  (agent runs are non-deterministic, so multiple samples per task are needed),
  appends structured JSON lines to a `.jsonl` file.
- `data_collection/requirements.txt` — just `anthropic>=0.90.0`.
- `data_collection/.venv/` — Python 3.12 venv already created with deps installed.

### Smoke test already run (2 real API calls, confirms the core hypothesis)

| Task | Turns | Total tokens | Per-turn input tokens |
|---|---|---|---|
| t13 (single-shot calculator, 0 tools that matter) | 2 | 1,375 | 593 -> 690 (flat) |
| t05 (open-ended, 4 tools, "research 5 sources -> report -> 3 emails") | 5 | 31,578 | 996 -> 2,074 -> 4,399 -> 7,711 -> 9,910 (clearly compounding) |

This already demonstrates the "context accumulates every turn" cost driver
the whole product is built around. Smoke-test output file was deleted after
inspection (`data/smoke_test.jsonl` no longer exists — don't expect to find it).

## Decision pending / next action

**We were about to run the FULL paid data collection** (20 tasks x 3-5
repeats against the real Sonnet 5 API) when the user paused to switch IDEs.

- User already declined switching to Haiku to save cost (explicitly chose to
  keep Sonnet 5 — see decision log below). Do not re-ask this.
- Options still on the table, not yet decided:
  1. Full collection, 4 repeats/task = 80 runs, est. **$3-7**
  2. Full collection, 5 repeats/task = 100 runs, est. **$5-10**
  3. Smaller pilot first, 10 tasks x 3 repeats = 30 runs, est. **$1-3**
  4. User runs it themselves via `python run_collection.py` (venv + API key
     already set up; just `cd data_collection && source .venv/Scripts/activate
     && python run_collection.py --runs-per-task 4`)

**Next step when resuming:** ask the user which of the above to do, then
(if delegated back to Claude) run `run_collection.py` with the chosen
`--runs-per-task`, writing to `data_collection/data/runs.jsonl` (default
path — don't overwrite with a different name unless asked).

After the full collection completes, the deliverable per the "how to work"
instructions is: show the user what was collected/learned, and confirm
before moving to Phase 2 (feature extraction + baseline model). If the
real data turns out too noisy/small for a real trained model, the fallback
per the original spec is to pivot to a transparent rule-based/heuristic
estimator instead, leading with the feature-explanation angle.

## Decision log (don't re-ask these)

- Model for data collection: **Sonnet 5** (`claude-sonnet-5`), explicitly
  chosen by user over Haiku 4.5 despite Haiku being ~2x cheaper.
- Repeats-per-task and whether to pilot-first: **not yet decided** (see above).

## Environment notes

- Platform: Windows, Git Bash shell available.
- Python 3.12.10 available as `python` (not `python3`) once in the right
  shell/venv.
- `ANTHROPIC_API_KEY` is set in the environment already — no auth setup needed.
- Repo has git initialized but **no commits yet** (`git status` showed "No
  commits yet" at session start) — nothing has been committed. All work
  described above exists only as uncommitted working-tree files.
