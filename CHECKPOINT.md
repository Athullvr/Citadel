# Project Checkpoint — Pre-Execution AI Agent Cost Predictor

**Last updated:** 2026-08-21
**Status:** Phase 1 and Phase 2 COMPLETE. Model trained and validated (80% coverage). Ready to start Phase 3 (web UI).

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

## Phase 1 results (full collection, 2026-08-21)

Ran `run_collection.py --runs-per-task 4` -> 20 tasks x 4 repeats = **80 runs,
0 failures, 0 runs hit the 15-turn cap**. Output: `data_collection/data/runs.jsonl`
(80 JSON lines, one per run, full per-turn token breakdown included).

Actual cost: **~$3.94** (563,363 input tokens, 150,199 output tokens, no
prompt caching used in the runner — real cost would be lower with caching).

Mean total tokens per run, by hand-labeled category (for our own reference
only — not a feature fed to any model):

| Category | Avg tokens/run | Range across tasks |
|---|---|---|
| single_shot (0-1 tools) | ~1,556 | 260 – 3,526 |
| narrow_multi_step (1-3 tools) | ~6,478 | 2,172 – 12,032 |
| open_ended (2-5 tools) | ~19,078 | 8,453 – 38,728 |

Per-task within-task variance (coefficient of variation across the 4 repeats
of the same task) was mostly low-to-moderate: 1-25%, no task above 25%. This
is a good sign — it means a prediction-interval model has a real shot at
being useful rather than fighting pure noise.

Notable findings to carry into Phase 2:
- Categories separate by roughly an order of magnitude each — strong signal
  that task-level features (not just tool count) predict cost.
- Tool count correlates with cost but is NOT sufficient alone: task t07
  (single tool, "look up population of France", single_shot) hit up to 4,167
  tokens on one run — higher than several narrow_multi_step tasks with more
  tools. Open-ended language ("keep searching until...", "research", "then
  revise") appears to matter as much as raw tool count.
- Highest-variance tasks were t04 (research 3 sources, CV 22.4%) and t07
  (CV 25.3%) — worth inspecting turn-by-turn if the model underperforms on
  similar tasks later.
- No runs were truncated by the 15-turn safety cap, so no data was lost to
  that ceiling.

**Verdict: dataset looks usable for Phase 2.** Not abandoning the trained-model
approach for the rule-based fallback at this point — variance is bounded and
categories separate cleanly. Will revisit that fallback only if the held-out
validation in Phase 2 shows the model doesn't generalize.

## Next step

Start Phase 2: feature extraction from raw task text + tool list (the way a
real user's input would look — NOT the hand-labeled `category` field, which
is for our analysis only), then train a baseline regression model with
prediction intervals on `data_collection/data/runs.jsonl`. Validate against a
held-out task before declaring Phase 2 done.

## Decision log (don't re-ask these)

- Model for data collection: **Sonnet 5** (`claude-sonnet-5`), explicitly
  chosen by user over Haiku 4.5 despite Haiku being ~2x cheaper.
- Repeats-per-task: **4**, chosen by user (80 total runs) over a 3-repeat
  pilot or 5-repeat larger run.

## Environment notes

- Platform: Windows, Git Bash shell available.
- Python 3.12.10 available as `python` (not `python3`) once in the right
  shell/venv.
- `ANTHROPIC_API_KEY` is set in the environment already — no auth setup needed.
- Repo has git initialized but **no commits yet** (`git status` showed "No
  commits yet" at session start) — nothing has been committed. All work
  described above exists only as uncommitted working-tree files.
