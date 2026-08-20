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

## Phase 2 results (feature extraction + model, 2026-08-21)

New files in `data_collection/`:
- `features.py` — `extract_features(task_text, tool_names)`: derives 10
  numeric features from ONLY raw task text + tool list (never the
  hand-labeled `category`), e.g. `num_tools`, `open_ended_keyword_hits`
  (keyword match against phrases like "research", "until", "investigate"),
  `narrow_keyword_hits`, explicit repeated-item counts ("5 sources"), text
  length, clause count. This is the exact function Phase 3's UI will call.
- `build_dataset.py` — builds one row per run (80 rows) with features +
  `log_total_tokens` target (log-transformed: token cost is heavy right-skewed).
- `train_model.py` — trains the model, runs leave-one-task-out (LOTO)
  validation, prints coverage, saves `model.joblib`.
- `predict.py` — loads `model.joblib` and predicts `[low, expected, high]`
  tokens + a list of human-readable "driving factors" for a new task.

**First approach tried and REJECTED:** three independently-fit quantile
gradient-boosted-tree models (10th/50th/90th percentile). With only ~19
training tasks per LOTO fold, the three models disagreed badly and produced
a band with only **45% coverage** against an 80% target — badly
overconfident. Documented as a rejected approach directly in `train_model.py`'s
docstring so it isn't retried without reason.

**Final approach:** a single point-estimate GradientBoostingRegressor
(shallow: max_depth=2, n_estimators=40, min_samples_leaf=4 — regularized
for n=20) predicting log-tokens, plus a prediction band built from the
empirical distribution of out-of-fold residuals (10th/90th percentile of
LOTO residuals in log space) — i.e. a split-conformal-style calibration.
This is standard practice specifically for tiny-n regimes like this one.

**Validation result: 80.0% overall coverage** (target ~80%, using
10%/90% residual quantiles) via leave-one-task-out CV (not leave-one-run-out
— the 4 repeats of a task share a feature vector, so LOTO is the only fold
structure that tests genuine generalization to an unseen task).
- By category: narrow_multi_step 100%, single_shot 75%, open_ended 71%.
- The two coverage misses in single_shot are t01 and t17 — the ONLY two
  zero-tool tasks in the whole 20-task set. When either is held out, only
  one zero-tool example remains in training, insufficient to anchor the
  model at that extreme (near-260-token) end. This is a legitimate small-n
  edge case, not a systemic flaw — flag in the README as a known limitation
  (model is weaker at the very-low-cost extreme with so few examples there).

**Feature importance** (from the final full-data model):
`num_tools` 69%, `open_ended_keyword_hits` 18%, `narrow_keyword_hits` 8%,
`text_char_len`/`text_word_len` ~5% combined, everything else (explicit
counts, clause count, step connectors, is_question) ~0%. Honest read: tool
count and open/narrow keyword phrasing are doing almost all the work; the
more elaborate structural features (clause counting, explicit-count
extraction) aren't adding signal yet at this dataset size — worth
revisiting if/when more data is collected, not a blocker for Phase 3.

**Important honesty caveat (documented in train_model.py and to repeat in
the README):** the residual band's width is calibrated from the SAME pooled
LOTO residuals that coverage is then measured against. This is a reasonable
proof-of-concept calibration but not a fully nested (double) cross-
validation — that would need more than 20 tasks to be stable. Be upfront
about this in any user-facing writeup.

**Verdict: Phase 2 succeeded — did NOT need to fall back to the rule-based/
heuristic estimator.** The trained model + calibrated band approach works
and generalizes reasonably at n=20. Proceeding to Phase 3.

## Next step

Start Phase 3: simple web UI (Next.js or plain React) where a user pastes a
task description + optional tool list and gets back `predict()`'s
`[low, expected, high]` range plus the "driving factors" explanation panel.
The UI should call into the existing `predict.py` logic (either via a small
Python backend, e.g. FastAPI, or by porting the (simple, ~10-feature) logic
to JS — decide with the user which is preferred before building). Must be
explicit in the UI copy that this is a statistical estimate from a small
observed dataset, not a guarantee (per original spec + the honesty caveats
above).

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
