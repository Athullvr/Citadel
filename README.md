# Pre-Execution AI Agent Cost Predictor

A narrow, focused tool that predicts the LLM-token cost of an AI agent task
**before the agent runs** — a range (low/expected/high), not a false-precision
single number, plus a plain-English explanation of what's driving the estimate.

## The gap this targets

The "agent cost governance" space (Portkey, LiteLLM, Datadog LLM Observability,
Braintrust, LangSmith, Arize Phoenix, and similar) is crowded, well-funded, and
entirely **reactive** — every one of these tools measures or enforces cost
during or after a run. None of them forecast cost before a single token is
spent. That's the specific, narrow gap this project targets. It deliberately
does **not** try to be a budget-enforcement/kill-switch tool, a general
observability dashboard, or a multi-framework agent runner — those are
someone else's crowded market.

## The core technical bet

Don't build a simulator. An agent loop is emergent and non-deterministic —
you cannot know in advance exactly how many turns it will take or how much
context will accumulate. Trying to predict that exactly is architecturally
unsolvable.

Instead: **learn from observed patterns.** Run a set of representative agent
tasks for real, log actual token usage, extract a handful of features from
the task description alone (the same information available *before* the
agent runs), and fit a model that maps those features to a calibrated cost
range. The claim is not "we know what will happen" — it's "here is what
happened on similar tasks before, with an honest uncertainty band."

## Architecture

```
data_collection/          Phase 1 + 2: data collection & model training
  tools.py                 7 synthetic mock tools (deterministic, no network calls)
  tasks.py                 20 hand-designed representative tasks
  agent_runner.py           manual ReAct loop against Claude Sonnet 5, full per-turn token logging
  run_collection.py         CLI driver: runs each task N times, logs to data/runs.jsonl
  features.py               extract_features(task_text, tools) -> 10 numeric features
  build_dataset.py           builds the training dataframe from runs.jsonl
  train_model.py             trains model + validates (leave-one-task-out), saves model.joblib
  generate_examples.py       generates the 4 "compared to real runs" demo examples
  predict.py                 predict(task_text, tools) -> low/expected/high + driving factors

backend/                  Phase 3: API
  main.py                   FastAPI wrapping predict.py directly (no reimplementation)

frontend/                 Phase 3: UI
  src/app/page.tsx          Next.js UI: task input, tool picker, range visualization,
                             "why this estimate" panel, "how this compares to real runs" section
```

### Why a model, not just tool-count × a multiplier

The first thing tried was three independently-fit quantile gradient-boosted
models (10th/50th/90th percentile). With only ~19 training tasks per
validation fold, they disagreed badly and produced a band that only covered
the true outcome 45% of the time against an 80% target. The fix: a single
point-estimate regressor plus a prediction band built from the empirical
distribution of leave-one-task-out residuals (a split-conformal-style
calibration) — a much lower-variance approach for a dataset this small. This
rejected-approach note is preserved in `train_model.py`'s docstring so it
isn't retried without reason.

## How to run it

```bash
# Backend (FastAPI) -- from data_collection/'s venv, which has all deps
cd backend
../data_collection/.venv/Scripts/activate   # Windows; use bin/activate on macOS/Linux
uvicorn main:app --port 8000

# Frontend (Next.js) -- separate terminal
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

## Deployment

- **Frontend**: deploy `frontend/` to **Vercel** (native Next.js support,
  zero-config). Set `NEXT_PUBLIC_API_BASE` to the deployed backend URL.
- **Backend**: deploy the container built from `backend/Dockerfile` to a
  container host (**Render**, **Fly.io**, or **Railway**) — not a serverless
  Python function host, since scikit-learn/numpy/pandas's combined size and
  cold-start cost are a poor fit for that model. Build from the **repo
  root** (the Dockerfile needs `data_collection/`'s inference files):

  ```bash
  docker build -f backend/Dockerfile -t agent-cost-predictor-api .
  docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-frontend.vercel.app agent-cost-predictor-api
  ```

  The image copies only the inference path (`features.py`, `predict.py`,
  `tools.py`, `model.joblib`, `validation_examples.json`) — not the training
  scripts (`agent_runner.py`, `train_model.py`, `run_collection.py`, the
  `anthropic` SDK, or the full `runs.jsonl` dataset), since none of those are
  needed to serve predictions. Set the `ALLOWED_ORIGINS` env var to your
  deployed frontend's URL (comma-separated for multiple origins) so CORS
  allows it; most container hosts inject `PORT` automatically, which the
  image's `CMD` already respects.

To regenerate the dataset/model from scratch (costs real API money against
Claude Sonnet 5 — roughly $4 for the full 80-run collection used here):

```bash
cd data_collection
source .venv/Scripts/activate
python run_collection.py --runs-per-task 4   # writes data/runs.jsonl
python train_model.py                         # trains + validates + saves model.joblib
python generate_examples.py                   # regenerates the demo comparison examples
```

## What was learned (validation results)

- **80 real Claude Sonnet 5 agent runs, 20 tasks, 4 repeats each.** Zero
  failures, zero runs hit the 15-turn safety cap.
- Mean tokens per run by task type: single-shot tasks (0-1 tools) averaged
  ~1,556 tokens; narrow multi-step tasks averaged ~6,478; open-ended tasks
  (implying research/iteration) averaged ~19,078 — roughly an order of
  magnitude apart per category, confirming that task-level language and tool
  count are strong, learnable cost signals.
- **Leave-one-task-out validation: 80.0% overall coverage** (target ~80%,
  using 10th/90th percentile residual quantiles) — meaning on tasks the
  model had never seen, its predicted range contained the actual outcome 4
  times out of 5.
  - By category: narrow_multi_step 100%, single_shot 75%, open_ended 71%.
- **Feature importance**: `num_tools` (69%) and open/narrow-ended keyword
  phrasing (26% combined) drive almost the entire prediction. More elaborate
  structural features (explicit-count extraction, clause counting) contribute
  ~0% at this dataset size — they may matter with more data, but aren't
  pulling weight yet.

## Known limitations (read before trusting this on a new domain)

- **Small dataset.** 20 hand-designed tasks, 80 runs. This is a
  proof-of-concept demonstrating the *approach* works, not a production-grade,
  broadly-generalized predictor.
- **Single model family.** All data collection used Claude Sonnet 5 with a
  fixed manual ReAct loop and 7 synthetic mock tools. Agent behavior (turn
  count, verbosity, retry patterns) will differ across other models, other
  agent frameworks, and real (non-mock) tools with different output-length
  distributions.
- **No multi-agent support.** This only models a single agent's own loop,
  not orchestrated multi-agent or sub-agent-spawning systems.
- **Band calibration is proof-of-concept, not fully nested.** The residual
  band's width is calibrated from the same pooled leave-one-task-out
  residuals that its coverage is measured against. This is standard practice
  at this dataset size but is not a fully independent (double) cross-
  validation — that would need more than 20 tasks to be stable.
- **Weaker at the extremes.** The model is least reliable on trivial
  zero-tool tasks (only 2 of 20 training tasks have zero tools) and on
  heavily-iterative open-ended tasks (71% coverage vs. 80% target) — see the
  "how this compares to real runs" section in the UI for concrete examples
  of both failure modes.

This is a statistical pattern match against a small observed dataset, not a
simulation and not a guarantee. Treat its output as a planning estimate.
