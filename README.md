# Citadel Predict — Pre-Execution AI Agent Cost & Token Budget Predictor

[![PyPI Version](https://img.shields.io/pypi/v/citadel-predict.svg)](https://pypi.org/project/citadel-predict/)
[![MCP Package](https://img.shields.io/pypi/v/citadel-predict-mcp.svg?label=mcp-server)](https://pypi.org/project/citadel-predict-mcp/)
[![Python Versions](https://img.shields.io/pypi/pyversions/citadel-predict.svg)](https://pypi.org/project/citadel-predict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Citadel Predict** is a pre-execution token budget and cost prediction platform for autonomous AI agent workflows. It predicts the LLM token consumption of an agent run **before the agent executes** — outputting a calibrated uncertainty range (`[low, expected, high]`), driving factor explanations, and out-of-distribution (OOD) risk flags.

---

## The Core Problem

Existing agent governance and observability tools (Langfuse, Portkey, LiteLLM, Helicone, LangSmith) are **reactive** — they record tokens and costs *during* or *after* execution.

**Citadel Predict** is **predictive** — forecasting token consumption and cost bounds *before* initiating expensive agent loops, enabling pre-flight budget guardrails, dynamic model routing, and human-in-the-loop approvals.

---

## ⚡ Quickstart

### 1. Python SDK & CLI (`citadel-predict`)

Install the official client from PyPI:

```bash
pip install citadel-predict
```

#### CLI Usage (Zero-Config)
```bash
# Pretty-printed cost estimate card
citadel-predict --task "Research competitor pricing across 3 sources and draft report" --tools web_search,draft_document

# Scripting / CI mode (JSON output)
citadel-predict --task "Calculate statistical metrics" --tools calculator --json
```

#### Python SDK (Pre-Flight Agent Guardrails)
```python
from citadel_predict import predict_cost

# Evaluate budget before running agent loops (LangGraph, CrewAI, AutoGen)
prediction = predict_cost(
    task_text="Perform exhaustive market research across 20 industry filings",
    tools=["web_search", "fetch_url", "draft_document"]
)

print(f"Expected: {prediction['expected_tokens']:,} tokens (Range: {prediction['low_tokens']:,} – {prediction['high_tokens']:,})")
print(f"Driving Factors: {prediction['driving_factors']}")

# Guardrail: Escalate if upper bound exceeds budget
if prediction["high_tokens"] > 10000:
    print("Warning: Upper estimate exceeds token budget. Escalating for human review.")
```

---

### 2. Claude Desktop & Claude Code (MCP Server)

Citadel Predict includes an official Model Context Protocol (MCP) server for Claude Desktop and Claude Code.

#### Claude Desktop Setup (Zero-Config)
In Claude Desktop, go to **Settings (⚙️) → Developer → Edit Config** and add `citadel-predict` under `mcpServers`:

```json
{
  "mcpServers": {
    "citadel-predict": {
      "command": "uvx",
      "args": ["citadel-predict-mcp"]
    }
  }
}
```
*(Or with pipx: `"command": "pipx", "args": ["run", "citadel-predict-mcp"]`)*

Restart Claude Desktop. The `estimate_agent_cost` tool will now be active in your Claude conversations.

#### Automated Configuration Script
You can also run the cross-platform setup script from your terminal:

```bash
python scripts/configure_mcp.py --non-interactive
```

---

## How It Works

Citadel Predict extracts **10 structural & lexical features** directly from the task prompt and available tool schemas:
1. `text_char_len`: Character length of the task prompt.
2. `text_word_len`: Word count of the prompt.
3. `num_tools`: Number of tools provided to the agent.
4. `open_ended_keyword_hits`: Matches for exploratory terms (`"research"`, `"investigate"`).
5. `narrow_keyword_hits`: Matches for single-step terms (`"calculate"`, `"convert"`).
6. `max_explicit_count`: Highest explicit quantity mentioned (`"5 sources"`, `"3 emails"`).
7. `sum_explicit_counts`: Total sum of explicit quantities.
8. `step_connector_hits`: Sequential step markers (`"then"`, `"next"`, `"after that"`).
9. `num_clauses`: Structural sentence clause count.
10. `is_question`: Boolean indicator for question syntax.

Features are evaluated by a **Gradient Boosting Regressor** with split-conformal prediction intervals calibrated via Leave-One-Task-Out (LOTO) cross-validation.

---

## Monorepo Architecture

```text
Citadel/
├── backend/                      # FastAPI inference API & production backend
│   ├── main.py                   # API endpoints, auth, rate limiting, error handlers
│   ├── Dockerfile                # Production Docker container
│   └── tests/                    # Backend unit & integration test suite
├── data_collection/              # Dataset, feature engineering & model training
│   ├── features.py               # 10-feature extraction engine
│   ├── predict.py                # Inference, model registry & OOD confidence evaluator
│   ├── tasks.py & tools.py       # Benchmark tasks and synthetic tools
│   ├── calibration_data/         # Serialized model bundles (.joblib)
│   └── data/runs.jsonl           # Benchmark empirical execution logs
├── frontend/                     # Next.js 16 Web Dashboard & interactive estimator
├── packages/
│   ├── citadel-predict/          # Official Python SDK & CLI (PyPI: citadel-predict)
│   └── citadel-predict-mcp/      # Official MCP Server (PyPI: citadel-predict-mcp)
├── scripts/                      # Setup scripts & git integrity hooks
├── CITATION.cff                  # Citation metadata
├── LICENSE                       # MIT License
└── pyproject.toml                # Monorepo Python configuration (pytest, ruff, mypy)
```

---

## Running Locally

### 1. Backend API
```bash
cd backend
source ../data_collection/.venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
# Accessible at http://localhost:3000
```

---

## Running Tests & Quality Checks

Execute all 73 automated tests across the monorepo:

```bash
pytest backend/tests packages/citadel-predict/tests packages/citadel-predict-mcp/tests -v
```

---

## Production Deployment

### Live API Backend
The backend is hosted live at `https://citadel-7j9u.onrender.com` (deployed on Render).

To deploy your own instance:
1. Connect this repository to your Render Dashboard.
2. Build Command: `pip install -r backend/requirements.txt`
3. Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Optional Env Vars: `CITADEL_REQUIRE_AUTH=true`, `CITADEL_API_KEY=cp_live_...`

---

## Honest Capabilities & Known Limitations

1. **Baseline Model**: Calibration is currently fit on **Claude Sonnet** ($N=20$ benchmark task archetypes, 80 empirical runs).
2. **Synthetic Tool Sizing**: Collected with deterministic mock tools with representative output expansion. Real tools with massive dynamic responses (e.g. raw HTML scraping) may exhibit higher variance.
3. **Statistical Interval**: Predictions provide empirical prediction intervals (~80% LOTO coverage), not guarantees against infinite loops or divergent agent reasoning.
4. **Out-of-Distribution Flags**: Tasks exceeding length thresholds or tool boundaries are clearly flagged with `out_of_distribution=True` and specific advisory reasons.

---

## Citation

If you use Citadel Predict in your research or applications, please cite:

```bibtex
@software{Athul_Citadel_2026,
  author = {Athul},
  title = {{Citadel: Pre-execution Token Budget and Cost Predictor for AI Agents}},
  year = {2026},
  url = {https://github.com/Athullvr/Citadel},
  version = {0.1.1}
}
```

---

## License

[MIT License](LICENSE) © 2026 Athul
