# CLAUDE.md — Citadel Predict Project Guide

Welcome to the **Citadel** codebase. This guide provides full context on the architecture, workflows, conventions, testing, and Model Context Protocol (MCP) integrations for Claude and Claude Code.

---

## 1. Project Overview

**Citadel Predict** is a pre-execution token budget and cost prediction platform for autonomous AI agent workflows.

### Core Value Proposition
- **Predictive, Not Reactive**: Existing agent governance tools (Langfuse, Portkey, LiteLLM) record token usage *during* or *after* runs. Citadel Predict estimates token consumption bounds ($low, expected, high$) and out-of-distribution (OOD) risks **before** agent execution or multi-step tool loops begin.
- **Statistical Prediction vs. Simulation**: Uses 10 structural & lexical features extracted directly from task text and available tools, evaluated by a Gradient Boosting Regressor with conformal prediction intervals calibrated via Leave-One-Task-Out (LOTO) cross-validation.

---

## 2. Monorepo Architecture & Directory Map

```text
Citadel/
├── backend/                  # FastAPI inference API & production backend
│   ├── main.py               # API endpoints, auth, rate limiting, error handlers
│   ├── Dockerfile            # Production Docker image (inference path only)
│   └── tests/                # Backend unit & integration tests
├── data_collection/          # Dataset, feature engineering & model training
│   ├── features.py           # 10-feature extraction engine (FEATURE_NAMES)
│   ├── predict.py            # Model loading, inference, OOD checks & explanations
│   ├── tasks.py              # 20 benchmark tasks (single_shot, narrow, open_ended)
│   ├── tools.py              # 7 synthetic tools with realistic output growth
│   ├── agent_runner.py       # ReAct agent loop for Claude Sonnet (Sonnet 5)
│   ├── build_dataset.py      # Feature matrix builder
│   ├── train_model.py        # Model trainer & LOTO residual band calibration
│   ├── calibration_data/     # Serialized model bundles (.joblib)
│   │   └── claude-sonnet.joblib # Frozen baseline bundle (N=20 tasks / 80 runs)
│   ├── data/                 # Benchmark runs (runs.jsonl) & validation examples
│   └── .venv/                # Main Python virtual environment (Python 3.12)
├── frontend/                 # Web UI (Next.js 16, React, Tailwind CSS)
│   ├── src/app/page.tsx      # Interactive cost estimator & benchmark comparison cards
│   └── package.json
├── packages/
│   ├── citadel-predict/      # Official Python SDK & CLI client
│   │   ├── src/citadel_predict/ # HTTP client, errors, config resolver, CLI
│   │   └── tests/            # SDK & CLI test suite
│   └── citadel-predict-mcp/  # Official Model Context Protocol (MCP) server
│       ├── src/citadel_predict_mcp/ # stdio MCP server for Claude Desktop / Code
│       └── tests/            # MCP tool registration & error translation tests
├── pyproject.toml            # Monorepo Python configuration (pytest, ruff, mypy)
├── CHECKPOINT.md             # Project implementation log and milestone history
└── MIGRATION_NOTES.md        # Guide for adding multi-model calibration profiles
```

---

## 3. Key Components & Specifications

### A. Feature Extraction (`data_collection/features.py`)
Extracts exactly 10 numerical features from the task string and tool list:
1. `text_char_len`: Character length of the task prompt.
2. `text_word_len`: Word count of the task prompt.
3. `num_tools`: Number of tools provided to the agent.
4. `open_ended_keyword_hits`: Matches for exploratory terms (`"research"`, `"until"`, `"investigate"`, etc.).
5. `narrow_keyword_hits`: Matches for deterministic/single-step terms (`"calculate"`, `"convert"`, `"format"`, etc.).
6. `max_explicit_count`: Highest explicit quantity mentioned (`"5 sources"`, `"3 emails"`).
7. `sum_explicit_counts`: Total sum of explicit quantities.
8. `step_connector_hits`: Sequential step markers (`"then"`, `"next"`, `"after that"`, etc.).
9. `num_clauses`: Structural sentence clause count.
10. `is_question`: Boolean indicator for question syntax.

### B. Backend API (`backend/main.py`)
- `POST /api/predict`: Returns `expected_tokens`, `low_tokens`, `high_tokens`, `out_of_distribution`, `ood_reasons`, `driving_factors`.
- `GET /api/tools`: Returns the list of standard tool schemas for frontend pickers.
- `GET /api/validation-examples`: Pre-computed out-of-fold benchmark cards (2 hits, 2 misses for transparency).
- `GET /api/health` & `GET /api/version`: Health checks and model metadata.
- **Middleware**: SlowAPI rate limiting, optional Bearer Token authentication (`CITADEL_REQUIRE_AUTH`), standardized JSON errors across 400/401/422/429/500 status codes.

### C. Python SDK & CLI (`packages/citadel-predict`)
- Direct programmatic invocation: `predict_cost(task_text="...", tools=[...])`.
- Pre-flight guardrails for LangGraph, CrewAI, AutoGen loops.
- Standalone CLI: `citadel-predict --task "..." --tools list_files,read_document`.

### D. MCP Server (`packages/citadel-predict-mcp`)
- Implements Model Context Protocol over standard I/O (`stdio`).
- Exposes tool: `estimate_agent_cost(task_text, tools, num_tools, model_id)`.
- Traps all HTTP/network/validation errors and formats them into actionable JSON responses so Claude can communicate issues cleanly without server termination.

---

## 4. Development & Testing Commands

### Virtual Environment
The primary virtual environment is located at `data_collection/.venv/`.

**Activate environment on Windows (PowerShell):**
```powershell
& ".\data_collection\.venv\Scripts\Activate.ps1"
```

**Activate environment on Windows (Git Bash / bash):**
```bash
source data_collection/.venv/Scripts/activate
```

### Running Tests
Execute pytest across all test suites (Backend, SDK, and MCP):
```bash
data_collection/.venv/Scripts/pytest
```

Run specific package test suites:
```bash
# Backend tests (17 tests)
data_collection/.venv/Scripts/pytest backend/tests -v

# SDK tests (26 tests)
data_collection/.venv/Scripts/pytest packages/citadel-predict/tests -v

# MCP server tests (13 tests)
data_collection/.venv/Scripts/pytest packages/citadel-predict-mcp/tests -v
```

### Running Backend API Locally
```bash
cd backend
../data_collection/.venv/Scripts/uvicorn main:app --reload --port 8000
```

### Running Web Frontend Locally
```bash
cd frontend
npm run dev
# Accessible at http://localhost:3000
```

### Testing MCP Server Directly via CLI
```bash
data_collection/.venv/Scripts/citadel-predict-mcp --help
data_collection/.venv/Scripts/citadel-predict-mcp --version
```

---

## 5. Model Context Protocol (MCP) Configuration

Configuration for **Claude Desktop** and **Claude Code** is automatically generated by the cross-platform setup script rather than hand-edited, ensuring identical, drift-free configurations across Windows, macOS, and Linux.

### One-Command Setup

Run the automated configuration script from your active Python environment:

```bash
# Interactive setup (prompts for optional API key / custom endpoint):
python scripts/configure_mcp.py

# Or non-interactive setup with options:
python scripts/configure_mcp.py --api-url http://localhost:8000 --non-interactive
```

### What `scripts/configure_mcp.py` Does:
1. **Auto-Detects Executable**: Resolves the platform-specific executable (`citadel-predict-mcp.exe` on Windows, `citadel-predict-mcp` on macOS/Linux) from your active Python environment.
2. **Targets Both Platforms Automatically**:
   - **Claude Desktop**:
     - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
     - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
     - Linux: `~/.config/Claude/claude_desktop_config.json`
   - **Claude Code**: Project-level `.claude/settings.json`.
3. **Safe & Idempotent**: Merges the `citadel-predict` configuration under `mcpServers` without overwriting other existing servers or user settings.

---

## 6. Critical Invariants & Rules

1. **Frozen Sonnet Baseline**: The dataset in `data_collection/data/runs.jsonl` and model bundle `data_collection/calibration_data/claude-sonnet.joblib` represent the frozen v1 baseline. Do not alter or rerun Claude Sonnet API data collection unless explicitly instructed.
2. **Feature Parity**: Any new model calibration profile added to the system MUST strictly use the exact 10 numerical features defined in `data_collection/features.py`.
3. **Additive Multi-Model Extensibility**: Adding a new model profile (e.g. Gemini, Llama) is done strictly by placing `<model_id>.joblib` in `data_collection/calibration_data/` and registering the ID in `predict.py` (see `MIGRATION_NOTES.md`).
4. **Clean Stdio for MCP**: The MCP server must NEVER print debug statements or non-JSON-RPC messages to standard output (`stdout`). All diagnostics must go to `stderr`.
5. **Standardized Error Schemas**: All API error responses must follow the schema:
   ```json
   {
     "error": "ErrorType",
     "message": "Human readable explanation",
     "detail": "Technical error detail",
     "retry_after": null
   }
   ```

---

## 7. Before Pushing to Main

To ensure zero regressions and protect the frozen baseline:

### Automated Safeguards Setup (Recommended)
Activate local git safeguards:
```bash
# Option A: Zero-dependency native git hooks installer
python scripts/install_git_hooks.py

# Option B: Standard pre-commit framework
pip install pre-commit && pre-commit install && pre-commit install --hook-type pre-push
```

### Manual Verification Checklist (Before Push)
1. **Run Full Test Suite**:
   ```bash
   data_collection/.venv/Scripts/pytest -v
   ```
2. **Verify Frozen Baseline is Untouched**:
   ```bash
   git diff --stat HEAD~1 | grep -E "claude-sonnet\.joblib|runs\.jsonl"
   # Must produce zero output
   ```
3. **Post-Deploy Health Check (Render)**:
   ```bash
   curl -i https://<your-render-service>.onrender.com/api/health
   # Must return 200 OK with {"status": "ok", "model_id": "claude-sonnet"}
   ```

