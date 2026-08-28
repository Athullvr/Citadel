# Citadel Predict — Pre-Execution AI Agent Cost Predictor

**Citadel Predict** is a pre-execution token budget prediction tool for AI agent workflows. It predicts the LLM-token cost of an agent run **before the agent executes** — outputting a calibrated uncertainty range (`[low, expected, high]`), driving factor explanations, and out-of-distribution confidence flags.

---

## The Gap Citadel Predict Targets

The agent cost governance ecosystem (Portkey, LiteLLM, Datadog LLM Observability, LangSmith, Arize Phoenix) is **reactive** — measuring and enforcing cost during or after an execution. Citadel Predict forecasts cost *before a single token is spent*, enabling proactive routing, human approval thresholds, and automated budgeting.

---

## Core Technical Architecture & Forward-Compatibility

```
Citadel/
├── .github/workflows/ci.yml     # GitHub Actions: Python & Node lints, types, tests, Docker build
├── pyproject.toml               # Python tooling configuration (ruff, mypy, pytest)
├── backend/
│   ├── main.py                  # FastAPI: Bearer Auth, Rate Limiting, JSON Logs, model_id routing
│   ├── requirements.txt         # Production & test dependencies (slowapi, httpx, pytest, ruff, mypy)
│   ├── Dockerfile               # Multi-worker container deployment
│   └── tests/
│       ├── conftest.py          # Pytest environment fixtures
│       ├── test_api.py          # API security, validation, rate limiting & endpoint tests
│       └── test_features.py     # Feature extraction edge-case test suite
├── data_collection/
│   ├── calibration_data/
│   │   └── claude-sonnet.joblib # Frozen v1 baseline calibration bundle (N=20 tasks, 80 runs)
│   ├── features.py              # Pure feature extractor (10 numerical features)
│   ├── predict.py               # Model registry, model_id lookup & OOD confidence evaluator
│   ├── tasks.py & tools.py      # Benchmark tasks and 7 mock tools
│   └── data/runs.jsonl          # Benchmark empirical execution logs
├── packages/
│   ├── citadel-predict/         # Python client library + CLI (pip install citadel-predict)
│   └── citadel-predict-mcp/     # MCP stdio server for Claude Desktop & Claude Code
├── frontend/
│   ├── src/app/page.tsx         # Next.js UI: Live predictor, confidence badge, OOD alerts, validation cards
│   ├── src/app/page.test.tsx    # Vitest component smoke tests
│   └── vitest.config.mjs        # Vitest configuration
├── MIGRATION_NOTES.md           # Instructions for Phase 2 incremental model additions (Gemini, Groq)
└── README.md
```

### Multi-Model-Ready Architecture (Phase 1)
- **Pluggable Model Registry**: Predictions accept `model_id` (default: `"claude-sonnet"`).
- **Decoupled Calibration Bundles**: Model weights and residual bands are stored under `data_collection/calibration_data/{model_id}.joblib`. Adding a future model (Gemini, Groq/Llama) is purely additive (see [MIGRATION_NOTES.md](file:///c:/Users/Athul%20VR/OneDrive/Desktop/Citadel/MIGRATION_NOTES.md)).
- **Frozen Sonnet Dataset**: The Claude Sonnet calibration baseline ($N=20$ tasks, 80 runs) is finalized.

---

## Security & Production Hardening

- **API Key Authentication**: Optional or enforced via `CITADEL_API_KEY` or `API_KEY` env var (`Authorization: Bearer <key>`).
- **Rate Limiting**: Integrated via `slowapi` (`30 req/min` on `/api/predict`, `60 req/min` on `/api/tools`).
- **Input Validation & ReDoS Protection**: Strict Pydantic constraints (`task_text` $\le 4000$ chars, `tools` $\le 20$ items).
- **CORS Enforcement**: Explicit origin whitelisting via `ALLOWED_ORIGINS` (default: `http://localhost:3000`).
- **Structured JSON Logging**: Standardized JSON lines emitted for every request (timestamp, method, path, status, latency, model_id).
- **Serving**: Configured for multi-worker uvicorn execution via `${WORKERS:-2}` in `backend/Dockerfile`.

---

## Running Locally

### 1. Backend API
```bash
cd backend
../data_collection/.venv/Scripts/activate   # Windows (.venv/bin/activate on Linux/macOS)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Running Tests & Quality Checks

### Backend Test Suite & Type Checking
```bash
# Run 16 automated tests (Auth, Rate Limiting, Validation, OOD, Features)
pytest backend/tests -v

# Run Ruff linter
ruff check backend/ data_collection/features.py data_collection/predict.py

# Run Mypy static type checker
mypy backend/main.py data_collection/features.py data_collection/predict.py
```

### Frontend Tests & Type Checking
```bash
cd frontend

# Run Vitest unit & component smoke tests
npm run test

# Run ESLint
npm run lint

# Run TypeScript type check
npx tsc --noEmit
```

---

## Deployment

### Container Build (Backend)
Build from the **repository root**:
```bash
docker build -f backend/Dockerfile -t citadel-predict-api .
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://your-frontend.vercel.app \
  -e CITADEL_API_KEY=your-secure-key \
  -e WORKERS=2 \
  citadel-predict-api
```

### Frontend (Vercel)
Deploy `frontend/` to Vercel. Configure environment variables:
- `NEXT_PUBLIC_API_BASE`: `https://your-backend-api.com`
- `NEXT_PUBLIC_CITADEL_API_KEY`: `your-secure-key` (if auth enabled)

---

## Honest Capabilities & Known Limitations

1. **Single Model Baseline**: Currently fit exclusively on **Claude Sonnet 5** ($N=20$ benchmark tasks, 80 runs).
2. **Synthetic Tool Sizing**: Data was collected using 7 deterministic mock tools. Real-world tools with high output length variance (e.g. 50KB JSON payloads) will cause higher context growth.
3. **Statistical Range, Not Exact Simulation**: Predictions represent an empirical prediction interval with split-conformal calibration (~80% LOTO coverage), not an exact guarantee.
4. **Out-of-Distribution Flags**: Tasks with 0 tools, $>5$ tools, $>600$ characters, or no standard action keywords are flagged as `confidence="low"` or `out_of_distribution=True` with specific advisory notes.
