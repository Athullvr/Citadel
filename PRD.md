# Citadel Predict — Current-State Product Requirements

**Document type:** As-built product requirements document  
**Status assessed:** 2026-10-29  
**Product stage:** Open-source MVP / beta  
**Purpose:** Explain what the current product does, who can use it, how it is delivered, and what needs validation before planning improvements.

This document describes the product that exists in this repository and the release evidence that could be checked. It separates verified implementation from deployment claims and unresolved operational facts. It is a baseline for future product changes, not a commitment to a roadmap.

## 1. Executive summary

Citadel Predict estimates the token use and approximate model cost of an AI-agent task **before the agent runs**. A user supplies a task description and the tools available to the agent. Citadel returns a low, expected, and high token estimate, approximate USD costs, factors that influenced the estimate, a confidence label, and warnings when the task differs from the small benchmark used to train the model.

The product is designed for developers adding a pre-flight budget check to an agent workflow, people drafting prompts in an editor, and Claude Desktop or Claude Code users who want an MCP tool to estimate a task. It does not run the agent, inspect the agent’s actual tool schemas or provider account, measure actual billing, block a run, or route work to another model. A developer must decide what to do with the estimate.

The repository contains a FastAPI inference API, a local Next.js estimator, a Python SDK and CLI, a stdio MCP server, and a VS Code-compatible extension. The Python SDK and MCP package are published on PyPI at version 0.2.0. A version 0.1.0 VSIX artifact is in the repository. The repository and client defaults identify a Render API URL, but the endpoint’s current availability and authentication configuration could not be confirmed from this workspace. The web UI has a local-development default and no deployment configuration was found in the repository.

## 2. Product and release status

| Area | Current evidence | Status and practical meaning |
|---|---|---|
| Prediction API | `backend/main.py`, `backend/Dockerfile`, and the root README describe a FastAPI service and a Render deployment at `https://citadel-7j9u.onrender.com`. SDK and extension defaults use that URL. | A hosted deployment is intended and documented. Its current live status is **unverified**: a health request from this workspace failed at DNS resolution, which does not establish whether the host itself is up or down. |
| Web estimator | Next.js app in `frontend/`; API base defaults to `http://localhost:8000`. | Usable as a local web app when connected to a running API. No frontend hosting/deployment setup was found in the repository. |
| Python SDK and CLI | `packages/citadel-predict`; published package page reports release 0.2.0. | Other developers can install it with `pip install citadel-predict`. It needs a reachable API; its configured default is the Render URL. [PyPI package](https://pypi.org/project/citadel-predict/). |
| MCP server | `packages/citadel-predict-mcp`; published package page reports release 0.2.0. | Other developers can install it with `pip install citadel-predict-mcp` and connect Claude Desktop/Code over local stdio. It calls the Citadel HTTP API. [PyPI package](https://pypi.org/project/citadel-predict-mcp/). |
| Editor extension | `packages/citadel-predict-vscode/` contains source, setup instructions, and `citadel-predict-vscode-0.1.0.vsix`. | Developers can build or install the repository’s VSIX directly. Marketplace publication was not established from repository evidence. |
| Model profiles | `data_collection/predict.py` registers only `claude-sonnet`. | Only this profile is supported by the backend. Other model names shown as examples in some package docs are not currently calibrated and will be rejected by the API. |
| User data and persistence | Inference loads a local model bundle and static validation examples. No application database or account store is present. | The current service is a stateless estimator. The API logs request metadata such as client IP, path, status, and latency; it does not log task text in its request middleware. |

The package listings were checked on 2026-10-29. The live API check could not resolve the host in this environment, so deployment health, uptime, access policy, and actual rate limits remain unknown. See [deployment and developer readiness](#9-deployment-and-developer-readiness).

## 3. Users and primary use cases

### Target users

- **Agent and application developers** who want a cost estimate before dispatching a potentially multi-step agent workflow.
- **Prompt authors** who want to see how task wording and available tools affect an estimate while drafting.
- **Claude Desktop and Claude Code users** who want the assistant to call an estimate tool during planning.
- **Operators of a self-hosted instance** who want the prediction service on infrastructure they control.

### Use case 1: estimate in the web UI

The user enters a task, selects tools from the API-provided list, and requests an estimate. The UI displays the low/expected/high token range, approximate cost range, confidence and out-of-distribution (OOD) warnings, and driving factors. A comparison section shows four saved benchmark examples: two estimates whose ranges contained the observed runs and two misses. This is the most direct demo path.

### Use case 2: add a pre-flight check to an application

A developer calls `predict_cost(...)` from Python or invokes `citadel-predict` from a terminal or CI job. The application can compare the high estimate with its own budget and decide whether to proceed, ask for approval, reduce scope, or select another model. The SDK supplies the prediction; budget policy and action remain the caller’s responsibility. LangGraph and CrewAI are integration examples in docs, not bundled runtime integrations.

### Use case 3: estimate from Claude through MCP

The MCP server exposes `estimate_agent_cost`. Claude Desktop or Claude Code launches the server as a local stdio process. The server forwards the task and tool list to the Citadel API and returns the estimate or a structured error.

### Use case 4: estimate while drafting in an editor

The VS Code-compatible extension provides a prompt panel, a tool selector, a debounced estimate, and a command to estimate selected editor text. The user can copy the prompt to another agent tool. The extension communicates with the same HTTP API.

## 4. Product scope and behavior as implemented

### Inputs

- Required natural-language task description.
- Optional list of tool names. The HTTP API accepts this list; the SDK and MCP signatures also expose `num_tools`, but the current SDK client does not convert that value into a tool list, so count-only input has no effect.
- Model calibration ID; defaults to `claude-sonnet`.

The API schema accepts task descriptions from 1 to 50,000 characters, up to 20 tools, and tool names up to 64 characters. Client documentation is inconsistent: the Python and MCP docs describe a 4,000-character limit, while the backend currently accepts up to 50,000. This should be resolved before setting a public limit.

### Prediction and response

The backend extracts ten features from task text and the supplied tool list: text length, tool count, action-keyword counts, explicit quantities, step connectors, rough clause count, and whether the prompt is a question. It returns:

- `low_tokens`, `expected_tokens`, and `high_tokens`;
- corresponding approximate USD costs when pricing is configured;
- `driving_factors` and extracted `features`;
- `confidence` (`high`, `moderate`, or `low`);
- `out_of_distribution` and `ood_reasons`.

The current model uses a Gradient Boosting Regressor to predict log token use and a residual-derived band to form the low/high range. The feature extractor counts the supplied tools; it does not interpret each tool’s schema, provider, output size, or behavior. Unknown tool names can still affect the estimate through their count.

### API surfaces

- `POST /api/predict` — estimate tokens and cost; bearer authentication is required only when the server is configured with a key or `CITADEL_REQUIRE_AUTH`.
- `GET /api/tools` — return the tool names and descriptions used by the web selector.
- `GET /api/validation-examples` — return the four precomputed benchmark comparison examples.
- `GET /api/health` and `GET /api/version` — return health/version and model metadata; both routes currently use the same handler.

The API is versioned as `1.0.0` in the FastAPI declaration. This is separate from the 0.2.0 Python package versions.

## 5. Model, data, and estimate limits

The checked-in baseline uses 20 hand-designed task archetypes and 80 observed agent runs, four repeats per task. Run records identify the collection model as `claude-sonnet-5`; the inference profile is named `claude-sonnet`. The model uses ten input features, although the project checkpoint reports that tool count and open-ended/narrow wording account for most measured feature importance in this small dataset.

The project checkpoint reports 80% overall leave-one-task-out coverage. Its reported coverage was 100% for narrow multi-step tasks, 75% for single-shot tasks, and 71% for open-ended tasks. The residual range was calibrated from the pooled leave-one-task-out residuals that were then used to report coverage; it is not a fully nested evaluation. The sample is small, and the range is empirical rather than a guarantee. The UI’s examples make two known failure modes visible: underestimation of an open-ended task and overestimation of a zero-tool question.

Cost estimates are less mature than token estimates. The pricing registry splits total tokens into an assumed 75% input / 25% output mix. Its `claude-sonnet` entry is dated 2025-01-01 and describes Claude 3.5 Sonnet pricing, while the training records identify Claude Sonnet 5. **The model-to-price mapping and current rates need confirmation before USD output is treated as current provider pricing.** Estimates are not invoices and do not include provider-specific discounts, caching, or account terms.

## 6. Non-goals and boundaries

The current product does not:

- execute agent tasks, call downstream tools, or monitor a run as it happens;
- guarantee a token ceiling, exact bill, or a particular interval coverage for a future task;
- automatically enforce budgets, request human approval, or route a task to another model;
- support calibrated profiles for models other than `claude-sonnet`;
- register user accounts, issue/manage API keys, bill customers, retain prediction history, or offer an admin console;
- learn from customer prediction requests or actual production run outcomes.

## 7. Security, privacy, and service behavior

Authentication is optional in the backend. If neither a key nor `CITADEL_REQUIRE_AUTH` is configured, prediction requests are open. When configured, the API accepts a bearer token. Health, tool-list, and validation-example routes do not depend on the prediction auth dependency. The repository does not provide a customer API-key issuance or rotation workflow, so teams requiring authentication must operate the service configuration themselves.

The service has SlowAPI rate limiting: `/api/predict` is configured at 30 requests per minute per client address; tool and validation-example routes are configured at 60 per minute. The limiter’s deployed storage/behavior has not been verified. The request middleware logs client IP, method, path, response status, and latency, but not the submitted task body.

The local web UI supports a `NEXT_PUBLIC_CITADEL_API_KEY` environment variable. Values with the `NEXT_PUBLIC_` prefix are sent to browser code and must not be treated as secret credentials for a public deployment. If a hosted UI is introduced, its auth design and CORS origin must be chosen accordingly. The API’s default allowed origin is only `http://localhost:3000`; a deployed browser UI needs an explicit `ALLOWED_ORIGINS` configuration.

## 8. Can another developer use it today?

**Yes, the repository provides usable developer interfaces.** The strongest ready-to-consume paths are the published Python package and MCP package. They can call the configured hosted URL if that service is reachable and permits the request, or a developer can run a self-hosted API from the source. The VSIX can be installed directly. The local web UI can be run against a local API.

What remains uncertain is the shared hosted service’s availability and access policy. The repository says it is deployed on Render and clients default to that URL, but a live health response could not be obtained from this workspace. A developer who needs reliability can deploy the backend with the included Dockerfile or the Render instructions in the root README. A clean-checkout setup guide should be made consistent: some existing docs refer to `data_collection/.venv` and Windows-specific activation paths, but that virtual environment is not tracked in the repository.

### Local developer quickstart

From the repository root, a developer can run the backend and UI with:

```bash
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

The frontend is then available at `http://localhost:3000`. For Python integration, install `citadel-predict`; for MCP, install `citadel-predict-mcp`. Set `CITADEL_API_URL` to a local or managed API URL if the default endpoint is unsuitable, and set `CITADEL_API_KEY` only when the API operator requires it.

## 9. Deployment and developer readiness

- **Backend deployment path:** documented Render setup and a Dockerfile that builds from the repository root. No deployment manifest or health-monitoring configuration is checked in.
- **Hosted backend:** documented at `https://citadel-7j9u.onrender.com`; current reachability is unknown. The environment used for this assessment could not resolve the hostname, so this is not evidence of a confirmed outage.
- **Frontend deployment:** not evidenced. The app defaults to a local API and no hosting configuration was found.
- **Installability:** Python SDK and MCP are published as PyPI releases; the editor extension has a committed VSIX and documented local build/install steps.
- **Operational readiness:** no published uptime target, support path, usage dashboard, key-management portal, or API service-level objective was found.

## 10. Current gaps and improvement candidates

These are evidence-based candidates to prioritize after confirming the desired product direction:

1. **Confirm and document the hosted service.** Check its health and auth setting from a normal external network; record the supported URL, whether anonymous access is allowed, and how developers get a key if not.
2. **Correct the pricing/model mapping.** Verify which exact Sonnet version the model represents and update its live pricing data and date, or make the USD estimate unavailable until that mapping is sound.
3. **Improve calibration evidence.** Add representative tasks and real-world runs, especially open-ended and zero-tool cases; report evaluation using a genuinely held-out or nested method and publish coverage by task family.
4. **Make API contracts consistent.** Align the 4,000 vs 50,000 character limits, make `num_tools` work or remove it from the SDK/MCP interface, only advertise registered model profiles, and synchronize SDK, MCP, extension, and API documentation.
5. **Make fresh-checkout setup reliable.** Provide one platform-neutral development guide, define a supported Python/Node version, and retain the actual tested setup commands.
6. **Decide whether a hosted web app and public extension listing are goals.** The code has the surfaces, but repository evidence currently supports local UI and direct VSIX installation rather than a discoverable public web/marketplace release.

No user research, adoption metrics, business model, service-level objective, or production usage data is recorded in the repository. These should be treated as unknowns rather than implied requirements.

## 11. Capability handoff

### Capability

Given a proposed AI-agent task and its available tool count, Citadel provides a pre-execution estimate of token use and approximate model cost, with an empirical range and warnings to help a developer make a budget decision.

### Fixed constraints in the current product

- Only the `claude-sonnet` calibration profile is implemented.
- The inference request is based on task text and tool count; it does not simulate an agent or inspect tool behavior.
- Estimates are advisory, empirical, and trained on a small benchmark.
- Callers own budget thresholds and actions.

### Implementation contract

The HTTP API is the shared inference interface. The web app, Python SDK/CLI, MCP server, and VS Code-compatible extension are clients of that API. Prediction output includes the estimate, approximate costs, confidence/OOD information, and explanation fields. The backend loads the model artifact at startup and serves requests without persistent user data storage.

### Open decisions

- Is the Render service intended to be publicly usable, authenticated, or only a demo?
- What exact provider/model release should `claude-sonnet` refer to, and what price schedule should be applied?
- Is a public hosted dashboard part of the product, or is the local web UI sufficient?
- Which models and developer environments should be officially supported next?
- What prediction accuracy, API availability, and latency targets would justify production use?

### Handoff

This PRD is ready to guide documentation cleanup and a focused readiness review. Before expanding features or making reliability promises, verify the hosted deployment and settle the Sonnet/pricing mapping. The improvement candidates above provide the next set of product decisions and implementation slices.

## 12. Source of truth

- [Root README](README.md) — product positioning, local run instructions, and Render deployment claim.
- [Backend API](backend/main.py) — endpoints, schema bounds, authentication, CORS, and rate limits.
- [Prediction model](data_collection/predict.py) and [feature extraction](data_collection/features.py) — supported model and current estimation logic.
- [Pricing registry](data_collection/pricing.py) — USD conversion rates and assumptions.
- [Validation examples](data_collection/data/validation_examples.json) and [project checkpoint](CHECKPOINT.md) — benchmark records and reported model results.
- [Python SDK](packages/citadel-predict/README.md), [MCP package](packages/citadel-predict-mcp/README.md), and [editor extension](packages/citadel-predict-vscode/README.md) — developer workflows and documented package behavior.
