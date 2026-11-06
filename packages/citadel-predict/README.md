# citadel-predict

[![PyPI Version](https://img.shields.io/pypi/v/citadel-predict.svg)](https://pypi.org/project/citadel-predict/)
[![Python Versions](https://img.shields.io/pypi/pyversions/citadel-predict.svg)](https://pypi.org/project/citadel-predict/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Pre-execution token budget and cost predictor client for AI agents.**

`citadel-predict` is a lightweight, pure HTTP Python client and CLI for the Citadel Predict API. It allows developers, CI pipelines, and autonomous agent loops to estimate LLM token consumption and cost ranges *before* initiating expensive agent runs.

---

## Installation

```bash
pip install citadel-predict
```

---

## Quickstart (3 Lines of Code)

```python
from citadel_predict import predict_cost

result = predict_cost(
    task_text="Research competitor pricing across 3 sources and draft report",
    tools=["web_search", "draft_document"]
)

print(f"Expected: {result['expected_tokens']:,} tokens (Range: {result['low_tokens']:,} – {result['high_tokens']:,})")
```

Output:
```text
Expected: 3,200 tokens (Range: 1,500 – 5,800)
```

## Adaptive execution profile (opt-in)

When execution feedback is configured, the SDK can create a prediction with a
workspace-scoped historical adjustment and then start that execution:

```python
from citadel_predict import CitadelClient

with CitadelClient() as client:
    execution = client.begin_execution(
        task_text="Review the release notes and draft a migration checklist.",
        tools=["read_release_notes", "draft_checklist"],
        provider="anthropic",
        use_adaptive_profile=True,
    )

profiled = execution["adaptive_prediction"]
print(profiled["original_prediction"]["expected_tokens"])
print(profiled["adjusted_prediction"]["expected_tokens"])
print(profiled["profile"]["status"], profiled["profile"]["profile_level"])
```

Adaptive profiles are opt-in and require a new `task_text`; attaching an
existing prediction reference keeps its saved estimate unchanged. The
`adjusted_prediction` contains token bounds only. Cost fields remain the
original predictor's estimates because total-token residuals do not reveal the
input/output split. If there is not enough verified, recent, task-deduplicated
history, the SDK returns the original estimate with a profile status and
fallback reason. The current frozen Claude estimator has no verified exact
provider-model identity. Its prediction is marked `legacy_unresolved`, so it
cannot activate an exact-model adaptive profile. Caller-submitted telemetry
remains unverified.

---

## CLI Usage

`citadel-predict` includes a full-featured CLI for terminal workflows and CI/CD cost checks:

```bash
# Pretty terminal card output
citadel-predict --task "Audit repository and write migration guide" --tools list_files,read_document,draft_document

# Scripting / CI mode (JSON output)
citadel-predict --task "Calculate statistical metrics" --tools calculator --json

# Override API key or URL
citadel-predict --task "..." --api-key "cp_live_12345" --api-url "https://api.citadel.dev"
```

### CLI Exit Codes
- `0`: Success
- `2`: Validation Error / Bad Request (HTTP 400 / 422 or missing task)
- `3`: Authentication Failure (HTTP 401)
- `4`: Rate Limit Exceeded (HTTP 429)
- `5`: Server Error (HTTP 5xx)
- `6`: Network / Timeout Error

---

## Real Agent Integration: Pre-Execution Guardrails

Existing agent governance tools (e.g., Portkey, Langfuse, LiteLLM) are **reactive**—they record costs during or after an execution. `citadel-predict` is **predictive**—enabling pre-flight budget checks and dynamic routing before running reasoning loops.

### LangGraph / CrewAI Pre-Flight Cost Guardrail Example

```python
from typing import TypedDict, List
from citadel_predict import predict_cost, CitadelError

class AgentState(TypedDict):
    task: str
    tools: List[str]
    budget_tokens: int
    approved: bool

def pre_flight_budget_guardrail(state: AgentState) -> AgentState:
    """
    Evaluates token budget before dispatching tools or multi-agent loops.
    """
    try:
        prediction = predict_cost(
            task_text=state["task"],
            tools=state["tools"],
            model_id="claude-sonnet"
        )
    except CitadelError as e:
        print(f"Cost prediction unavailable: {e}. Falling back to default budget.")
        return state

    expected = prediction["expected_tokens"]
    high = prediction["high_tokens"]
    is_ood = prediction["out_of_distribution"]

    print(f"Pre-flight estimate: ~{expected:,} tokens (Upper bound: {high:,})")
    if is_ood:
        print(f"Warning: Out-of-Distribution task ({prediction['ood_reasons']})")

    # Guardrail Policy: Escalate if upper bound exceeds budget
    if high > state["budget_tokens"]:
        print(f"[BLOCKED] High-estimate ({high:,}) exceeds budget ({state['budget_tokens']:,})")
        # In a real agent: switch to smaller model, ask human for approval, or prune tool access
        state["approved"] = False
    else:
        state["approved"] = True

    return state

# Example usage in workflow
initial_state: AgentState = {
    "task": "Perform exhaustive market research across 20 industry filings",
    "tools": ["web_search", "fetch_url", "draft_document"],
    "budget_tokens": 10000,
    "approved": False
}

state = pre_flight_budget_guardrail(initial_state)
if not state["approved"]:
    print("Action required: Human-in-the-loop approval or task reformulation needed.")
```

---

## Authentication & Configuration

The client resolves your API key and base URL according to the following priority:

1. **Explicit argument**: `predict_cost(..., api_key="...", api_url="...")` or CLI `--api-key` / `--api-url`
2. **Environment variables**: `CITADEL_API_KEY` and `CITADEL_API_URL`
3. **Configuration file**: `~/.citadel/config.toml`

### Example `~/.citadel/config.toml`
```toml
api_key = "cp_live_your_api_key_here"
api_url = "https://api.citadel.dev"
```

---

## Error Handling

`citadel-predict` surfaces typed, catchable exceptions:

```python
from citadel_predict import (
    predict_cost,
    CitadelAuthError,
    CitadelRateLimitError,
    CitadelValidationError,
    CitadelServerError,
    CitadelNetworkError,
)

try:
    result = predict_cost("Analyze dataset", tools=["calculator"])
except CitadelAuthError:
    # 401: Missing or invalid API key
    ...
except CitadelRateLimitError as e:
    # 429: Rate limited; check e.retry_after
    print(f"Retry after {e.retry_after} seconds")
except CitadelValidationError as e:
    # 422: Input validation bounds exceeded (e.g. task > 4000 chars)
    ...
except CitadelNetworkError as e:
    # Timeout or connection failure
    ...
```

---

## Honest Limitations

`citadel-predict` is a thin client wrapping the hosted calibration model. It directly inherits the current system characteristics:

1. **Legacy estimator identity**: `claude-sonnet` selects the frozen estimator, but the training records contain only the configured request string and the artifact has no exact provider-model metadata. Citadel does not claim exact-model calibration.
2. **Validation scope**: Point estimates use leave-one-task-out validation over 20 task archetypes and 80 repeated runs. The pooled residual band is descriptive and is not an independently validated interval.
3. **Synthetic Tool Sizing**: Ground-truth data was collected using deterministic mock tool outputs with representative context expansion. Real-world tools with unbounded payload returns (e.g., massive scraped DOMs) may exhibit higher variance.
4. **Pre-execution Estimation**: Token predictions include an empirical range, not a calibrated probability interval or runtime guarantee against infinite loops or divergent agent reasoning. Confidence labels are heuristic.

---

## License

MIT
