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

1. **Single-Model Calibration**: Calibration is currently tuned specifically for **Claude Sonnet** (`claude-sonnet`). Future releases will introduce multi-model support via `model_id`.
2. **Calibration Dataset Scale**: Calibrated on $N=20$ diverse task archetypes across 80 benchmarked runs.
3. **Synthetic Tool Sizing**: Ground-truth data was collected using deterministic mock tool outputs with representative context expansion. Real-world tools with unbounded payload returns (e.g., massive scraped DOMs) may exhibit higher variance.
4. **Pre-execution Estimation**: Token predictions represent calibrated statistical ranges $[low, expected, high]$, not runtime guarantees against infinite loops or divergent agent reasoning.

---

## License

MIT
