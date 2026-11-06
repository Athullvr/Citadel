# citadel-predict-mcp

[![Python Versions](https://img.shields.io/pypi/pyversions/citadel-predict-mcp.svg)](https://pypi.org/project/citadel-predict-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Model Context Protocol (MCP) server for Citadel prediction and explicit Anthropic execution tracking.**

`citadel-predict-mcp` connects your hosted [Citadel Predict API](https://github.com/Athullvr/Citadel) directly into **Claude Desktop** and **Claude Code** via a local standard I/O (stdio) MCP server.

With this server configured, Claude can natively estimate token usage ranges ($low, expected, high$) and flag out-of-distribution risks for autonomous workflows **before running them**—without requiring manual CLI execution.

---

## Features

- ⚡ **Native Claude Tool Calling**: Claude can estimate a task and coordinate an explicit execution lifecycle.
- 🔒 **Zero Network Exposure**: Runs strictly as a local `stdio` subprocess spawned by Claude Desktop / Claude Code.
- 🎯 **Pre-Execution Guardrails**: Predicts token consumption bounds before multi-step tools or reasoning loops execute.
- 📊 **Execution Feedback**: Registers runs, records idempotent usage events, and reports completion and telemetry completeness.
- 🛡️ **User-Friendly Error Handling**: Catches authentication, rate-limiting, and validation issues, presenting clear, actionable suggestions to Claude rather than raw stack traces.

---

## Installation

Install the package via `pip`:

```bash
pip install citadel-predict-mcp
```

*(For local development from the repository root: `pip install -e packages/citadel-predict-mcp`)*

Verifying installation:
```bash
citadel-predict-mcp --help
```

---

## Quickstart: One-Command Auto-Configuration (Recommended)

To automatically configure **both** Claude Desktop and Claude Code on Windows, macOS, or Linux with zero manual JSON editing:

```bash
# Run from repository root with your active Python environment:
python scripts/configure_mcp.py

# Or pass parameters non-interactively:
python scripts/configure_mcp.py --api-key cp_live_your_key_here --api-url http://localhost:8000
```

The script automatically detects your active virtual environment, locates the platform-specific executable (`citadel-predict-mcp.exe` on Windows or `citadel-predict-mcp` on macOS/Linux), and cleanly merges the server definition into both `claude_desktop_config.json` and `.claude/settings.json` while preserving all existing configurations.

---

## Manual Configuration (Alternative)

If you prefer to configure manually:

### 1. Claude Desktop (`claude_desktop_config.json`)

| Operating System | Exact Configuration File Path |
| :--- | :--- |
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |
| **Linux** | `~/.config/Claude/claude_desktop_config.json` |

Add `citadel-predict` under `mcpServers`:

```json
{
  "mcpServers": {
    "citadel-predict": {
      "command": "citadel-predict-mcp",
      "env": {
        "CITADEL_API_KEY": "cp_live_your_api_key_here",
        "CITADEL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 2. Claude Code (`.claude/settings.json`)

```bash
claude mcp add citadel-predict -- citadel-predict-mcp
```

---

## Available Tools

### `estimate_agent_cost`

**Description**:
> *Estimate token cost and usage range for an AI agent task BEFORE running it. Use this when the user is about to execute, dispatch, or run a multi-step agent task and cost/budget matters.*

**Input Parameters**:
- `task_text` (*string, required*): The natural language description of the agent task (1 to 4000 characters).
- `tools` (*array of strings, optional*): List of tool names available to the agent (e.g. `["web_search", "draft_document"]`).
- `num_tools` (*integer, optional*): Tool count if specific tool names are not listed.
- `model_id` (*string, optional, default: `"claude-sonnet"`*): Model calibration profile to evaluate against.

**Sample Return Payload**:
```json
{
  "success": true,
  "model_id": "claude-sonnet",
  "expected_tokens": 3200,
  "low_tokens": 1500,
  "high_tokens": 5800,
  "out_of_distribution": false,
  "ood_reasons": [],
  "confidence": "normal",
  "driving_factors": ["task_length", "tools_count"],
  "summary": "Expected: 3,200 tokens (Range: 1,500 – 5,800)"
}
```

---

## Explicit execution lifecycle

The execution tools use Citadel's authenticated execution feedback API and the
server-selected workspace. They currently accept Anthropic usage semantics.
Usage submitted through this MCP server is classified as
`unverified_submitted`; the MCP caller's event data is not authenticated as a
provider-signed receipt. Citadel never stores the raw task text as execution
telemetry, although `begin_execution` sends it to the existing prediction API
to create the prediction snapshot.

Pass Anthropic's raw `input_tokens` and `output_tokens` counters. In the current
Citadel evaluator, Anthropic `input_tokens` is the cache-exclusive input
component; cache creation and cache read counters are parallel breakdowns and
are not subtracted from that value again. Citadel only reports a cache-inclusive
aggregate when both optional cache counters are present for every event.

### `begin_execution`

Provide exactly one of `task_text` or `prediction_id`. Supply an exact provider
model only when the provider has resolved one for this execution. The current
Claude estimator cannot establish that identity, so omit it for this legacy
prediction. With `task_text`, Citadel creates a persisted prediction through
the existing estimator before registering the execution. The response contains the
`execution_id` and a safe prediction summary. The tool does not return a
workspace ID, task hash, prompt, or feature snapshot.

```json
{
  "task_text": "Review the release notes and draft a migration checklist.",
  "tools": ["read_release_notes", "draft_checklist"],
  "model_id": "claude-sonnet",
  "provider": "anthropic",
  "integration_name": "my-agent",
  "integration_version": "1.4.0",
  "use_adaptive_profile": true
}
```

`use_adaptive_profile` is optional and defaults to `false`. Set it to `true`
with a new `task_text` to request an eligible workspace profile. The returned
`adaptive_prediction` includes the unchanged model estimate, the adjusted token
interval, profile level, sample counts, provenance and fallback reason. When
history is insufficient or unverified, the original estimate is used. This
option cannot be combined with an existing `prediction_id`. The current
legacy-unresolved Claude identity cannot activate an exact-model profile.

### `record_execution_usage`

Append one usage report while the execution is started. Use a stable UUID
`event_id` for each provider response and reuse that same ID on retries. The
required input and output counters are non-negative integers. Cache counters
may be `null` when the provider response omitted them; do not substitute zero.
The `source` field is set internally by the MCP SDK and is not accepted from
the caller.

```json
{
  "execution_id": "<execution_id from begin_execution>",
  "event_id": "c115e9ac-d80b-4e31-b497-96cc1a85a8b0",
  "provider": "anthropic",
  "exact_provider_model": null,
  "input_tokens": 1200,
  "output_tokens": 240,
  "cache_creation_input_tokens": 300,
  "cache_read_input_tokens": null
}
```

This event remains caller-submitted and unverified. The API checks its
provider/model identity against the execution and does not let the event replace
the execution identity. When a provider does not report an exact model, leave
the field `null`; such telemetry is not eligible for exact-model calibration.

### `complete_execution` and `get_execution_summary`

Complete an execution once with `status` equal to `completed`, `failed`, or
`cancelled`. If the integration knows how many unique usage events it expected,
pass that count as `expected_usage_event_count`; equality with stored events is
reported as `complete`, a mismatch as `incomplete`, and omission as `unknown`.
Event-count equality describes the caller's declaration and does not prove the
telemetry is truthful. `get_execution_summary` can be called at any time to
retrieve status, safe prediction details, recorded aggregate counters, and
completeness.

```json
{
  "execution_id": "<execution_id from begin_execution>",
  "status": "completed",
  "expected_usage_event_count": 1
}
```

Then read the saved record:

```json
{
  "execution_id": "<execution_id from begin_execution>"
}
```

### Full client workflow

1. Call `begin_execution` before starting the agent. Save the returned
   `execution_id`.
2. Run the task using the selected provider. For each Anthropic response, call
   `record_execution_usage` with its counters and a stable unique event ID.
3. Call `complete_execution` with the terminal result and known event count.
   Omit the count when completeness is unknown.
4. Call `get_execution_summary` to read the final status, aggregate, and
   completeness. The record can then be considered by Citadel's evaluation API
   when its prediction, telemetry, and evaluation eligibility requirements are
   met.

The tools record explicit integration feedback. They do not intercept agent
traffic automatically, and they do not support universal provider token
semantics.

---

## Manual Test Checklist for Testers

Follow this 5-minute checklist to verify your MCP setup:

1. **Installation**:
   - [ ] Run `citadel-predict-mcp --help` in your terminal to verify the command is accessible on your PATH.
2. **Configuration**:
   - [ ] Add the JSON entry to `claude_desktop_config.json` with your active Citadel API key.
3. **Restart**:
   - [ ] Fully quit and reopen Claude Desktop.
4. **Invocation Test**:
   - [ ] Send the following prompt in a new Claude Desktop chat:
     > *"I am planning to have an agent research competitor pricing across 5 company sites and compile a markdown report. Estimate the token cost and budget range before we start."*
5. **Verification**:
   - [ ] Confirm Claude invokes the `estimate_agent_cost` tool (indicated by a tool-call widget in the conversation).
   - [ ] Confirm Claude receives the token ranges (`expected_tokens`, `low_tokens`, `high_tokens`) and presents a natural language summary with the budget estimate to you.

---

## License

MIT
