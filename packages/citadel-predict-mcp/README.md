# citadel-predict-mcp

[![Python Versions](https://img.shields.io/pypi/pyversions/citadel-predict-mcp.svg)](https://pypi.org/project/citadel-predict-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Model Context Protocol (MCP) server for Citadel Predict pre-execution AI agent cost estimation.**

`citadel-predict-mcp` connects your hosted [Citadel Predict API](https://github.com/Athullvr/Citadel) directly into **Claude Desktop** and **Claude Code** via a local standard I/O (stdio) MCP server.

With this server configured, Claude can natively estimate token usage ranges ($low, expected, high$) and flag out-of-distribution risks for autonomous workflows **before running them**—without requiring manual CLI execution.

---

## Features

- ⚡ **Native Claude Tool Calling**: Claude automatically decides when to call `estimate_agent_cost` when planning or dispatching tasks.
- 🔒 **Zero Network Exposure**: Runs strictly as a local `stdio` subprocess spawned by Claude Desktop / Claude Code.
- 🎯 **Pre-Execution Guardrails**: Predicts token consumption bounds before multi-step tools or reasoning loops execute.
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
