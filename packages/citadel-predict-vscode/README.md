# Citadel Predict — Live Prompt Cost (VS Code / Antigravity IDE)

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Athullvr/Citadel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight companion sidebar extension for **VS Code**, **Google Antigravity IDE**, **Cursor**, and other VS Code–compatible editors.

Draft your agent prompts and see a **live, debounced token budget and USD cost estimate** update in real time as you type — before pasting or sending the prompt to Antigravity's agent chat, Claude Code, or any other agent tool.

---

## 🚀 Features

- **Live Debounced Cost Forecasting**: Computes expected tokens and dollar cost (~$0.XX) within 400ms of pausing typing — zero keystroke lag.
- **Token & USD Estimate**: Shows an empirical token range and estimated cost. The range is descriptive, not an independently validated confidence interval.
- **Interactive Tool Selector**: Toggle agent tool chips (`web_search`, `fetch_url`, `read_document`, `list_files`, `calculator`, `draft_document`, `bash`, `python_repl`, etc.) to see how tool access impacts execution cost.
- **Legacy Claude Estimator**: Select the `claude-sonnet` alias. The frozen estimator has no verified exact provider-model identity and is not exact-model calibrated. Registered OpenAI and Google models are unavailable for prediction.
- **Heuristic Confidence & OOD Alerts**: Shows a heuristic confidence label (`HIGH` / `MEDIUM` / `LOW`) and out-of-distribution (OOD) warnings; the label is not a calibrated probability.
- **Editor Selection Shortcut**: Highlight any text in your editor, right-click, and select **"Citadel Predict: Estimate Selected Prompt"**.
- **One-Click Copy**: "Copy Prompt to Clipboard" button with instant feedback to paste into your favorite agent harness.
- **Native VS Code Theming**: 100% compliant with VS Code dark, light, and high-contrast theme tokens.

---

## 🖼️ User Interface

```
+-----------------------------------------------------------+
| 🛡️ CITADEL PREDICT                            [ ● Active ] |
+-----------------------------------------------------------+
| AGENT PROMPT DRAFT                               [ Clear ]|
| +-------------------------------------------------------+ |
| | Search the web for top 5 AI benchmarks, fetch details | |
| | on each, and draft a structured comparison report.    | |
| +-------------------------------------------------------+ |
| 112 chars                             [ Insert Sample... ▾]|
|                                                           |
| MODEL TARGET                                              |
| [ Claude Sonnet (legacy estimator; exact model unresolved)▾]|
|                                                           |
| AVAILABLE AGENT TOOLS                    [ All | None ]   |
| [✓ web_search] [✓ fetch_url] [✓ draft_document]           |
| [+ list_files] [+ calculator] [+ bash] [+ python_repl]    |
|                                                           |
| +-------------------------------------------------------+ |
| | EXPECTED BUDGET & COST                         [ HIGH]| |
| | +---------------------------------------------------+ | |
| | | PREDICTED RUN COST                                | | |
| | | 42,500 tokens                         (~$0.34)    | | |
| | +---------------------------------------------------+ | |
| | Empirical Range (Tokens):  18,200 – 68,000            | |
| | Estimated Range (USD):    ~$0.15 – $0.54              | |
| |                                                       | |
| | KEY COST FACTORS                                      | |
| | • Web search results accumulation                     | |
| | • Multi-page document synthesis                       | |
| |                                                       | |
| | [ 📋 Copy Prompt to Clipboard                       ] | |
| +-------------------------------------------------------+ |
+-----------------------------------------------------------+
```

---

## 📦 Installation & Setup

### For Local Development / Extension Host

1. Navigate to the extension folder:
   ```bash
   cd packages/citadel-predict-vscode
   ```
2. Install dependencies and compile TypeScript:
   ```bash
   npm install
   npm run compile
   ```
3. Open the directory in VS Code and press `F5` to launch a new **Extension Development Host** window.
4. Click the **Citadel Predict** shield icon in the Activity Bar sidebar.

### Building the `.vsix` Package

```bash
npm run package
# Generates citadel-predict-vscode-0.1.0.vsix
```

Install directly via VS Code CLI:
```bash
code --install-extension citadel-predict-vscode-0.1.0.vsix
```

---

## ⚙️ Configuration Settings

Configure via **VS Code Settings** (`Ctrl+,` or `Cmd+,` -> search `Citadel Predict`):

| Setting Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `citadelPredict.apiUrl` | `string` | `https://citadel-7j9u.onrender.com` | Base URL of the Citadel Predict API |
| `citadelPredict.defaultModel` | `string` | `claude-sonnet` | Legacy estimator alias; its exact provider model is unresolved |
| `citadelPredict.apiKey` | `string` | `""` | Optional Bearer API key (`CITADEL_API_KEY`) |
| `citadelPredict.debounceDelayMs` | `number` | `400` | Typing debounce delay in milliseconds |
| `citadelPredict.timeoutSeconds` | `number` | `10` | HTTP request timeout in seconds |

---

## 🔌 API Integration Details

The extension communicates directly with Citadel Predict's `/api/predict` endpoint:

- **Request**:
  ```json
  POST /api/predict
  Content-Type: application/json
  {
    "task_text": "Draft an analysis of quarterly metrics",
    "tools": ["read_document", "calculator"],
    "model_id": "claude-sonnet"
  }
  ```
- **Response**:
  ```json
  {
    "model_id": "claude-sonnet",
    "low_tokens": 12000,
    "expected_tokens": 28500,
    "high_tokens": 49000,
    "expected_cost_usd": 0.23,
    "low_cost_usd": 0.10,
    "high_cost_usd": 0.39,
    "driving_factors": ["Context accumulation from file reads"],
    "confidence": "high",
    "out_of_distribution": false,
    "ood_reasons": []
  }
  ```

---

## 🧪 Testing

Run the automated test suite:
```bash
npm test
```

---

## 📄 License

MIT © Citadel Contributors
