/**
 * WebviewViewProvider for Citadel Predict sidebar panel.
 */

import * as vscode from 'vscode';
import { CitadelApiError, CitadelClient } from './citadelClient';
import { ExtensionConfig, HostToWebviewMessage, WebviewPersistedState, WebviewToHostMessage } from './types';

export class CitadelViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = 'citadelPredict.costView';

  private _view?: vscode.WebviewView;
  private readonly _extensionUri: vscode.Uri;
  private _client: CitadelClient;
  private _config: ExtensionConfig;
  private _context: vscode.ExtensionContext;

  constructor(
    extensionUri: vscode.Uri,
    context: vscode.ExtensionContext,
    config: ExtensionConfig
  ) {
    this._extensionUri = extensionUri;
    this._context = context;
    this._config = config;
    this._client = new CitadelClient(config);
  }

  public updateConfig(config: ExtensionConfig): void {
    this._config = config;
    this._client.updateConfig(config);

    if (this._view) {
      const msg: HostToWebviewMessage = {
        type: 'configUpdate',
        config: {
          apiUrl: config.apiUrl,
          defaultModel: config.defaultModel,
          debounceDelayMs: config.debounceDelayMs,
        },
      };
      this._view.webview.postMessage(msg);
    }
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(async (data: WebviewToHostMessage) => {
      switch (data.type) {
        case 'predict': {
          await this._handlePrediction(data.taskText, data.tools, data.modelId);
          break;
        }

        case 'copyPrompt': {
          if (data.text) {
            await vscode.env.clipboard.writeText(data.text);
            vscode.window.setStatusBarMessage('$(check) Prompt copied to clipboard!', 2500);
          }
          break;
        }

        case 'saveState': {
          if (data.state) {
            await this._context.workspaceState.update('citadelPredict.lastState', data.state);
          }
          break;
        }

        case 'openSettings': {
          await vscode.commands.executeCommand(
            'workbench.action.openSettings',
            'citadelPredict'
          );
          break;
        }

        case 'requestInitialState': {
          const lastState = this._context.workspaceState.get<WebviewPersistedState>(
            'citadelPredict.lastState'
          );
          if (lastState && lastState.prompt) {
            this._postMessage({
              type: 'setPrompt',
              text: lastState.prompt,
            });
          }
          this._postMessage({
            type: 'configUpdate',
            config: {
              apiUrl: this._config.apiUrl,
              defaultModel: this._config.defaultModel,
              debounceDelayMs: this._config.debounceDelayMs,
            },
          });
          break;
        }
      }
    });
  }

  public setPrompt(text: string): void {
    if (this._view) {
      this._view.show?.(true);
      this._postMessage({
        type: 'setPrompt',
        text: text,
      });
    }
  }

  public clearPrompt(): void {
    if (this._view) {
      this._postMessage({
        type: 'setPrompt',
        text: '',
      });
    }
  }

  private async _handlePrediction(
    taskText: string,
    tools: string[],
    modelId: string
  ): Promise<void> {
    if (!this._view) {
      return;
    }

    this._postMessage({ type: 'predictionLoading' });

    try {
      const result = await this._client.predict({
        task_text: taskText,
        tools: tools,
        model_id: modelId || this._config.defaultModel || 'claude-sonnet',
      });

      this._postMessage({
        type: 'predictionSuccess',
        data: result,
        taskText: taskText,
      });
    } catch (err: unknown) {
      if (err instanceof CitadelApiError) {
        this._postMessage({
          type: 'predictionError',
          error: err.message,
          isUncalibratedModel: err.isUncalibratedModel,
          uncalibratedModelId: err.uncalibratedModelId || modelId,
          isNetworkError: err.isNetworkError,
          isAuthError: err.isAuthError,
          isRateLimited: err.isRateLimited,
          retryAfter: err.retryAfter,
        });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        this._postMessage({
          type: 'predictionError',
          error: message,
        });
      }
    }
  }

  private _postMessage(message: HostToWebviewMessage): void {
    if (this._view) {
      this._view.webview.postMessage(message);
    }
  }

  private _getHtmlForWebview(webview: vscode.Webview): string {
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'webview.css')
    );
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'webview.js')
    );
    const iconUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'citadel-icon.svg')
    );

    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} https: data:;">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${styleUri}" rel="stylesheet">
  <title>Citadel Predict</title>
</head>
<body>
  <!-- Header Bar -->
  <header class="app-header">
    <div class="brand-container">
      <img src="${iconUri}" class="brand-icon" alt="Citadel Predict Logo" />
      <span class="brand-title">Citadel Predict</span>
    </div>
    <div class="header-status">
      <div id="status-dot" class="status-dot" title="Ready"></div>
    </div>
  </header>

  <!-- Prompt Input Section -->
  <section class="section">
    <div class="section-label-row">
      <label for="prompt-input" class="section-label">Agent Prompt Draft</label>
      <div class="quick-links">
        <button type="button" class="link-btn" id="clear-prompt-btn">Clear</button>
      </div>
    </div>
    <div class="textarea-container">
      <textarea
        id="prompt-input"
        class="prompt-textarea"
        placeholder="Type or paste your agent prompt here... (e.g. 'Search for recent sales figures, fetch the report, and draft a summary document.')"
        spellcheck="false"
      ></textarea>
    </div>
    <div class="input-meta-bar">
      <span id="char-counter" class="char-counter">0 chars</span>
      <select id="samples-select" class="link-btn" style="background: transparent; border: none; cursor: pointer;">
        <option value="">Insert Sample...</option>
      </select>
    </div>
  </section>

  <!-- Model Selection Section -->
  <section class="section">
    <div class="section-label-row">
      <label for="model-select" class="section-label">Model Target</label>
    </div>
    <select id="model-select" class="select-input">
      <option value="claude-sonnet">Claude 3.5 Sonnet (Calibrated ✓)</option>
      <option value="gemini-pro">Gemini 1.5 Pro (Coming soon)</option>
      <option value="gpt-4o">GPT-4o (Coming soon)</option>
    </select>
  </section>

  <!-- Tools Selection Section -->
  <section class="section">
    <div class="section-label-row">
      <span class="section-label">Available Agent Tools</span>
      <div class="tools-actions">
        <button type="button" class="link-btn" id="select-all-tools">All</button>
        <span style="opacity: 0.3;">|</span>
        <button type="button" class="link-btn" id="clear-tools">None</button>
      </div>
    </div>
    <div id="tools-container" class="tools-grid">
      <!-- Tool chips rendered by JS -->
    </div>
  </section>

  <!-- Dynamic Results & Error Area -->
  <div id="error-container" style="display: none;"></div>

  <div id="empty-state" class="empty-state">
    <div class="empty-state-icon">⚡</div>
    <div class="empty-state-text">
      Start typing an agent prompt above to see real-time token budget and dollar cost estimates.
    </div>
  </div>

  <div id="result-card" class="result-card" style="display: none;">
    <div class="result-header">
      <span class="result-title">Expected Budget & Cost</span>
      <span id="confidence-badge" class="confidence-badge high">HIGH</span>
    </div>

    <div class="primary-metric-box">
      <div class="primary-metric-label">Predicted Run Cost</div>
      <div class="primary-metric-value-row">
        <span id="expected-tokens" class="expected-tokens">0 tokens</span>
        <span id="expected-cost" class="expected-cost-badge">~$0.00</span>
      </div>
    </div>

    <div class="metric-row">
      <span class="metric-name">Expected Range (Tokens)</span>
      <span id="range-tokens" class="metric-val">0 – 0</span>
    </div>

    <div class="metric-row" id="cost-range-row">
      <span class="metric-name">Expected Range (USD)</span>
      <span id="range-cost" class="metric-val">~$0.00 – $0.00</span>
    </div>

    <div id="ood-container" style="display: none;"></div>

    <div id="factors-container" class="factors-section" style="display: none;">
      <div class="factors-title">Key Cost Factors</div>
      <ul id="factors-list" class="factors-list"></ul>
    </div>

    <button type="button" id="copy-prompt-btn" class="btn-primary">
      <span>Copy Prompt to Clipboard</span>
    </button>
  </div>

  <footer class="footer-tip">
    💡 Refine your prompt before sending to Antigravity, Claude Code, or Cursor.
  </footer>

  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}

function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
