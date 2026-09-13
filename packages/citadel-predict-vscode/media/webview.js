// Citadel Predict Webview Client Script
(function () {
  // @ts-ignore
  const vscode = acquireVsCodeApi();

  // Standard tool list matching Citadel Predict schemas
  const DEFAULT_TOOLS = [
    'web_search',
    'fetch_url',
    'read_document',
    'list_files',
    'calculator',
    'send_email',
    'draft_document',
    'file_editor',
    'bash',
    'python_repl'
  ];

  // Sample prompt templates for quick test
  const SAMPLE_PROMPTS = [
    {
      title: 'Competitive Research',
      text: 'Search the web for top 5 AI agent evaluation benchmarks, fetch details on each methodology, and draft a structured markdown comparison report.',
      tools: ['web_search', 'fetch_url', 'draft_document']
    },
    {
      title: 'Codebase Refactoring',
      text: 'List files in the src directory, read all configuration handlers, refactor the error handling hierarchy across all modules, and run the test suite.',
      tools: ['list_files', 'read_document', 'file_editor', 'bash']
    },
    {
      title: 'Data Analysis & Email',
      text: 'Read quarterly metrics from data.csv, evaluate profit margins using the calculator, and email an executive brief to stakeholders.',
      tools: ['read_document', 'calculator', 'send_email']
    }
  ];

  // State
  let state = vscode.getState() || {
    prompt: '',
    selectedTools: ['web_search', 'fetch_url', 'draft_document'],
    modelId: 'claude-sonnet',
    lastResult: null
  };

  let debounceTimer = null;
  let debounceDelayMs = 400;

  // DOM Elements
  const promptInput = document.getElementById('prompt-input');
  const charCounter = document.getElementById('char-counter');
  const modelSelect = document.getElementById('model-select');
  const toolsContainer = document.getElementById('tools-container');
  const selectAllToolsBtn = document.getElementById('select-all-tools');
  const clearToolsBtn = document.getElementById('clear-tools');
  const samplesSelect = document.getElementById('samples-select');
  const clearPromptBtn = document.getElementById('clear-prompt-btn');
  const copyPromptBtn = document.getElementById('copy-prompt-btn');
  const statusDot = document.getElementById('status-dot');
  const resultCard = document.getElementById('result-card');
  const emptyState = document.getElementById('empty-state');
  const errorContainer = document.getElementById('error-container');

  // Result Card DOM fields
  const expectedTokensEl = document.getElementById('expected-tokens');
  const expectedCostEl = document.getElementById('expected-cost');
  const rangeTokensEl = document.getElementById('range-tokens');
  const rangeCostEl = document.getElementById('range-cost');
  const confidenceBadgeEl = document.getElementById('confidence-badge');
  const oodContainerEl = document.getElementById('ood-container');
  const factorsContainerEl = document.getElementById('factors-container');
  const factorsListEl = document.getElementById('factors-list');

  // Initialize UI from state
  function initialize() {
    if (promptInput) {
      promptInput.value = state.prompt || '';
      updateCharCount();
    }

    if (modelSelect && state.modelId) {
      modelSelect.value = state.modelId;
    }

    renderToolChips();

    // Populate samples dropdown
    if (samplesSelect) {
      samplesSelect.innerHTML = '<option value="">-- Insert Sample Prompt --</option>';
      SAMPLE_PROMPTS.forEach((sample, idx) => {
        const opt = document.createElement('option');
        opt.value = idx.toString();
        opt.textContent = sample.title;
        samplesSelect.appendChild(opt);
      });
    }

    // Restore cached result if available
    if (state.lastResult && state.prompt && state.prompt.trim()) {
      renderResult(state.lastResult);
    } else {
      showEmptyState();
    }

    // Ask extension host for any initial configuration
    vscode.postMessage({ type: 'requestInitialState' });
  }

  function renderToolChips() {
    if (!toolsContainer) return;
    toolsContainer.innerHTML = '';

    DEFAULT_TOOLS.forEach(tool => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `tool-chip ${state.selectedTools.includes(tool) ? 'selected' : ''}`;
      chip.dataset.tool = tool;
      chip.innerHTML = `<span class="chip-check">${state.selectedTools.includes(tool) ? '✓' : '+'}</span> ${tool}`;
      
      chip.addEventListener('click', () => {
        toggleTool(tool);
      });

      toolsContainer.appendChild(chip);
    });
  }

  function toggleTool(tool) {
    const idx = state.selectedTools.indexOf(tool);
    if (idx >= 0) {
      state.selectedTools.splice(idx, 1);
    } else {
      state.selectedTools.push(tool);
    }
    persistState();
    renderToolChips();
    triggerPrediction();
  }

  function updateCharCount() {
    if (!charCounter || !promptInput) return;
    const len = promptInput.value.length;
    charCounter.textContent = `${len.toLocaleString()} char${len === 1 ? '' : 's'}`;
    
    charCounter.classList.remove('warning', 'danger');
    if (len > 45000) {
      charCounter.classList.add('danger');
    } else if (len > 35000) {
      charCounter.classList.add('warning');
    }
  }

  function persistState() {
    vscode.setState(state);
    vscode.postMessage({
      type: 'saveState',
      state: state
    });
  }

  function showEmptyState() {
    if (emptyState) emptyState.style.display = 'block';
    if (resultCard) resultCard.style.display = 'none';
    if (errorContainer) errorContainer.style.display = 'none';
    if (statusDot) {
      statusDot.className = 'status-dot';
      statusDot.title = 'Ready';
    }
  }

  function showLoading() {
    if (statusDot) {
      statusDot.className = 'status-dot loading';
      statusDot.title = 'Predicting prompt cost...';
    }
    if (resultCard) {
      resultCard.classList.add('loading');
    }
  }

  function hideLoading() {
    if (resultCard) {
      resultCard.classList.remove('loading');
    }
  }

  function renderResult(data) {
    hideLoading();
    if (errorContainer) errorContainer.style.display = 'none';
    if (emptyState) emptyState.style.display = 'none';
    if (resultCard) resultCard.style.display = 'block';

    if (statusDot) {
      statusDot.className = 'status-dot';
      statusDot.title = 'Active';
    }

    // Expected tokens & cost
    if (expectedTokensEl) {
      expectedTokensEl.textContent = `${Number(data.expected_tokens || 0).toLocaleString()} tokens`;
    }

    if (expectedCostEl) {
      if (data.expected_cost_usd != null) {
        expectedCostEl.textContent = `~$${data.expected_cost_usd.toFixed(2)}`;
        expectedCostEl.style.display = 'inline-block';
      } else {
        expectedCostEl.style.display = 'none';
      }
    }

    // Range tokens
    if (rangeTokensEl) {
      const low = Number(data.low_tokens || 0).toLocaleString();
      const high = Number(data.high_tokens || 0).toLocaleString();
      rangeTokensEl.textContent = `${low} – ${high}`;
    }

    // Range cost
    if (rangeCostEl) {
      if (data.low_cost_usd != null && data.high_cost_usd != null) {
        rangeCostEl.textContent = `~$${data.low_cost_usd.toFixed(2)} – $${data.high_cost_usd.toFixed(2)}`;
        rangeCostEl.parentElement.style.display = 'flex';
      } else {
        rangeCostEl.parentElement.style.display = 'none';
      }
    }

    // Confidence badge
    if (confidenceBadgeEl) {
      const conf = (data.confidence || 'medium').toLowerCase();
      confidenceBadgeEl.className = `confidence-badge ${conf}`;
      confidenceBadgeEl.textContent = conf.toUpperCase();
    }

    // OOD Warning
    if (oodContainerEl) {
      if (data.out_of_distribution) {
        const reasons = (data.ood_reasons && data.ood_reasons.length > 0)
          ? data.ood_reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('')
          : '<li>Task parameters fall outside training distribution.</li>';
        
        oodContainerEl.innerHTML = `
          <div class="alert-box ood-alert">
            <div class="alert-title">⚠️ Out-of-Distribution Task</div>
            <ul class="alert-list">${reasons}</ul>
          </div>
        `;
        oodContainerEl.style.display = 'block';
      } else {
        oodContainerEl.innerHTML = '';
        oodContainerEl.style.display = 'none';
      }
    }

    // Driving factors
    if (factorsContainerEl && factorsListEl) {
      const factors = data.driving_factors || [];
      if (factors.length > 0) {
        factorsListEl.innerHTML = factors.map(f => `<li>${escapeHtml(f)}</li>`).join('');
        factorsContainerEl.style.display = 'block';
      } else {
        factorsContainerEl.style.display = 'none';
      }
    }
  }

  function renderError(message, options = {}) {
    hideLoading();
    if (resultCard) resultCard.style.display = 'none';
    if (emptyState) emptyState.style.display = 'none';

    if (statusDot) {
      statusDot.className = 'status-dot error';
      statusDot.title = 'Prediction error';
    }

    if (!errorContainer) return;

    if (options.isUncalibratedModel) {
      errorContainer.innerHTML = `
        <div class="alert-box uncalibrated-alert">
          <div class="alert-title">ℹ️ Model Not Calibrated</div>
          <p>Model <strong>'${escapeHtml(options.uncalibratedModelId || state.modelId)}'</strong> does not have an active calibration bundle in Citadel Predict yet.</p>
          <p style="margin-top: 6px; font-size: 11px; opacity: 0.9;">Switch to <strong>claude-sonnet</strong> for calibrated token & cost estimates.</p>
        </div>
      `;
    } else if (options.isNetworkError) {
      errorContainer.innerHTML = `
        <div class="alert-box error-alert">
          <div class="alert-title">⚡ Unable to reach Citadel Predict</div>
          <p>${escapeHtml(message)}</p>
          <div style="margin-top: 8px;">
            <button type="button" class="btn-secondary" id="retry-btn">Retry Connection</button>
          </div>
        </div>
      `;
      const retryBtn = document.getElementById('retry-btn');
      if (retryBtn) {
        retryBtn.addEventListener('click', () => triggerPrediction(true));
      }
    } else {
      errorContainer.innerHTML = `
        <div class="alert-box error-alert">
          <div class="alert-title">❌ Prediction Error</div>
          <p>${escapeHtml(message)}</p>
        </div>
      `;
    }

    errorContainer.style.display = 'block';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function triggerPrediction(immediate = false) {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }

    const text = promptInput ? promptInput.value.trim() : '';
    state.prompt = promptInput ? promptInput.value : '';
    state.modelId = modelSelect ? modelSelect.value : state.modelId;
    persistState();

    if (!text) {
      showEmptyState();
      return;
    }

    const doRequest = () => {
      showLoading();
      vscode.postMessage({
        type: 'predict',
        taskText: text,
        tools: state.selectedTools,
        modelId: state.modelId
      });
    };

    if (immediate) {
      doRequest();
    } else {
      debounceTimer = setTimeout(doRequest, debounceDelayMs);
    }
  }

  // Event Listeners
  if (promptInput) {
    promptInput.addEventListener('input', () => {
      updateCharCount();
      triggerPrediction(false);
    });
  }

  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      state.modelId = modelSelect.value;
      persistState();
      triggerPrediction(true);
    });
  }

  if (selectAllToolsBtn) {
    selectAllToolsBtn.addEventListener('click', () => {
      state.selectedTools = [...DEFAULT_TOOLS];
      persistState();
      renderToolChips();
      triggerPrediction(true);
    });
  }

  if (clearToolsBtn) {
    clearToolsBtn.addEventListener('click', () => {
      state.selectedTools = [];
      persistState();
      renderToolChips();
      triggerPrediction(true);
    });
  }

  if (samplesSelect) {
    samplesSelect.addEventListener('change', (e) => {
      const idx = parseInt(e.target.value, 10);
      if (!isNaN(idx) && SAMPLE_PROMPTS[idx]) {
        const sample = SAMPLE_PROMPTS[idx];
        if (promptInput) promptInput.value = sample.text;
        state.prompt = sample.text;
        state.selectedTools = [...sample.tools];
        updateCharCount();
        renderToolChips();
        persistState();
        triggerPrediction(true);
      }
      e.target.value = '';
    });
  }

  if (clearPromptBtn) {
    clearPromptBtn.addEventListener('click', () => {
      if (promptInput) promptInput.value = '';
      state.prompt = '';
      state.lastResult = null;
      updateCharCount();
      persistState();
      showEmptyState();
    });
  }

  if (copyPromptBtn) {
    copyPromptBtn.addEventListener('click', () => {
      const text = promptInput ? promptInput.value : '';
      if (!text) return;

      vscode.postMessage({
        type: 'copyPrompt',
        text: text
      });

      const originalHtml = copyPromptBtn.innerHTML;
      copyPromptBtn.innerHTML = '<span>✓ Copied to Clipboard!</span>';
      copyPromptBtn.classList.add('copied');

      setTimeout(() => {
        copyPromptBtn.innerHTML = originalHtml;
        copyPromptBtn.classList.remove('copied');
      }, 2000);
    });
  }

  // Listen for messages from host extension
  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg) return;

    switch (msg.type) {
      case 'predictionLoading':
        showLoading();
        break;

      case 'predictionSuccess':
        state.lastResult = msg.data;
        persistState();
        renderResult(msg.data);
        break;

      case 'predictionError':
        renderError(msg.error, {
          isUncalibratedModel: msg.isUncalibratedModel,
          uncalibratedModelId: msg.uncalibratedModelId,
          isNetworkError: msg.isNetworkError
        });
        break;

      case 'configUpdate':
        if (msg.config) {
          if (msg.config.debounceDelayMs) {
            debounceDelayMs = msg.config.debounceDelayMs;
          }
          if (msg.config.defaultModel && (!state.modelId || state.modelId === 'claude-sonnet')) {
            state.modelId = msg.config.defaultModel;
            if (modelSelect) modelSelect.value = state.modelId;
          }
        }
        break;

      case 'setPrompt':
        if (typeof msg.text === 'string') {
          if (promptInput) promptInput.value = msg.text;
          state.prompt = msg.text;
          updateCharCount();
          persistState();
          triggerPrediction(true);
        }
        break;
    }
  });

  // Run initial mount
  initialize();
})();
