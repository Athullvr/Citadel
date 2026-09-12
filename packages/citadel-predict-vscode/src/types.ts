/**
 * Types and interfaces for Citadel Predict VS Code Extension.
 */

export interface PredictRequest {
  task_text: string;
  tools: string[];
  model_id: string;
}

export interface PredictResponse {
  model_id: string;
  low_tokens: number;
  expected_tokens: number;
  high_tokens: number;
  expected_cost_usd: number | null;
  low_cost_usd: number | null;
  high_cost_usd: number | null;
  cost_note: string | null;
  driving_factors: string[];
  features: Record<string, unknown>;
  confidence: 'high' | 'medium' | 'low' | string;
  out_of_distribution: boolean;
  ood_reasons: string[];
}

export interface ApiErrorResponse {
  error?: string;
  message?: string;
  detail?: string | Array<{ loc?: Array<string | number>; msg?: string }>;
  retry_after?: number;
}

export interface ToolDefinition {
  name: string;
  label: string;
  description: string;
  category?: 'Standard' | 'Workspace' | 'Execution' | 'Communication';
  icon?: string;
}

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  isCalibrated: boolean;
}

export interface ExtensionConfig {
  apiUrl: string;
  defaultModel: string;
  apiKey: string;
  debounceDelayMs: number;
  timeoutSeconds: number;
}

export interface WebviewPersistedState {
  prompt: string;
  selectedTools: string[];
  modelId: string;
  lastResult: PredictResponse | null;
}

// Messages sent from Webview to Host Extension
export type WebviewToHostMessage =
  | {
      type: 'predict';
      taskText: string;
      tools: string[];
      modelId: string;
    }
  | {
      type: 'copyPrompt';
      text: string;
    }
  | {
      type: 'saveState';
      state: WebviewPersistedState;
    }
  | {
      type: 'openSettings';
    }
  | {
      type: 'requestInitialState';
    };

// Messages sent from Host Extension to Webview
export type HostToWebviewMessage =
  | {
      type: 'predictionLoading';
    }
  | {
      type: 'predictionSuccess';
      data: PredictResponse;
      taskText: string;
    }
  | {
      type: 'predictionError';
      error: string;
      isUncalibratedModel?: boolean;
      uncalibratedModelId?: string;
      isNetworkError?: boolean;
      isAuthError?: boolean;
      isRateLimited?: boolean;
      retryAfter?: number;
    }
  | {
      type: 'configUpdate';
      config: {
        apiUrl: string;
        defaultModel: string;
        debounceDelayMs: number;
      };
    }
  | {
      type: 'setPrompt';
      text: string;
    };
