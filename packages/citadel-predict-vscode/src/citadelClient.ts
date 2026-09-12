/**
 * HTTP Client for Citadel Predict API.
 */

import { ApiErrorResponse, ExtensionConfig, PredictRequest, PredictResponse } from './types';

export class CitadelApiError extends Error {
  public readonly statusCode?: number;
  public readonly errorType?: string;
  public readonly isUncalibratedModel: boolean;
  public readonly uncalibratedModelId?: string;
  public readonly isNetworkError: boolean;
  public readonly isAuthError: boolean;
  public readonly isRateLimited: boolean;
  public readonly retryAfter?: number;
  public readonly detail?: unknown;

  constructor(options: {
    message: string;
    statusCode?: number;
    errorType?: string;
    isUncalibratedModel?: boolean;
    uncalibratedModelId?: string;
    isNetworkError?: boolean;
    isAuthError?: boolean;
    isRateLimited?: boolean;
    retryAfter?: number;
    detail?: unknown;
  }) {
    super(options.message);
    this.name = 'CitadelApiError';
    this.statusCode = options.statusCode;
    this.errorType = options.errorType;
    this.isUncalibratedModel = !!options.isUncalibratedModel;
    this.uncalibratedModelId = options.uncalibratedModelId;
    this.isNetworkError = !!options.isNetworkError;
    this.isAuthError = !!options.isAuthError;
    this.isRateLimited = !!options.isRateLimited;
    this.retryAfter = options.retryAfter;
    this.detail = options.detail;
  }
}

export class CitadelClient {
  private config: ExtensionConfig;

  constructor(config: ExtensionConfig) {
    this.config = config;
  }

  public updateConfig(config: ExtensionConfig): void {
    this.config = config;
  }

  public getBaseUrl(): string {
    let url = (this.config.apiUrl || 'https://citadel-7j9u.onrender.com').trim();
    if (url.endsWith('/')) {
      url = url.slice(0, -1);
    }
    return url;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'User-Agent': 'Citadel-Predict-VSCode/0.1.0',
    };

    const apiKey = (this.config.apiKey || '').trim();
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }

    return headers;
  }

  public async predict(req: PredictRequest): Promise<PredictResponse> {
    const baseUrl = this.getBaseUrl();
    const endpoint = `${baseUrl}/api/predict`;

    const controller = new AbortController();
    const timeoutMs = (this.config.timeoutSeconds || 10) * 1000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify({
          task_text: req.task_text,
          tools: req.tools,
          model_id: req.model_id,
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (response.ok) {
        const data = (await response.json()) as PredictResponse;
        return data;
      }

      // Handle non-200 responses
      let errorBody: ApiErrorResponse = {};
      try {
        errorBody = (await response.json()) as ApiErrorResponse;
      } catch {
        // Response is not JSON
      }

      const rawDetail = typeof errorBody.detail === 'string'
        ? errorBody.detail
        : errorBody.message || response.statusText || 'Unknown error';

      // Check if it's an uncalibrated / unsupported model error
      if (
        response.status === 400 &&
        (rawDetail.toLowerCase().includes('unsupported model_id') ||
          rawDetail.toLowerCase().includes('not yet calibrated') ||
          rawDetail.toLowerCase().includes('model_id'))
      ) {
        throw new CitadelApiError({
          message: `Model '${req.model_id}' is not yet calibrated in Citadel Predict.`,
          statusCode: 400,
          errorType: 'uncalibrated_model',
          isUncalibratedModel: true,
          uncalibratedModelId: req.model_id,
          detail: errorBody,
        });
      }

      if (response.status === 401 || response.status === 403) {
        throw new CitadelApiError({
          message:
            'Citadel Predict API authentication failed. Please check your citadelPredict.apiKey setting.',
          statusCode: response.status,
          errorType: 'unauthorized',
          isAuthError: true,
          detail: errorBody,
        });
      }

      if (response.status === 429) {
        const retryAfterHeader = response.headers.get('Retry-After');
        const retryAfter = retryAfterHeader
          ? parseInt(retryAfterHeader, 10)
          : errorBody.retry_after;
        throw new CitadelApiError({
          message: `Rate limit reached. Please wait ${retryAfter || 30}s before retrying.`,
          statusCode: 429,
          errorType: 'rate_limited',
          isRateLimited: true,
          retryAfter,
          detail: errorBody,
        });
      }

      if (response.status === 422) {
        throw new CitadelApiError({
          message: `Validation error: ${rawDetail}`,
          statusCode: 422,
          errorType: 'validation_error',
          detail: errorBody,
        });
      }

      throw new CitadelApiError({
        message: `Citadel API error (${response.status}): ${rawDetail}`,
        statusCode: response.status,
        errorType: errorBody.error || 'server_error',
        detail: errorBody,
      });
    } catch (err: unknown) {
      clearTimeout(timer);

      if (err instanceof CitadelApiError) {
        throw err;
      }

      const isAbort = (err as { name?: string })?.name === 'AbortError';
      const errorMsg = isAbort
        ? `Request timed out after ${this.config.timeoutSeconds || 10}s reaching ${baseUrl}`
        : `Unable to reach Citadel Predict API at ${baseUrl}. Check your connection or apiUrl setting.`;

      throw new CitadelApiError({
        message: errorMsg,
        isNetworkError: true,
        detail: err,
      });
    }
  }

  public async health(): Promise<{ status: string; version: string; supported_models?: string[] }> {
    const baseUrl = this.getBaseUrl();
    const endpoint = `${baseUrl}/api/health`;

    const controller = new AbortController();
    const timeoutMs = (this.config.timeoutSeconds || 5) * 1000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: this.getHeaders(),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!response.ok) {
        throw new Error(`Health check failed with status ${response.status}`);
      }

      return (await response.json()) as { status: string; version: string; supported_models?: string[] };
    } catch (err: unknown) {
      clearTimeout(timer);
      throw new CitadelApiError({
        message: `Health check failed for ${baseUrl}`,
        isNetworkError: true,
        detail: err,
      });
    }
  }
}
