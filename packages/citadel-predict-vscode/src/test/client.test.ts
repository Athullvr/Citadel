/**
 * Tests for CitadelClient and API Error Handling
 */

import * as assert from 'assert';
import { CitadelApiError, CitadelClient } from '../citadelClient';
import { ExtensionConfig, PredictResponse } from '../types';

export async function runClientTests(): Promise<void> {
  console.log('Running CitadelClient Unit Tests...\n');

  const baseConfig: ExtensionConfig = {
    apiUrl: 'https://citadel-7j9u.onrender.com',
    defaultModel: 'claude-sonnet',
    apiKey: '',
    debounceDelayMs: 400,
    timeoutSeconds: 10,
  };

  // 1. Test Base URL normalization
  {
    const client = new CitadelClient({ ...baseConfig, apiUrl: 'https://example.com/api/' });
    assert.strictEqual(client.getBaseUrl(), 'https://example.com/api');
    console.log('✓ Base URL normalization test passed');
  }

  // 2. Test Success response handling with mocked fetch
  {
    const mockData: PredictResponse = {
      model_id: 'claude-sonnet',
      low_tokens: 15000,
      expected_tokens: 32000,
      high_tokens: 58000,
      expected_cost_usd: 0.24,
      low_cost_usd: 0.11,
      high_cost_usd: 0.44,
      cost_note: '75% input / 25% output blend',
      driving_factors: ['Task length', 'Tool: web_search'],
      features: { prompt_chars: 120 },
      confidence: 'high',
      out_of_distribution: false,
      ood_reasons: [],
    };

    const originalFetch = global.fetch;
    global.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      assert.strictEqual(url, 'https://citadel-7j9u.onrender.com/api/predict');
      assert.strictEqual(init?.method, 'POST');
      const body = JSON.parse(init?.body as string);
      assert.strictEqual(body.task_text, 'Test prompt');
      assert.deepStrictEqual(body.tools, ['web_search']);
      assert.strictEqual(body.model_id, 'claude-sonnet');

      return {
        ok: true,
        status: 200,
        json: async () => mockData,
      } as Response;
    }) as unknown as typeof fetch;

    const client = new CitadelClient(baseConfig);
    const result = await client.predict({
      task_text: 'Test prompt',
      tools: ['web_search'],
      model_id: 'claude-sonnet',
    });

    assert.strictEqual(result.expected_tokens, 32000);
    assert.strictEqual(result.expected_cost_usd, 0.24);
    assert.strictEqual(result.confidence, 'high');
    assert.strictEqual(result.out_of_distribution, false);
    console.log('✓ Mock prediction success test passed');

    global.fetch = originalFetch;
  }

  // 3. Test Uncalibrated Model Error Handling (HTTP 400 unsupported model)
  {
    const originalFetch = global.fetch;
    global.fetch = (async () => {
      return {
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({
          error: 'bad_request',
          message: "Unsupported model_id 'gemini-pro'. Supported models in v1: ['claude-sonnet']",
          detail: "Unsupported model_id 'gemini-pro'. Supported models in v1: ['claude-sonnet']",
        }),
      } as Response;
    }) as unknown as typeof fetch;

    const client = new CitadelClient(baseConfig);
    try {
      await client.predict({
        task_text: 'Test prompt',
        tools: ['web_search'],
        model_id: 'gemini-pro',
      });
      assert.fail('Expected uncalibrated model error was not thrown');
    } catch (err) {
      assert.ok(err instanceof CitadelApiError, 'Error should be CitadelApiError');
      if (err instanceof CitadelApiError) {
        assert.strictEqual(err.isUncalibratedModel, true);
        assert.strictEqual(err.uncalibratedModelId, 'gemini-pro');
        assert.strictEqual(err.statusCode, 400);
      }
      console.log('✓ Uncalibrated model error test passed');
    }

    global.fetch = originalFetch;
  }

  // 4. Test Auth Error Handling (HTTP 401)
  {
    const originalFetch = global.fetch;
    global.fetch = (async () => {
      return {
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        json: async () => ({
          error: 'unauthorized',
          message: 'Invalid API key',
        }),
      } as Response;
    }) as unknown as typeof fetch;

    const client = new CitadelClient(baseConfig);
    try {
      await client.predict({
        task_text: 'Test prompt',
        tools: [],
        model_id: 'claude-sonnet',
      });
      assert.fail('Expected auth error was not thrown');
    } catch (err) {
      assert.ok(err instanceof CitadelApiError);
      if (err instanceof CitadelApiError) {
        assert.strictEqual(err.isAuthError, true);
        assert.strictEqual(err.statusCode, 401);
      }
      console.log('✓ Authentication error test passed');
    }

    global.fetch = originalFetch;
  }

  // 5. Test Rate Limit Error Handling (HTTP 429)
  {
    const originalFetch = global.fetch;
    global.fetch = (async () => {
      const headers = new Map<string, string>();
      headers.set('Retry-After', '45');
      return {
        ok: false,
        status: 429,
        statusText: 'Too Many Requests',
        headers: {
          get: (name: string) => headers.get(name) || null,
        },
        json: async () => ({
          error: 'rate_limited',
          message: 'Rate limit exceeded: 30/minute',
          retry_after: 45,
        }),
      } as unknown as Response;
    }) as unknown as typeof fetch;

    const client = new CitadelClient(baseConfig);
    try {
      await client.predict({
        task_text: 'Test prompt',
        tools: [],
        model_id: 'claude-sonnet',
      });
      assert.fail('Expected rate limit error was not thrown');
    } catch (err) {
      assert.ok(err instanceof CitadelApiError);
      if (err instanceof CitadelApiError) {
        assert.strictEqual(err.isRateLimited, true);
        assert.strictEqual(err.retryAfter, 45);
      }
      console.log('✓ Rate limit error test passed');
    }

    global.fetch = originalFetch;
  }

  // 6. Test Network / Timeout Error Handling
  {
    const originalFetch = global.fetch;
    global.fetch = (async () => {
      throw new TypeError('fetch failed (connect ECONNREFUSED)');
    }) as unknown as typeof fetch;

    const client = new CitadelClient(baseConfig);
    try {
      await client.predict({
        task_text: 'Test prompt',
        tools: [],
        model_id: 'claude-sonnet',
      });
      assert.fail('Expected network error was not thrown');
    } catch (err) {
      assert.ok(err instanceof CitadelApiError);
      if (err instanceof CitadelApiError) {
        assert.strictEqual(err.isNetworkError, true);
        assert.ok(err.message.includes('Unable to reach Citadel Predict API'));
      }
      console.log('✓ Network error handling test passed');
    }

    global.fetch = originalFetch;
  }

  // 7. Test Auth Header Propagation
  {
    const originalFetch = global.fetch;
    let sentAuthHeader: string | undefined;

    global.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      const h = init?.headers as Record<string, string>;
      sentAuthHeader = h?.Authorization;
      return {
        ok: true,
        status: 200,
        json: async () => ({ expected_tokens: 100 }),
      } as Response;
    }) as unknown as typeof fetch;

    const authedClient = new CitadelClient({ ...baseConfig, apiKey: 'citadel-secret-123' });
    await authedClient.predict({ task_text: 'Secret test', tools: [], model_id: 'claude-sonnet' });

    assert.strictEqual(sentAuthHeader, 'Bearer citadel-secret-123');
    console.log('✓ Auth header propagation test passed');

    global.fetch = originalFetch;
  }

  console.log('\nAll CitadelClient unit tests passed successfully!');
}
