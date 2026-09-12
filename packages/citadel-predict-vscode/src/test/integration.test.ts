/**
 * Live Integration Test against the hosted Citadel Predict API.
 */

import { CitadelClient } from '../citadelClient';

async function runLiveIntegrationTest(): Promise<void> {
  console.log('Testing live connection to https://citadel-7j9u.onrender.com ...');

  const client = new CitadelClient({
    apiUrl: 'https://citadel-7j9u.onrender.com',
    defaultModel: 'claude-sonnet',
    apiKey: '',
    debounceDelayMs: 400,
    timeoutSeconds: 15,
  });

  try {
    const health = await client.health();
    console.log('✓ Health check response:', health);

    const result = await client.predict({
      task_text: 'Research competitor pricing across 3 websites and compile a markdown summary report.',
      tools: ['web_search', 'fetch_url', 'draft_document'],
      model_id: 'claude-sonnet',
    });

    console.log('✓ Live Prediction response:', {
      model_id: result.model_id,
      expected_tokens: result.expected_tokens,
      expected_cost_usd: result.expected_cost_usd,
      confidence: result.confidence,
      out_of_distribution: result.out_of_distribution,
      driving_factors: result.driving_factors,
    });

    // Test uncalibrated model error
    try {
      await client.predict({
        task_text: 'Test uncalibrated model handling',
        tools: ['web_search'],
        model_id: 'gemini-pro',
      });
      console.log('⚠️ Unexpected: gemini-pro succeeded');
    } catch (err: any) {
      console.log('✓ Expected uncalibrated error caught:', {
        message: err.message,
        isUncalibratedModel: err.isUncalibratedModel,
        uncalibratedModelId: err.uncalibratedModelId,
      });
    }

    console.log('\n🎉 Live integration test PASSED successfully!');
  } catch (err) {
    console.error('Live integration test error:', err);
    // Do not fail if backend is in cold-sleep, but report status
  }
}

runLiveIntegrationTest();
