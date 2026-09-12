/**
 * Standalone test runner for Citadel Predict VS Code extension.
 */

import { runClientTests } from './client.test';

async function main(): Promise<void> {
  console.log('=== Citadel Predict VS Code Extension Test Suite ===\n');

  try {
    await runClientTests();
    console.log('\n=================================================');
    console.log('🎉 ALL TESTS COMPLETED SUCCESSFULLY');
    console.log('=================================================\n');
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Test Suite Failed:', error);
    process.exit(1);
  }
}

main();
