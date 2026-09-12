/**
 * Extension entrypoint for Citadel Predict — Live Prompt Cost
 */

import * as vscode from 'vscode';
import { CitadelViewProvider } from './citadelViewProvider';
import { ExtensionConfig } from './types';

export function activate(context: vscode.ExtensionContext): void {
  const readConfig = (): ExtensionConfig => {
    const wsConfig = vscode.workspace.getConfiguration('citadelPredict');
    return {
      apiUrl: wsConfig.get<string>('apiUrl', 'https://citadel-7j9u.onrender.com'),
      defaultModel: wsConfig.get<string>('defaultModel', 'claude-sonnet'),
      apiKey: wsConfig.get<string>('apiKey', ''),
      debounceDelayMs: wsConfig.get<number>('debounceDelayMs', 400),
      timeoutSeconds: wsConfig.get<number>('timeoutSeconds', 10),
    };
  };

  const initialConfig = readConfig();
  const provider = new CitadelViewProvider(context.extensionUri, context, initialConfig);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(CitadelViewProvider.viewType, provider, {
      webviewOptions: {
        retainContextWhenHidden: true,
      },
    })
  );

  // Command: Focus Cost Estimation Panel
  context.subscriptions.push(
    vscode.commands.registerCommand('citadelPredict.focusView', async () => {
      await vscode.commands.executeCommand('citadelPredict.costView.focus');
    })
  );

  // Command: Clear Prompt
  context.subscriptions.push(
    vscode.commands.registerCommand('citadelPredict.clearPrompt', () => {
      provider.clearPrompt();
    })
  );

  // Command: Estimate Selected Text from Active Text Editor
  context.subscriptions.push(
    vscode.commands.registerCommand('citadelPredict.estimateSelectedText', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showInformationMessage('No active editor found to extract prompt from.');
        return;
      }

      const selection = editor.selection;
      const selectedText = editor.document.getText(selection).trim();

      if (!selectedText) {
        vscode.window.showInformationMessage('Please highlight some prompt text to estimate.');
        return;
      }

      // Focus panel and send text
      await vscode.commands.executeCommand('citadelPredict.costView.focus');
      provider.setPrompt(selectedText);
    })
  );

  // Listen for configuration updates
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('citadelPredict')) {
        const updatedConfig = readConfig();
        provider.updateConfig(updatedConfig);
      }
    })
  );
}

export function deactivate(): void {
  // Cleanup resources if needed
}
