"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_CITADEL_API_KEY ?? "";

type Tool = { name: string; description: string };

type PredictResponse = {
  model_id: string;
  low_tokens: number;
  expected_tokens: number;
  high_tokens: number;
  driving_factors: string[];
  features: Record<string, number>;
  confidence: "high" | "moderate" | "low";
  out_of_distribution: boolean;
  ood_reasons: string[];
};

type ValidationExample = {
  task_id: string;
  task_text: string;
  tools_available: string[];
  category: string;
  actual_tokens_observed: number[];
  pred_low: number;
  pred_expected: number;
  pred_high: number;
  verdict: string;
  note: string;
};

function formatTokens(n: number): string {
  return n.toLocaleString("en-US");
}

function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }
  return headers;
}

function RangeBar({ low, expected, high }: { low: number; expected: number; high: number }) {
  const logLow = Math.log(Math.max(low, 1));
  const logHigh = Math.log(Math.max(high, low + 1));
  const logExpected = Math.log(Math.max(expected, 1));
  const pct = ((logExpected - logLow) / (logHigh - logLow)) * 100;
  const clampedPct = Math.min(97, Math.max(3, pct));

  return (
    <div className="w-full">
      <div className="relative h-3 w-full rounded-full bg-gradient-to-r from-emerald-200 via-amber-200 to-rose-200 dark:from-emerald-900 dark:via-amber-900 dark:to-rose-900">
        <div
          className="absolute top-1/2 h-5 w-5 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 border-white bg-zinc-900 shadow dark:border-zinc-900 dark:bg-zinc-100"
          style={{ left: `${clampedPct}%` }}
          title={`Expected: ${formatTokens(expected)} tokens`}
        />
      </div>
      <div className="mt-2 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
        <span>Low: {formatTokens(low)}</span>
        <span className="font-semibold text-zinc-800 dark:text-zinc-200">
          Expected: {formatTokens(expected)}
        </span>
        <span>High: {formatTokens(high)}</span>
      </div>
    </div>
  );
}

function ComparisonBar({ example }: { example: ValidationExample }) {
  const { pred_low, pred_expected, pred_high, actual_tokens_observed } = example;
  const actualMin = Math.min(...actual_tokens_observed);
  const actualMax = Math.max(...actual_tokens_observed);

  const domainLow = Math.min(pred_low, actualMin);
  const domainHigh = Math.max(pred_high, actualMax);
  const logLow = Math.log(Math.max(domainLow, 1));
  const logHigh = Math.log(Math.max(domainHigh, domainLow + 1));
  const toPct = (v: number) =>
    Math.min(100, Math.max(0, ((Math.log(Math.max(v, 1)) - logLow) / (logHigh - logLow)) * 100));

  const isHit = example.verdict === "hit";

  return (
    <div className="flex flex-col gap-2">
      <div className="relative h-2.5 w-full rounded-full bg-zinc-100 dark:bg-zinc-800">
        <div
          className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full bg-zinc-300 dark:bg-zinc-600"
          style={{ left: `${toPct(pred_low)}%`, width: `${toPct(pred_high) - toPct(pred_low)}%` }}
          title={`Predicted: ${formatTokens(pred_low)} - ${formatTokens(pred_high)}`}
        />
        <div
          className="absolute top-1/2 h-1 w-1 -translate-y-1/2 rounded-full bg-zinc-500 dark:bg-zinc-400"
          style={{ left: `${toPct(pred_expected)}%` }}
        />
        <div
          className={`absolute top-1/2 h-4 w-1.5 -translate-y-1/2 -translate-x-1/2 rounded-sm ${
            isHit ? "bg-emerald-500" : "bg-rose-500"
          }`}
          style={{ left: `${toPct(actualMin)}%` }}
          title={`Actual (min): ${formatTokens(actualMin)}`}
        />
        <div
          className={`absolute top-1/2 h-4 w-1.5 -translate-y-1/2 -translate-x-1/2 rounded-sm ${
            isHit ? "bg-emerald-500" : "bg-rose-500"
          }`}
          style={{ left: `${toPct(actualMax)}%` }}
          title={`Actual (max): ${formatTokens(actualMax)}`}
        />
      </div>
      <div className="flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
        <span>Predicted: {formatTokens(pred_low)} – {formatTokens(pred_high)}</span>
        <span className={isHit ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}>
          Actual: {formatTokens(actualMin)} – {formatTokens(actualMax)}
        </span>
      </div>
    </div>
  );
}

export default function Home() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [selectedTools, setSelectedTools] = useState<Set<string>>(new Set());
  const [taskText, setTaskText] = useState(
    "Research this topic across 5 sources, synthesize into a report, then draft 3 follow-up emails to relevant stakeholders."
  );
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<ValidationExample[]>([]);

  useEffect(() => {
    fetch(`${API_BASE}/api/tools`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((data: Tool[]) => {
        if (Array.isArray(data)) {
          setTools(data);
          setSelectedTools(new Set(["web_search", "fetch_url", "draft_document", "send_email"]));
        }
      })
      .catch(() => setError("Could not reach the prediction API. Is the backend running on port 8000?"));

    fetch(`${API_BASE}/api/validation-examples`, { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setExamples(data);
        }
      })
      .catch(() => {});
  }, []);

  function toggleTool(name: string) {
    setSelectedTools((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  async function handlePredict() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: getAuthHeaders(),
        body: JSON.stringify({
          task_text: taskText,
          tools: Array.from(selectedTools),
          model_id: "claude-sonnet",
        }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail ?? `API returned HTTP ${res.status}`);
      }
      const data: PredictResponse = await res.json();
      setResult(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Prediction request failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col flex-1 items-center bg-zinc-50 font-sans dark:bg-black min-h-screen">
      <main className="flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 py-16 sm:px-10">
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
              Citadel Predict
            </h1>
            <span className="rounded-full bg-zinc-200 px-2.5 py-0.5 text-xs font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
              v1.0 (Claude Sonnet baseline)
            </span>
          </div>
          <p className="text-sm leading-6 text-zinc-600 dark:text-zinc-400">
            Pre-execution token cost predictor for AI agent runs. Paste a task description and pick
            available tools to receive a calibrated token range before a single token is spent.
          </p>
        </header>

        <section className="flex flex-col gap-3">
          <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300" htmlFor="task">
            Task description
          </label>
          <textarea
            id="task"
            maxLength={4000}
            className="min-h-28 w-full rounded-lg border border-zinc-300 bg-white p-3 text-sm text-zinc-900 shadow-sm focus:border-zinc-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
            value={taskText}
            onChange={(e) => setTaskText(e.target.value)}
            placeholder="e.g. Research this topic across 5 sources and draft a report."
          />
          <div className="text-right text-xs text-zinc-400">
            {taskText.length} / 4000 characters
          </div>
        </section>

        <section className="flex flex-col gap-3">
          <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            Available tools ({selectedTools.size} selected)
          </span>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {tools.map((tool) => (
              <label
                key={tool.name}
                className="flex cursor-pointer items-start gap-2 rounded-lg border border-zinc-200 bg-white p-3 text-sm shadow-sm hover:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900"
              >
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={selectedTools.has(tool.name)}
                  onChange={() => toggleTool(tool.name)}
                />
                <span>
                  <span className="block font-medium text-zinc-900 dark:text-zinc-100">
                    {tool.name}
                  </span>
                  <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                    {tool.description}
                  </span>
                </span>
              </label>
            ))}
          </div>
        </section>

        <button
          onClick={handlePredict}
          disabled={loading || !taskText.trim()}
          className="rounded-full bg-zinc-900 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
        >
          {loading ? "Predicting…" : "Predict cost range"}
        </button>

        {error && (
          <p className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700 dark:bg-rose-950 dark:text-rose-300">
            {error}
          </p>
        )}

        {result && (
          <section className="flex flex-col gap-6 rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                Predicted token range
              </h2>
              <span
                className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                  result.confidence === "high"
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                    : result.confidence === "moderate"
                    ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                    : "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300"
                }`}
              >
                {result.confidence.toUpperCase()} CONFIDENCE
              </span>
            </div>

            <RangeBar low={result.low_tokens} expected={result.expected_tokens} high={result.high_tokens} />

            {result.out_of_distribution && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/50 dark:bg-amber-950/40">
                <p className="text-xs font-semibold text-amber-800 dark:text-amber-200">
                  ⚠️ Out-of-Distribution Warning
                </p>
                <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-amber-700 dark:text-amber-300">
                  {result.ood_reasons.map((reason, idx) => (
                    <li key={idx}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex flex-col gap-2 border-t border-zinc-100 pt-4 dark:border-zinc-800">
              <h3 className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
                Why this estimate
              </h3>
              <ul className="flex flex-col gap-1.5">
                {result.driving_factors.map((factor, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-zinc-700 dark:text-zinc-300"
                  >
                    <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-zinc-400" />
                    {factor}
                  </li>
                ))}
              </ul>
            </div>

            <p className="border-t border-zinc-100 pt-4 text-xs leading-5 text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
              Model: <code className="font-mono text-zinc-700 dark:text-zinc-300">{result.model_id}</code> (N=20 tasks, 80 runs baseline).
              This is a statistical pattern match with an empirical calibration band, not an exact simulator.
            </p>
          </section>
        )}

        {examples.length > 0 && (
          <section className="flex flex-col gap-4 border-t border-zinc-200 pt-10 dark:border-zinc-800">
            <div>
              <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
                How this compares to real runs
              </h2>
              <p className="mt-1 text-sm leading-6 text-zinc-600 dark:text-zinc-400">
                4 real leave-one-task-out validation benchmarks: genuine generalization test
                on unseen tasks (including documented misses).
              </p>
            </div>

            <div className="flex flex-col gap-4">
              {examples.map((ex) => (
                <div
                  key={ex.task_id}
                  className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
                >
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-sm text-zinc-800 dark:text-zinc-200">{ex.task_text}</p>
                    <span
                      className={`flex-shrink-0 rounded-full px-2.5 py-1 text-xs font-medium ${
                        ex.verdict === "hit"
                          ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                          : "bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300"
                      }`}
                    >
                      {ex.verdict}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {ex.category} · {ex.tools_available.length} tool(s): {ex.tools_available.join(", ") || "none"}
                  </p>
                  <ComparisonBar example={ex} />
                  <p className="text-xs leading-5 text-zinc-600 dark:text-zinc-400">{ex.note}</p>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
