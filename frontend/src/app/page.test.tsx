import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Home from "./page";

describe("Frontend Smoke Test — Citadel Predict", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders page header and tool list from API", async () => {
    const mockTools = [
      { name: "web_search", description: "Search the web" },
      { name: "calculator", description: "Perform arithmetic calculations" },
    ];
    const mockExamples = [
      {
        task_id: "t15",
        task_text: "Calculate cost breakdown",
        tools_available: ["calculator"],
        category: "narrow_multi_step",
        actual_tokens_observed: [9950, 13734],
        pred_low: 4192,
        pred_expected: 12949,
        pred_high: 26476,
        verdict: "hit",
        note: "Observed within predicted range",
      },
    ];

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/api/tools")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockTools),
        });
      }
      if (url.endsWith("/api/validation-examples")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockExamples),
        });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<Home />);

    expect(screen.getByText("Citadel Predict")).toBeDefined();
    expect(screen.getByLabelText("Task description")).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText("web_search")).toBeDefined();
      expect(screen.getByText("calculator")).toBeDefined();
    });
  });

  it("submits prediction and displays result with confidence badge", async () => {
    const mockPrediction = {
      model_id: "claude-sonnet",
      low_tokens: 4200,
      expected_tokens: 12000,
      high_tokens: 25000,
      driving_factors: ["3 tools available -> higher branching/retry risk"],
      features: { num_tools: 3 },
      confidence: "high",
      out_of_distribution: false,
      ood_reasons: [],
    };

    global.fetch = vi.fn().mockImplementation((url: string, opts?: RequestInit) => {
      if (url.endsWith("/api/tools")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]),
        });
      }
      if (url.endsWith("/api/validation-examples")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]),
        });
      }
      if (url.endsWith("/api/predict") && opts?.method === "POST") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockPrediction),
        });
      }
      return Promise.reject(new Error("Unknown URL"));
    });

    render(<Home />);

    const predictButton = screen.getByRole("button", { name: /Predict cost range/i });
    fireEvent.click(predictButton);

    await waitFor(() => {
      expect(screen.getByText(/HIGH CONFIDENCE/i)).toBeDefined();
      expect(screen.getByText(/Expected: 12,000/i)).toBeDefined();
      expect(screen.getByText(/3 tools available/i)).toBeDefined();
    });
  });
});
