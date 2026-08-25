# Migration Notes — Adding Multi-Model Calibration in Phase 2

This document provides step-by-step instructions for adding calibration datasets for new model families (e.g. Gemini via Google AI Studio, Groq open-weight models like Llama 3 / DeepSeek) in **Phase 2**.

---

## 1. Architectural Model Registry

The v1 architecture introduces a `model_id`-keyed lookup designed for incremental addition of new models without refactoring the API, schema, or UI.

```
data_collection/
└── calibration_data/
    ├── claude-sonnet.joblib      # Frozen v1 baseline (N=20/80 runs)
    ├── gemini-2.0-flash.joblib   # Example Phase 2 addition
    └── llama-3.3-70b.joblib      # Example Phase 2 addition
```

---

## 2. Model Bundle Interface Specification

Every calibration artifact must be a Python `joblib` bundle serialized as a dictionary with the following schema:

```python
{
    "model": <sklearn.base.BaseEstimator>, # Trained point regressor predicting log_total_tokens
    "q_low": float,                       # 10th percentile residual from LOTO CV in log space
    "q_high": float,                      # 90th percentile residual from LOTO CV in log space
    "feature_names": [                    # Must match FEATURE_NAMES in data_collection/features.py
        "text_char_len",
        "text_word_len",
        "num_tools",
        "open_ended_keyword_hits",
        "narrow_keyword_hits",
        "max_explicit_count",
        "sum_explicit_counts",
        "step_connector_hits",
        "num_clauses",
        "is_question",
    ],
    "model_metadata": {                   # Optional descriptive metadata
        "model_id": "gemini-2.0-flash",
        "provider": "google-ai-studio",
        "num_tasks": 20,
        "num_runs": 80,
        "calibration_date": "2026-09-01",
        "loto_coverage": 0.825
    }
}
```

---

## 3. Step-by-Step Addition Procedure

When a new model's API access (e.g., free tier Gemini / Groq) is ready:

### Step 1: Collect Runs & Train Model
1. Implement a model caller driver in `data_collection/` (similar to `agent_runner.py` for Claude Sonnet).
2. Collect $N$ runs per task across the benchmark suite (saving to `data/runs_{model_id}.jsonl`).
3. Train the point regressor and compute the out-of-fold leave-one-task-out (LOTO) residuals `q_low` and `q_high`.
4. Save the bundle to `data_collection/calibration_data/{model_id}.joblib`.

### Step 2: Register `model_id` in Predictor
Open [data_collection/predict.py](file:///c:/Users/Athul%20VR/OneDrive/Desktop/Citadel/data_collection/predict.py) and add the new identifier to `SUPPORTED_MODELS`:

```python
SUPPORTED_MODELS = {
    "claude-sonnet",
    "gemini-2.0-flash",    # <-- Add new model_id here
}
```

### Step 3: Verify with Pytest
Run the test suite to ensure the new model loads, predicts, and passes validation:

```bash
pytest backend/tests -v
```

That is all! The FastAPI backend automatically accepts `model_id="gemini-2.0-flash"` in requests, caches the loaded bundle in memory, and serves predictions without any further code changes.

---

## 4. Key Constraints & Guarantees

1. **Untouchable Sonnet Dataset**: The existing Claude Sonnet dataset (`claude-sonnet.joblib` and `data/runs.jsonl`) is finalized and will not be altered or re-run.
2. **Feature Parity**: Any new model must use the exact 10 numerical features derived by `data_collection/features.py`.
3. **Additive Deployment**: Adding a new model is strictly a single-file addition (`.joblib` drop-in) + single line addition in `SUPPORTED_MODELS`.
