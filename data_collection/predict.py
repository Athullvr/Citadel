"""
Loads the trained model bundle and predicts a [low, expected, high] token
range for a new task description + tool list.

Supports model_id-based calibration bundle lookup (default: "claude-sonnet")
so future Phase 2 calibration models (e.g. Gemini, Groq/Llama) can be added
without changing the prediction interface.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from features import FEATURE_NAMES, extract_features

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration_data"

SUPPORTED_MODELS = {"claude-sonnet"}
DEFAULT_MODEL = "claude-sonnet"

# In-memory bundle cache keyed by model_id to avoid reloading joblib per request
_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}


def get_bundle_path(model_id: str = DEFAULT_MODEL) -> Path:
    """Resolve the joblib path for a given model_id."""
    # First check calibration_data/{model_id}.joblib
    path = CALIBRATION_DIR / f"{model_id}.joblib"
    if path.exists():
        return path
    # Backward-compat fallback for root model.joblib if model_id is default
    fallback = BASE_DIR / "model.joblib"
    if model_id == DEFAULT_MODEL and fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Calibration bundle for model '{model_id}' not found. "
        f"Expected path: {path}. Supported models: {list(SUPPORTED_MODELS)}"
    )


def load_bundle(model_id: str = DEFAULT_MODEL, path: str | None = None) -> dict[str, Any]:
    """Load and cache a calibration bundle for the given model_id."""
    if path is not None:
        bundle_path = Path(path)
        cache_key = str(bundle_path)
    else:
        bundle_path = get_bundle_path(model_id)
        cache_key = model_id

    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    bundle = joblib.load(bundle_path)
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def assess_confidence_and_ood(feats: dict[str, Any]) -> tuple[str, bool, list[str]]:
    """
    Assess whether the input task falls outside the N=20 training distribution
    and determine the prediction confidence level.
    
    Training distribution characteristics (N=20 tasks, 80 runs):
    - Tools available: 0 to 5 (only 2 tasks had 0 tools; maximum was 5)
    - Task text length: typically 20-300 characters
    - Keyword signal: contains either narrow or open-ended action phrasing
    """
    ood_reasons = []

    # Check tool counts against training range
    if feats["num_tools"] > 5:
        ood_reasons.append(
            f"Tool count ({feats['num_tools']}) exceeds training benchmark maximum (5 tools). "
            "Multi-tool compounding error may be underestimated."
        )
    elif feats["num_tools"] == 0:
        ood_reasons.append(
            "Zero tools provided. Benchmark dataset contains only 2 zero-tool tasks; "
            "variance at the low-cost extreme is elevated."
        )

    # Check task description length
    if feats["text_char_len"] > 600:
        ood_reasons.append(
            f"Task description is unusually long ({feats['text_char_len']} chars). "
            "Prompt token overhead may exceed training baseline."
        )

    # Check keyword patterns
    if feats["open_ended_keyword_hits"] == 0 and feats["narrow_keyword_hits"] == 0:
        ood_reasons.append(
            "No recognized action keyword patterns detected (e.g. 'research', 'summarize', 'investigate'). "
            "Estimate relies strictly on tool count and text length."
        )

    is_ood = len(ood_reasons) > 0

    # Determine confidence level
    if is_ood:
        confidence = "low"
    elif feats["num_tools"] in (1, 2, 3) and feats["narrow_keyword_hits"] > 0:
        confidence = "high"
    else:
        confidence = "moderate"

    return confidence, is_ood, ood_reasons


def predict(
    task_text: str,
    tool_names: list[str],
    model_id: str = DEFAULT_MODEL,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Predict token cost range [low, expected, high] for a given task and tool list.
    """
    if bundle is None:
        bundle = load_bundle(model_id=model_id)

    feats = extract_features(task_text, tool_names)
    X = pd.DataFrame([feats])[FEATURE_NAMES]

    pred_log = bundle["model"].predict(X)[0]
    low = float(np.exp(pred_log + bundle["q_low"]))
    expected = float(np.exp(pred_log))
    high = float(np.exp(pred_log + bundle["q_high"]))

    driving_factors = []
    if feats["num_tools"] >= 3:
        driving_factors.append(f"{feats['num_tools']} tools available -> higher branching/retry risk")
    elif feats["num_tools"] == 0:
        driving_factors.append("no tools available -> likely single-turn, low cost")
    if feats["open_ended_keyword_hits"] > 0:
        driving_factors.append(
            f"{feats['open_ended_keyword_hits']} open-ended phrase(s) detected "
            "(e.g. 'research', 'until', 'investigate') -> task likely requires "
            "multiple turns/iteration"
        )
    if feats["narrow_keyword_hits"] > 0:
        driving_factors.append(
            f"{feats['narrow_keyword_hits']} narrow/bounded phrase(s) detected -> "
            "likely a single well-specified action"
        )
    if feats["sum_explicit_counts"] > 3:
        driving_factors.append(
            f"task mentions {feats['sum_explicit_counts']} explicit repeated items "
            "(sources/emails/etc.) -> more sub-actions expected"
        )
    if not driving_factors:
        driving_factors.append("no strong signals detected -> estimate based on task length and tool count alone")

    confidence, out_of_distribution, ood_reasons = assess_confidence_and_ood(feats)

    return {
        "model_id": model_id,
        "low_tokens": round(low),
        "expected_tokens": round(expected),
        "high_tokens": round(high),
        "features": feats,
        "driving_factors": driving_factors,
        "confidence": confidence,
        "out_of_distribution": out_of_distribution,
        "ood_reasons": ood_reasons,
    }


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else \
        "Research this topic across 5 sources and draft a report."
    if len(sys.argv) > 2:
        tools = [t for t in sys.argv[2].split(",") if t]
    else:
        tools = ["web_search", "fetch_url", "draft_document"]

    result = predict(text, tools)
    print(f"Model: {result['model_id']}")
    print(f"Task: {text}")
    print(f"Tools: {tools}")
    print(f"Predicted range: {result['low_tokens']:,} - {result['expected_tokens']:,} - {result['high_tokens']:,} tokens")
    print(f"Confidence: {result['confidence']} (OOD: {result['out_of_distribution']})")
    if result["ood_reasons"]:
        print("OOD Warnings:")
        for r in result["ood_reasons"]:
            print(f"  * {r}")
    print("Driving factors:")
    for f in result["driving_factors"]:
        print(f"  - {f}")
