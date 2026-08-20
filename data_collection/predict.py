"""
Loads the trained model bundle and predicts a [low, expected, high] token
range for a new task description + tool list. This is the exact function
Phase 3's web UI will call.
"""

import numpy as np
import pandas as pd
import joblib

from features import FEATURE_NAMES, extract_features

MODEL_PATH = "model.joblib"


def load_bundle(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(task_text: str, tool_names: list[str], bundle: dict = None) -> dict:
    if bundle is None:
        bundle = load_bundle()

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

    return {
        "low_tokens": round(low),
        "expected_tokens": round(expected),
        "high_tokens": round(high),
        "features": feats,
        "driving_factors": driving_factors,
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
    print(f"Task: {text}")
    print(f"Tools: {tools}")
    print(f"Predicted range: {result['low_tokens']:,} - {result['expected_tokens']:,} - {result['high_tokens']:,} tokens (low/expected/high)")
    print("Driving factors:")
    for f in result["driving_factors"]:
        print(f"  - {f}")
