"""
Builds the training dataframe from data/runs.jsonl: one row per run, with
features extracted from the run's task_text + tools_available, and the
target (total_tokens, log-transformed since token cost is heavy-right-skewed).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, extract_features

RUNS_PATH = Path(__file__).parent / "data" / "runs.jsonl"


def load_runs(path: Path = RUNS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataframe(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        feats = extract_features(r["task_text"], r["tools_available"])
        row = dict(feats)
        row["task_id"] = r["task_id"]
        row["category"] = r["category"]  # kept for analysis, not used as a model feature
        row["total_tokens"] = r["total_tokens"]
        row["log_total_tokens"] = np.log(r["total_tokens"])
        row["num_turns"] = r["num_turns"]
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataframe(load_runs())
    print(df[["task_id", "category", "num_tools", "open_ended_keyword_hits",
              "narrow_keyword_hits", "num_clauses", "total_tokens"]]
          .groupby("task_id").first())
    print()
    print(f"{len(df)} rows, {df['task_id'].nunique()} unique tasks")
    print("Feature columns:", FEATURE_NAMES)
