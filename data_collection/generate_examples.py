"""
Generates the 4 "how this compares to real runs" examples used in the demo
(frontend validation section + README). Uses genuine LEAVE-ONE-TASK-OUT
predictions (the model never saw the held-out task during training for that
example), not the in-sample final model, so these are honest generalization
examples rather than the model just recalling data it was trained on.

Picked deliberately to show a mix, not just wins:
  - t15, t04: the model's predicted range contains the actual outcome
  - t05: a MISS -- an open-ended task whose real cost compounded past the
    predicted high end (the model underestimates heavy iterative research
    tasks; open_ended is the weakest-calibrated category at 71% LOTO coverage)
  - t17: a MISS -- a trivial zero-tool task where the model overestimates,
    because only 2 of the 20 tasks have zero tools, so leave-one-out starves
    the model of examples at that extreme
"""

import json
from pathlib import Path

from build_dataset import build_dataframe, load_runs
from train_model import BAND_HIGH_QUANTILE, BAND_LOW_QUANTILE, apply_band, leave_one_task_out_residuals

OUTPUT_PATH = Path(__file__).parent / "data" / "validation_examples.json"

SELECTED = {
    "t15": {
        "verdict": "hit",
        "note": "Predicted range comfortably contained the actual cost -- a "
                 "narrow_multi_step task with a clear, bounded set of steps "
                 "(read files, calculate) is the kind of task this model "
                 "handles best.",
    },
    "t04": {
        "verdict": "hit",
        "note": "Open-ended research task, correctly flagged as higher-cost "
                 "by the open-ended-keyword feature, and the actual outcome "
                 "landed inside the predicted band.",
    },
    "t05": {
        "verdict": "miss (underestimate)",
        "note": "The model's high estimate was exceeded: this task chains "
                 "research -> synthesis -> 3 separate emails, and the real "
                 "run's context compounded faster across turns than the "
                 "model predicted. open_ended tasks are the weakest-"
                 "calibrated category overall (71% LOTO coverage vs the "
                 "80% target) -- this is that failure mode in practice.",
    },
    "t17": {
        "verdict": "miss (overestimate)",
        "note": "A trivial zero-tool factual question. The model overshot "
                 "because only 2 of the 20 training tasks have zero tools, "
                 "so leaving either one out starves the model of examples "
                 "at that extreme. A known small-dataset limitation, not a "
                 "systemic flaw.",
    },
}


def main():
    runs = load_runs()
    df = build_dataframe(runs)
    resid_df = leave_one_task_out_residuals(df)

    q_low = resid_df["residual"].quantile(BAND_LOW_QUANTILE)
    q_high = resid_df["residual"].quantile(BAND_HIGH_QUANTILE)
    band = apply_band(resid_df["pred_log"].values, q_low, q_high)
    resid_df["pred_low"] = band["low"]
    resid_df["pred_expected"] = band["expected"]
    resid_df["pred_high"] = band["high"]

    task_lookup = {r["task_id"]: r for r in runs}

    examples = []
    for task_id, meta in SELECTED.items():
        rows = resid_df[resid_df["task_id"] == task_id]
        task = task_lookup[task_id]
        examples.append({
            "task_id": task_id,
            "task_text": task["task_text"],
            "tools_available": task["tools_available"],
            "category": task["category"],
            "actual_tokens_observed": sorted(int(v) for v in rows["actual_tokens"]),
            "pred_low": round(float(rows["pred_low"].mean())),
            "pred_expected": round(float(rows["pred_expected"].mean())),
            "pred_high": round(float(rows["pred_high"].mean())),
            "verdict": meta["verdict"],
            "note": meta["note"],
        })

    OUTPUT_PATH.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} validation examples to {OUTPUT_PATH}")
    for ex in examples:
        print(f"  {ex['task_id']} ({ex['category']}, {ex['verdict']}): "
              f"predicted {ex['pred_low']}-{ex['pred_expected']}-{ex['pred_high']}, "
              f"actual {ex['actual_tokens_observed']}")


if __name__ == "__main__":
    main()
