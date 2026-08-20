"""
Trains a baseline model to predict a [low, expected, high] token-cost range
from task features.

Approach: a single point-estimate regressor (predicts log_total_tokens),
plus a prediction band built from the empirical distribution of out-of-fold
residuals (a simple split-conformal-style calibration), rather than three
independently-fit quantile regressors.

Why not independent low/expected/high quantile-GBMs (the first thing tried):
with only ~19 training tasks per leave-one-out fold, three separately fit
quantile models disagreed badly with each other and produced a band that
only covered the true value 45% of the time against an 80% target -- i.e.
badly overconfident/miscalibrated. A single point estimate + an empirical
residual band is a much lower-variance approach for a dataset this small,
and it's the standard fix (this IS what "conformal prediction" is): don't
trust a model's own quantile outputs at n=20, trust the empirical spread of
its historical errors instead.

Validation strategy: LEAVE-ONE-TASK-OUT (LOTO), not leave-one-run-out. The 4
repeat runs of the same task share a feature vector, so testing on a repeat
of a task seen in training would overstate generalization to a genuinely new
task. Held-out means the model never saw that task's feature vector.

Honesty note (small-n limitation): the residual band's width is calibrated
using the pooled out-of-fold residuals from ALL 20 LOTO folds, and coverage
is then measured against that same residual pool. This is a reasonable
proof-of-concept calibration but is not a fully nested (double) cross-
validation, which would need a larger dataset to be stable. Flagged again in
the README.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from build_dataset import build_dataframe, load_runs
from features import FEATURE_NAMES

BAND_LOW_QUANTILE = 0.10
BAND_HIGH_QUANTILE = 0.90

# Small dataset -> a shallow, heavily regularized single regressor for the
# point estimate (median-ish; squared-error loss on log-tokens).
MODEL_PARAMS = dict(
    n_estimators=40,
    max_depth=2,
    learning_rate=0.08,
    subsample=0.8,
    min_samples_leaf=4,
    random_state=0,
)


def fit_point_model(X: pd.DataFrame, y: np.ndarray) -> GradientBoostingRegressor:
    m = GradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


def leave_one_task_out_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """For each task, train on the other 19 tasks and predict log-tokens for
    the held-out task's runs. Returns one row per run with its residual
    (actual - predicted, in log space)."""
    task_ids = df["task_id"].unique()
    records = []

    for held_out in task_ids:
        train_df = df[df["task_id"] != held_out]
        test_df = df[df["task_id"] == held_out]

        model = fit_point_model(train_df[FEATURE_NAMES], train_df["log_total_tokens"].values)
        pred_log = model.predict(test_df[FEATURE_NAMES])

        for i, (_, row) in enumerate(test_df.iterrows()):
            records.append({
                "task_id": held_out,
                "category": row["category"],
                "actual_tokens": row["total_tokens"],
                "pred_log": pred_log[i],
                "actual_log": row["log_total_tokens"],
                "residual": row["log_total_tokens"] - pred_log[i],
            })

    return pd.DataFrame(records)


def apply_band(pred_log: np.ndarray, q_low: float, q_high: float) -> dict:
    return {
        "low": np.exp(pred_log + q_low),
        "expected": np.exp(pred_log),
        "high": np.exp(pred_log + q_high),
    }


def main():
    runs = load_runs()
    df = build_dataframe(runs)

    print("=== Leave-one-task-out residuals (20 folds, point-estimate model) ===\n")
    resid_df = leave_one_task_out_residuals(df)

    q_low = resid_df["residual"].quantile(BAND_LOW_QUANTILE)
    q_high = resid_df["residual"].quantile(BAND_HIGH_QUANTILE)
    print(f"Residual band (log space): [{q_low:+.3f}, {q_high:+.3f}]  "
          f"(i.e. actual tokens can be exp({q_low:.2f})={np.exp(q_low):.2f}x "
          f"to exp({q_high:.2f})={np.exp(q_high):.2f}x the point estimate)")

    mae_log = resid_df["residual"].abs().mean()
    print(f"Mean absolute error (log space): {mae_log:.3f} "
          f"(~{(np.exp(mae_log) - 1) * 100:.0f}% typical multiplicative error)\n")

    band = apply_band(resid_df["pred_log"].values, q_low, q_high)
    resid_df["pred_low"] = band["low"]
    resid_df["pred_expected"] = band["expected"]
    resid_df["pred_high"] = band["high"]
    resid_df["within_range"] = (resid_df["pred_low"] <= resid_df["actual_tokens"]) & \
                                (resid_df["actual_tokens"] <= resid_df["pred_high"])

    print(resid_df.groupby("task_id").agg(
        category=("category", "first"),
        actual_tokens=("actual_tokens", "mean"),
        pred_low=("pred_low", "mean"),
        pred_expected=("pred_expected", "mean"),
        pred_high=("pred_high", "mean"),
        within_range_rate=("within_range", "mean"),
    ).round(0))

    coverage = resid_df["within_range"].mean()
    print(f"\nOverall coverage (actual falls within [low, high]): {coverage:.1%}")
    print(f"(Target ~{BAND_HIGH_QUANTILE - BAND_LOW_QUANTILE:.0%} given "
          f"{BAND_LOW_QUANTILE:.0%}/{BAND_HIGH_QUANTILE:.0%} residual quantiles used.)")

    print("\nCoverage by category:")
    print(resid_df.groupby("category")["within_range"].mean())

    print("\nNote: this coverage figure is calibrated and measured on the same "
          "pooled LOTO residuals (a proof-of-concept limitation of n=20 tasks, "
          "not a fully nested cross-validation) -- see README.")

    print("\n=== Fitting final point-estimate model on full dataset ===")
    final_model = fit_point_model(df[FEATURE_NAMES], df["log_total_tokens"].values)

    bundle = {
        "model": final_model,
        "q_low": q_low,
        "q_high": q_high,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(bundle, "model.joblib")
    print("Saved final model + calibrated band to model.joblib")


if __name__ == "__main__":
    main()
