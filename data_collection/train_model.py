"""
Trains a baseline quantile-regression model (gradient-boosted trees) to
predict a [low, expected, high] token-cost range from task features.

Validation strategy: LEAVE-ONE-TASK-OUT, not leave-one-run-out. The 4 repeat
runs of the same task are correlated (same features), so testing on a repeat
of a task seen in training would overstate how well this generalizes to a
genuinely new task. Held-out means the model has NEVER seen that task's
feature vector during training.

With only 20 unique tasks this is a proof-of-concept validation, not a
statistically powerful one -- reported honestly in the README.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from build_dataset import build_dataframe, load_runs
from features import FEATURE_NAMES

QUANTILES = {"low": 0.1, "expected": 0.5, "high": 0.9}

# Small dataset -> shallow, regularized trees to avoid overfitting the
# ~19-task training folds used during leave-one-task-out validation.
MODEL_PARAMS = dict(
    n_estimators=60,
    max_depth=2,
    learning_rate=0.08,
    subsample=0.8,
    min_samples_leaf=4,
    random_state=0,
)


def fit_quantile_models(X: pd.DataFrame, y: np.ndarray) -> dict:
    models = {}
    for name, q in QUANTILES.items():
        m = GradientBoostingRegressor(loss="quantile", alpha=q, **MODEL_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X, y)
        models[name] = m
    return models


def predict_range(models: dict, X: pd.DataFrame) -> dict:
    """Returns log-token predictions; caller exponentiates back to tokens."""
    preds = {name: models[name].predict(X) for name in QUANTILES}
    # Enforce low <= expected <= high in case quantile crossing occurs
    # (a known artifact of independently-fit quantile GBMs on tiny data).
    stacked = np.vstack([preds["low"], preds["expected"], preds["high"]])
    stacked.sort(axis=0)
    return {"low": stacked[0], "expected": stacked[1], "high": stacked[2]}


def leave_one_task_out_validation(df: pd.DataFrame) -> pd.DataFrame:
    task_ids = df["task_id"].unique()
    records = []

    for held_out in task_ids:
        train_df = df[df["task_id"] != held_out]
        test_df = df[df["task_id"] == held_out]

        X_train = train_df[FEATURE_NAMES]
        y_train = train_df["log_total_tokens"].values

        models = fit_quantile_models(X_train, y_train)

        X_test = test_df[FEATURE_NAMES]
        preds = predict_range(models, X_test)

        low_tok = np.exp(preds["low"])
        exp_tok = np.exp(preds["expected"])
        high_tok = np.exp(preds["high"])

        for i, (_, row) in enumerate(test_df.iterrows()):
            actual = row["total_tokens"]
            records.append({
                "task_id": held_out,
                "category": row["category"],
                "actual_tokens": actual,
                "pred_low": low_tok[i],
                "pred_expected": exp_tok[i],
                "pred_high": high_tok[i],
                "within_range": bool(low_tok[i] <= actual <= high_tok[i]),
            })

    return pd.DataFrame(records)


def main():
    runs = load_runs()
    df = build_dataframe(runs)

    print("=== Leave-one-task-out validation (20 folds) ===\n")
    results = leave_one_task_out_validation(df)

    coverage = results["within_range"].mean()
    print(results.groupby("task_id").agg(
        category=("category", "first"),
        actual_tokens=("actual_tokens", "mean"),
        pred_low=("pred_low", "mean"),
        pred_expected=("pred_expected", "mean"),
        pred_high=("pred_high", "mean"),
        within_range_rate=("within_range", "mean"),
    ).round(0))

    print(f"\nOverall coverage (actual falls within [low, high]): {coverage:.1%}")
    print("(Target ~80% given 10th/90th percentile quantiles used for the band.)")

    by_cat = results.groupby("category")["within_range"].mean()
    print("\nCoverage by category:")
    print(by_cat)

    # Fit the final production model on ALL data (no held-out task) --
    # this is what Phase 3's UI will actually call.
    print("\n=== Fitting final model on full dataset (for production use) ===")
    X_full = df[FEATURE_NAMES]
    y_full = df["log_total_tokens"].values
    final_models = fit_quantile_models(X_full, y_full)

    import joblib
    joblib.dump(final_models, "model.joblib")
    print("Saved final models to model.joblib")


if __name__ == "__main__":
    main()
