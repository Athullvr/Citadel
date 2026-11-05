"""Workspace-scoped, conservative residual profiles for prediction estimates."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from evaluation_service import (
    EvaluationFilters,
    _EligibleRecord,
    load_evaluation_dataset,
)
from execution_models import TELEMETRY_PROVENANCE_VERIFIED_PROVIDER
from sqlalchemy.orm import Session

PROFILE_METHODOLOGY_VERSION = "citadel-adaptive-profile-v1"
PROFILE_LOOKBACK_DAYS = 180
MIN_INDEPENDENT_TASKS = 30
MEDIAN_CI_CONFIDENCE = 0.95
MAX_ADJUSTMENT_FRACTION = 0.50
SUPPORTED_PROVIDER = "anthropic"
SUPPORTED_PREDICTION_MODEL = "claude-sonnet"


@dataclass(frozen=True, slots=True)
class _Cohort:
    status: str
    execution_sample_count: int
    independent_task_count: int
    collapsed_repeat_count: int
    median_residual_tokens: float | None
    median_ci_low_tokens: float | None
    median_ci_high_tokens: float | None
    exclusion_counts: dict[str, int]
    provenance_counts: dict[str, int]
    fallback_reason: str | None


def distribution_free_median_confidence_interval(
    values: list[float], confidence: float = MEDIAN_CI_CONFIDENCE
) -> tuple[float, float] | None:
    """Return an exact order-statistic CI for a population median.

    The interval uses the binomial sign-test distribution and has coverage at
    least ``confidence`` when observations are independent.  It returns None
    if the sample cannot support the requested confidence level.
    """
    if not values or not 0.0 < confidence < 1.0:
        return None
    ordered = sorted(values)
    tail_probability = (1.0 - confidence) / 2.0
    log_tail_probability = math.log(tail_probability)
    n = len(ordered)
    log_cdf = -math.inf
    rank = 0
    for successes in range(n // 2 + 1):
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(n - successes + 1)
            - n * math.log(2.0)
        )
        if log_cdf == -math.inf:
            log_cdf = log_pmf
        else:
            larger = max(log_cdf, log_pmf)
            log_cdf = larger + math.log(math.exp(log_cdf - larger) + math.exp(log_pmf - larger))
        if log_cdf <= log_tail_probability:
            rank = successes + 1
        else:
            break
    if rank == 0:
        return None
    return ordered[rank - 1], ordered[n - rank]


def _empty_cohort(
    status: str,
    reason: str | None,
    *,
    exclusions: Counter[str] | None = None,
    provenance: Counter[str] | None = None,
) -> _Cohort:
    return _Cohort(
        status=status,
        execution_sample_count=0,
        independent_task_count=0,
        collapsed_repeat_count=0,
        median_residual_tokens=None,
        median_ci_low_tokens=None,
        median_ci_high_tokens=None,
        exclusion_counts=dict(exclusions or {}),
        provenance_counts=dict(provenance or {}),
        fallback_reason=reason,
    )


def _cohort_for_records(
    records: list[_EligibleRecord],
    *,
    expected_artifact_version: str,
    exact_provider_model: str | None,
) -> _Cohort:
    scoped_records = [
        record
        for record in records
        if exact_provider_model is None
        or record.exact_provider_model == exact_provider_model
    ]
    provenance = Counter(record.provenance for record in scoped_records)
    exclusions: Counter[str] = Counter()
    task_residuals: dict[str, list[float]] = defaultdict(list)
    compatible_execution_count = 0

    for record in scoped_records:
        if record.is_synthetic:
            exclusions["is_synthetic"] += 1
            continue
        if record.provenance != TELEMETRY_PROVENANCE_VERIFIED_PROVIDER:
            exclusions[f"provenance_{record.provenance}"] += 1
            continue
        if record.model_artifact_version != expected_artifact_version:
            exclusions["prediction_artifact_mismatch"] += 1
            continue
        if record.task_hash is None:
            exclusions["task_hash_missing"] += 1
            continue
        actual = record.metric_observation.actual_tokens
        predicted = record.metric_observation.predicted_expected_tokens
        if actual is None or predicted is None:
            exclusions["invalid_prediction_or_actual"] += 1
            continue
        compatible_execution_count += 1
        task_residuals[record.task_hash].append(float(actual) - float(predicted))

    independent_residuals = [float(median(residuals)) for residuals in task_residuals.values()]
    independent_count = len(independent_residuals)
    repeats_collapsed = compatible_execution_count - independent_count
    reason: str | None
    if not scoped_records:
        return _empty_cohort("unavailable", "no_evaluator_eligible_history")
    if not independent_count:
        verified_records = [
            record
            for record in scoped_records
            if record.provenance == TELEMETRY_PROVENANCE_VERIFIED_PROVIDER
            and not record.is_synthetic
        ]
        has_matching_artifact = any(
            record.model_artifact_version == expected_artifact_version
            for record in verified_records
        )
        status = (
            "incompatible"
            if verified_records and not has_matching_artifact
            else "insufficient_data"
        )
        reason = (
            "prediction_artifact_mismatch"
            if status == "incompatible"
            else "no_verified_independent_observations"
        )
        return _Cohort(
            status=status,
            execution_sample_count=0,
            independent_task_count=0,
            collapsed_repeat_count=0,
            median_residual_tokens=None,
            median_ci_low_tokens=None,
            median_ci_high_tokens=None,
            exclusion_counts=dict(exclusions),
            provenance_counts=dict(provenance),
            fallback_reason=reason,
        )

    residual_median = float(median(independent_residuals))
    interval = distribution_free_median_confidence_interval(independent_residuals)
    if independent_count < MIN_INDEPENDENT_TASKS:
        status = "insufficient_data"
        reason = "insufficient_independent_tasks"
    elif interval is None or interval[0] <= 0 <= interval[1]:
        status = "insufficient_data"
        reason = "median_confidence_interval_includes_zero"
    else:
        status = "active"
        reason = None
    return _Cohort(
        status=status,
        execution_sample_count=compatible_execution_count,
        independent_task_count=independent_count,
        collapsed_repeat_count=repeats_collapsed,
        median_residual_tokens=residual_median,
        median_ci_low_tokens=interval[0] if interval else None,
        median_ci_high_tokens=interval[1] if interval else None,
        exclusion_counts=dict(exclusions),
        provenance_counts=dict(provenance),
        fallback_reason=reason,
    )


def _cohort_dict(cohort: _Cohort) -> dict[str, Any]:
    return {
        "status": cohort.status,
        "execution_sample_count": cohort.execution_sample_count,
        "independent_task_count": cohort.independent_task_count,
        "collapsed_repeat_count": cohort.collapsed_repeat_count,
        "median_residual_tokens": cohort.median_residual_tokens,
        "median_residual_confidence_interval_tokens": (
            {
                "confidence": MEDIAN_CI_CONFIDENCE,
                "method": "distribution_free_order_statistic_sign_test",
                "low": cohort.median_ci_low_tokens,
                "high": cohort.median_ci_high_tokens,
            }
            if cohort.median_ci_low_tokens is not None
            and cohort.median_ci_high_tokens is not None
            else None
        ),
        "exclusion_counts": cohort.exclusion_counts,
        "provenance_counts": cohort.provenance_counts,
        "fallback_reason": cohort.fallback_reason,
    }


def _with_stale_fallback(recent: _Cohort, historical: _Cohort) -> _Cohort:
    if recent.status == "active":
        return recent
    if historical.status == "active":
        return _Cohort(
            status="stale",
            execution_sample_count=historical.execution_sample_count,
            independent_task_count=historical.independent_task_count,
            collapsed_repeat_count=historical.collapsed_repeat_count,
            median_residual_tokens=historical.median_residual_tokens,
            median_ci_low_tokens=historical.median_ci_low_tokens,
            median_ci_high_tokens=historical.median_ci_high_tokens,
            exclusion_counts=historical.exclusion_counts,
            provenance_counts=historical.provenance_counts,
            fallback_reason="active_evidence_outside_180_day_window",
        )
    return recent


def _adjustment_bounds(
    residual: float,
    *,
    low_tokens: int,
    expected_tokens: int,
    high_tokens: int,
) -> tuple[int, bool, str | None]:
    """Bound one shared interval shift by baseline scale and interval width."""
    interval_width = max(0, high_tokens - low_tokens)
    limit = min(int(expected_tokens * MAX_ADJUSTMENT_FRACTION), interval_width)
    lower_bound = max(-limit, -low_tokens)
    upper_bound = limit
    rounded = int(math.copysign(math.floor(abs(residual) + 0.5), residual))
    adjustment = max(lower_bound, min(upper_bound, rounded))
    if adjustment == 0 and rounded != 0:
        return 0, True, "prediction_interval_does_not_allow_a_safe_adjustment"
    return adjustment, adjustment != rounded, None


def _profile_status_without_query(
    reason: str, *, evaluated_at: datetime
) -> dict[str, Any]:
    cutoff = evaluated_at - timedelta(days=PROFILE_LOOKBACK_DAYS)
    return {
        "status": "incompatible",
        "profile_level": "global_baseline",
        "profile_sample_count": 0,
        "profile_execution_sample_count": 0,
        "independent_task_count": 0,
        "adjustment_tokens": 0,
        "median_residual_tokens": None,
        "median_residual_confidence_interval_tokens": None,
        "adjustment_clipped": False,
        "fallback_reason": reason,
        "methodology_version": PROFILE_METHODOLOGY_VERSION,
        "evaluation_window": {
            "start_at": cutoff.isoformat(),
            "end_at": evaluated_at.isoformat(),
            "basis": "execution.started_at; start inclusive, end exclusive",
            "stale_before": cutoff.isoformat(),
        },
        "profiles": {
            "workspace": _cohort_dict(_empty_cohort("incompatible", reason)),
            "provider_model": _cohort_dict(_empty_cohort("incompatible", reason)),
            "project": {
                "status": "unavailable",
                "fallback_reason": "project_identity_not_stored",
            },
            "task_category": {
                "status": "unavailable",
                "fallback_reason": "task_category_not_stored",
            },
        },
        "eligibility": {
            "provenance_required": TELEMETRY_PROVENANCE_VERIFIED_PROVIDER,
            "task_hash_required": True,
            "minimum_independent_tasks": MIN_INDEPENDENT_TASKS,
            "lookback_days": PROFILE_LOOKBACK_DAYS,
        },
    }


def get_adaptive_profile(
    db: Session,
    *,
    workspace_id: str,
    provider: str,
    exact_provider_model: str,
    model_alias: str,
    model_artifact_version: str | None,
    low_tokens: int,
    expected_tokens: int,
    high_tokens: int,
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Derive an adjustment from canonical eligible executions in one workspace."""
    evaluated_at = evaluated_at or datetime.now(UTC)
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone")
    evaluated_at = evaluated_at.astimezone(UTC)
    if provider != SUPPORTED_PROVIDER:
        return _profile_status_without_query(
            "unsupported_provider_token_semantics", evaluated_at=evaluated_at
        )
    if model_alias != SUPPORTED_PREDICTION_MODEL:
        return _profile_status_without_query(
            "unsupported_prediction_model", evaluated_at=evaluated_at
        )
    if not model_artifact_version:
        return _profile_status_without_query(
            "prediction_artifact_unavailable", evaluated_at=evaluated_at
        )
    if min(low_tokens, expected_tokens, high_tokens) < 0 or not (
        low_tokens <= expected_tokens <= high_tokens
    ):
        return _profile_status_without_query(
            "invalid_baseline_prediction", evaluated_at=evaluated_at
        )

    cutoff = evaluated_at - timedelta(days=PROFILE_LOOKBACK_DAYS)
    recent_data = load_evaluation_dataset(
        db,
        workspace_id=workspace_id,
        filters=EvaluationFilters(
            start_at=cutoff,
            end_at=evaluated_at,
            model_alias=model_alias,
            provider=provider,
        ),
    )
    historical_data = load_evaluation_dataset(
        db,
        workspace_id=workspace_id,
        filters=EvaluationFilters(
            end_at=cutoff,
            model_alias=model_alias,
            provider=provider,
        ),
    )

    workspace_recent = _cohort_for_records(
        recent_data.completed_records,
        expected_artifact_version=model_artifact_version,
        exact_provider_model=None,
    )
    workspace_historical = _cohort_for_records(
        historical_data.completed_records,
        expected_artifact_version=model_artifact_version,
        exact_provider_model=None,
    )
    model_recent = _cohort_for_records(
        recent_data.completed_records,
        expected_artifact_version=model_artifact_version,
        exact_provider_model=exact_provider_model,
    )
    model_historical = _cohort_for_records(
        historical_data.completed_records,
        expected_artifact_version=model_artifact_version,
        exact_provider_model=exact_provider_model,
    )
    workspace = _with_stale_fallback(workspace_recent, workspace_historical)
    provider_model = _with_stale_fallback(model_recent, model_historical)

    selected: _Cohort | None = None
    selected_level = "global_baseline"
    # Prefer the narrowest active evidence, then fall back to the workspace cohort.
    if provider_model.status == "active":
        selected = provider_model
        selected_level = "provider_model"
    elif workspace.status == "active":
        selected = workspace
        selected_level = "workspace"

    adjustment = 0
    clipped = False
    adjustment_bound_reason: str | None = None
    if selected is not None and selected.median_residual_tokens is not None:
        adjustment, clipped, adjustment_bound_reason = _adjustment_bounds(
            selected.median_residual_tokens,
            low_tokens=low_tokens,
            expected_tokens=expected_tokens,
            high_tokens=high_tokens,
        )

    if selected is not None:
        status = "active"
        fallback_reason = adjustment_bound_reason
    else:
        candidates = (provider_model, workspace)
        if any(candidate.status == "stale" for candidate in candidates):
            status = "stale"
            fallback_reason = "no_active_profile_within_180_day_window"
        elif any(candidate.status == "incompatible" for candidate in candidates):
            status = "incompatible"
            fallback_reason = next(
                candidate.fallback_reason
                for candidate in candidates
                if candidate.status == "incompatible"
            )
        elif any(candidate.status == "insufficient_data" for candidate in candidates):
            status = "insufficient_data"
            fallback_reason = next(
                candidate.fallback_reason
                for candidate in (provider_model, workspace)
                if candidate.status == "insufficient_data"
            )
        else:
            status = "unavailable"
            fallback_reason = "no_evaluator_eligible_history"

    applied_residual = selected.median_residual_tokens if selected else None
    recent_provenance = Counter(
        record.provenance for record in recent_data.completed_records
    )
    profile_exclusions = Counter(recent_data.exclusion_counts)
    if recent_data.failed_records:
        profile_exclusions["failed_execution_not_used_for_prediction_accuracy"] += len(
            recent_data.failed_records
        )
    result = {
        "status": status,
        "profile_level": selected_level,
        "profile_sample_count": selected.independent_task_count if selected else 0,
        "profile_execution_sample_count": selected.execution_sample_count if selected else 0,
        "adjustment_tokens": adjustment,
        "median_residual_tokens": applied_residual,
        "median_residual_confidence_interval_tokens": (
            _cohort_dict(selected)["median_residual_confidence_interval_tokens"]
            if selected
            else None
        ),
        "adjustment_clipped": clipped,
        "fallback_reason": fallback_reason,
        "methodology_version": PROFILE_METHODOLOGY_VERSION,
        "evaluation_window": {
            "start_at": cutoff.isoformat(),
            "end_at": evaluated_at.isoformat(),
            "basis": "execution.started_at; start inclusive, end exclusive",
            "stale_before": cutoff.isoformat(),
        },
        "eligibility": {
            "provenance_required": TELEMETRY_PROVENANCE_VERIFIED_PROVIDER,
            "task_hash_required": True,
            "minimum_independent_tasks": MIN_INDEPENDENT_TASKS,
            "lookback_days": PROFILE_LOOKBACK_DAYS,
            "confidence_level": MEDIAN_CI_CONFIDENCE,
            "confidence_method": "distribution_free_order_statistic_sign_test",
            "maximum_adjustment_fraction_of_expected": MAX_ADJUSTMENT_FRACTION,
            "adjustment_bounds_tokens": {
                "minimum": max(
                    -min(
                        int(expected_tokens * MAX_ADJUSTMENT_FRACTION),
                        max(0, high_tokens - low_tokens),
                    ),
                    -low_tokens,
                ),
                "maximum": min(
                    int(expected_tokens * MAX_ADJUSTMENT_FRACTION),
                    max(0, high_tokens - low_tokens),
                ),
            },
            "primary_token_target": "anthropic.cache_exclusive_input_plus_output.v1",
            "evaluator_excluded_execution_counts": recent_data.exclusion_counts,
            "eligible_completed_execution_count": len(recent_data.completed_records),
            "eligible_failed_execution_count": len(recent_data.failed_records),
            "profile_excluded_execution_counts": dict(profile_exclusions),
            "provenance_distribution": dict(recent_provenance),
            "workspace_profile_exclusions": workspace_recent.exclusion_counts,
        },
        "profiles": {
            "workspace": _cohort_dict(workspace),
            "provider_model": _cohort_dict(provider_model),
            "project": {
                "status": "unavailable",
                "fallback_reason": "project_identity_not_stored",
            },
            "task_category": {
                "status": "unavailable",
                "fallback_reason": "task_category_not_stored",
            },
        },
    }
    return result


def adjusted_token_interval(
    low_tokens: int,
    expected_tokens: int,
    high_tokens: int,
    adjustment_tokens: int,
) -> dict[str, int]:
    """Apply one bounded integer shift while preserving interval ordering."""
    return {
        "low_tokens": low_tokens + adjustment_tokens,
        "expected_tokens": expected_tokens + adjustment_tokens,
        "high_tokens": high_tokens + adjustment_tokens,
    }
