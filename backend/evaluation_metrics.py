"""Pure token normalization and prediction evaluation metrics.

This module deliberately has no database, HTTP, or model dependencies. The
Anthropic usage fields are normalized according to the stored raw-counter
contract: ``input_tokens`` already excludes cache creation and cache reads.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One candidate prediction/actual pair and its optional interval."""

    predicted_expected_tokens: int | float | None
    actual_tokens: int | float | None
    predicted_low_tokens: int | float | None = None
    predicted_high_tokens: int | float | None = None


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A metric value with its denominator and excluded-observation reasons."""

    value: float | None
    sample_count: int
    excluded_count: int
    exclusion_counts: dict[str, int]
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mean_absolute_error_tokens: MetricValue
    weighted_absolute_percentage_error_percent: MetricValue
    median_absolute_error_tokens: MetricValue
    p90_absolute_error_tokens: MetricValue
    underprediction_rate_percent: MetricValue
    overprediction_rate_percent: MetricValue
    signed_prediction_bias_tokens: MetricValue
    prediction_interval_coverage_percent: MetricValue
    mean_prediction_interval_width_tokens: MetricValue

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Return a JSON-ready mapping while preserving per-metric counts."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnthropicUsageSummary:
    """Normalized primary target and cache statistics from SQL aggregates."""

    primary_total_tokens: int | None
    cache_exclusive_input_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_creation_known_sum: int | None
    cache_creation_missing_event_count: int
    cache_read_input_tokens: int | None
    cache_read_known_sum: int | None
    cache_read_missing_event_count: int
    cache_inclusive_input_tokens: int | None
    cache_inclusive_total_tokens: int | None
    unavailable_reason: str | None = None


def _valid_nonnegative_number(value: int | float | None) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _invalid_pair_reason(observation: MetricObservation) -> str | None:
    if observation.actual_tokens is None:
        return "missing_actual_usage"
    if not _valid_nonnegative_number(observation.actual_tokens):
        return "invalid_actual_usage"
    if not _valid_nonnegative_number(observation.predicted_expected_tokens):
        return "invalid_prediction"
    return None


def _metric_value(
    values: list[float],
    exclusions: dict[str, int],
    *,
    unavailable_reason: str | None = None,
) -> MetricValue:
    sample_count = len(values)
    if not sample_count:
        return MetricValue(
            value=None,
            sample_count=0,
            excluded_count=sum(exclusions.values()),
            exclusion_counts=exclusions,
            unavailable_reason=unavailable_reason or "no_valid_observations",
        )
    return MetricValue(
        value=None,
        sample_count=sample_count,
        excluded_count=sum(exclusions.values()),
        exclusion_counts=exclusions,
        unavailable_reason=unavailable_reason,
    )


def _p90_nearest_rank(values: list[float]) -> float:
    """Return the nearest-rank 90th percentile (rank ceil(0.9 * n))."""
    ordered = sorted(values)
    rank = math.ceil(0.9 * len(ordered))
    return ordered[rank - 1]


def evaluate_metrics(observations: Iterable[MetricObservation]) -> EvaluationMetrics:
    """Calculate point and interval metrics without silently dropping rows.

    Signed error is ``actual - predicted``: positive bias means
    underprediction and negative bias means overprediction. Percentage rates
    use 0--100 units. Exact predictions are in both rate denominators but in
    neither directional numerator. WAPE is unavailable when total actual
    tokens are zero.
    """
    rows = list(observations)
    valid: list[tuple[float, float, MetricObservation]] = []
    pair_exclusions: dict[str, int] = {}
    for observation in rows:
        reason = _invalid_pair_reason(observation)
        if reason:
            pair_exclusions[reason] = pair_exclusions.get(reason, 0) + 1
            continue
        predicted = float(observation.predicted_expected_tokens)
        actual = float(observation.actual_tokens)
        valid.append((predicted, actual, observation))

    errors = [actual - predicted for predicted, actual, _ in valid]
    absolute_errors = [abs(error) for error in errors]
    metric_count = len(valid)
    if metric_count:
        mae = sum(absolute_errors) / metric_count
        actual_sum = sum(actual for _, actual, _ in valid)
        wape = 100.0 * sum(absolute_errors) / actual_sum if actual_sum > 0 else None
        wape_reason = None if actual_sum > 0 else "zero_actual_token_denominator"
        median_absolute = float(median(absolute_errors))
        p90_absolute = _p90_nearest_rank(absolute_errors)
        underprediction_rate = 100.0 * sum(error > 0 for error in errors) / metric_count
        overprediction_rate = 100.0 * sum(error < 0 for error in errors) / metric_count
        signed_bias = sum(errors) / metric_count
    else:
        mae = median_absolute = p90_absolute = None
        wape = underprediction_rate = overprediction_rate = signed_bias = None
        wape_reason = "no_valid_observations"

    point_metrics = {
        "mae": MetricValue(
        mae, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
        "wape": MetricValue(
        wape, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        wape_reason,
        ),
        "median": MetricValue(
        median_absolute, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
        "p90": MetricValue(
        p90_absolute, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
        "under": MetricValue(
        underprediction_rate, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
        "over": MetricValue(
        overprediction_rate, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
        "bias": MetricValue(
        signed_bias, metric_count, sum(pair_exclusions.values()), dict(pair_exclusions),
        None if metric_count else "no_valid_observations",
        ),
    }

    interval_coverage_values: list[float] = []
    interval_widths: list[float] = []
    interval_exclusions = dict(pair_exclusions)
    for predicted, actual, observation in valid:
        low = observation.predicted_low_tokens
        high = observation.predicted_high_tokens
        if low is None or high is None:
            reason = "missing_prediction_interval"
        elif not _valid_nonnegative_number(low) or not _valid_nonnegative_number(high):
            reason = "invalid_prediction_interval_bound"
        elif low > high:
            reason = "prediction_interval_low_exceeds_high"
        else:
            low_value = float(low)
            high_value = float(high)
            interval_coverage_values.append(
                100.0 if low_value <= actual <= high_value else 0.0
            )
            interval_widths.append(high_value - low_value)
            continue
        interval_exclusions[reason] = interval_exclusions.get(reason, 0) + 1

    interval_count = len(interval_widths)
    coverage = (
        sum(interval_coverage_values) / interval_count if interval_count else None
    )
    mean_width = sum(interval_widths) / interval_count if interval_count else None
    interval_unavailable = None if interval_count else "no_valid_prediction_intervals"

    return EvaluationMetrics(
        mean_absolute_error_tokens=point_metrics["mae"],
        weighted_absolute_percentage_error_percent=point_metrics["wape"],
        median_absolute_error_tokens=point_metrics["median"],
        p90_absolute_error_tokens=point_metrics["p90"],
        underprediction_rate_percent=point_metrics["under"],
        overprediction_rate_percent=point_metrics["over"],
        signed_prediction_bias_tokens=point_metrics["bias"],
        prediction_interval_coverage_percent=MetricValue(
            coverage,
            interval_count,
            sum(interval_exclusions.values()),
            dict(interval_exclusions),
            interval_unavailable,
        ),
        mean_prediction_interval_width_tokens=MetricValue(
            mean_width,
            interval_count,
            sum(interval_exclusions.values()),
            dict(interval_exclusions),
            interval_unavailable,
        ),
    )


def normalize_anthropic_aggregate(
    *,
    event_count: int,
    input_tokens: int | None,
    output_tokens: int | None,
    cache_creation_known_sum: int | None,
    cache_creation_missing_event_count: int,
    cache_read_known_sum: int | None,
    cache_read_missing_event_count: int,
) -> AnthropicUsageSummary:
    """Normalize one SQL-aggregated Anthropic execution without null coercion.

    Anthropic's raw ``input_tokens`` is already the cache-exclusive input
    component. Cache fields are parallel input breakdowns and are never
    subtracted from it. Missing cache counters make cache-inclusive totals
    unavailable, but do not change the cache-exclusive primary target.
    """
    if event_count < 0:
        raise ValueError("event_count cannot be negative")
    if not 0 <= cache_creation_missing_event_count <= event_count:
        raise ValueError("cache creation missing count is inconsistent with event count")
    if not 0 <= cache_read_missing_event_count <= event_count:
        raise ValueError("cache read missing count is inconsistent with event count")
    raw_values = (input_tokens, output_tokens, cache_creation_known_sum, cache_read_known_sum)
    if any(
        value is not None
        and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
        for value in raw_values
    ):
        return AnthropicUsageSummary(
            primary_total_tokens=None,
            cache_exclusive_input_tokens=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_creation_known_sum=cache_creation_known_sum,
            cache_creation_missing_event_count=cache_creation_missing_event_count,
            cache_read_input_tokens=None,
            cache_read_known_sum=cache_read_known_sum,
            cache_read_missing_event_count=cache_read_missing_event_count,
            cache_inclusive_input_tokens=None,
            cache_inclusive_total_tokens=None,
            unavailable_reason="negative_or_invalid_usage_counter",
        )
    if event_count == 0:
        reason = "missing_usage_events"
        primary_total = None
        exclusive_input = None
    elif input_tokens is None or output_tokens is None:
        reason = "missing_required_input_or_output_counter"
        primary_total = None
        exclusive_input = None
    else:
        reason = None
        exclusive_input = input_tokens
        primary_total = input_tokens + output_tokens

    cache_creation_complete = cache_creation_missing_event_count == 0 and event_count > 0
    cache_read_complete = cache_read_missing_event_count == 0 and event_count > 0
    cache_creation_total = cache_creation_known_sum if cache_creation_complete else None
    cache_read_total = cache_read_known_sum if cache_read_complete else None
    caches_complete = (
        cache_creation_complete
        and cache_read_complete
        and cache_creation_total is not None
        and cache_read_total is not None
    )
    if input_tokens is not None and caches_complete:
        cache_inclusive_input = input_tokens + cache_creation_total + cache_read_total
        cache_inclusive_total = (
            cache_inclusive_input + output_tokens if output_tokens is not None else None
        )
    else:
        cache_inclusive_input = None
        cache_inclusive_total = None

    return AnthropicUsageSummary(
        primary_total_tokens=primary_total,
        cache_exclusive_input_tokens=exclusive_input,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_total,
        cache_creation_known_sum=cache_creation_known_sum,
        cache_creation_missing_event_count=cache_creation_missing_event_count,
        cache_read_input_tokens=cache_read_total,
        cache_read_known_sum=cache_read_known_sum,
        cache_read_missing_event_count=cache_read_missing_event_count,
        cache_inclusive_input_tokens=cache_inclusive_input,
        cache_inclusive_total_tokens=cache_inclusive_total,
        unavailable_reason=reason,
    )
