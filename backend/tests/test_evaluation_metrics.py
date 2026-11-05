from __future__ import annotations

from decimal import Decimal

import pytest
from evaluation_metrics import (
    MetricObservation,
    evaluate_metrics,
    normalize_anthropic_aggregate,
)


def test_exact_prediction_has_zero_point_error_and_covered_interval():
    result = evaluate_metrics(
        [MetricObservation(100, 100, predicted_low_tokens=80, predicted_high_tokens=120)]
    )

    assert result.mean_absolute_error_tokens.value == 0
    assert result.weighted_absolute_percentage_error_percent.value == 0
    assert result.underprediction_rate_percent.value == 0
    assert result.overprediction_rate_percent.value == 0
    assert result.signed_prediction_bias_tokens.value == 0
    assert result.prediction_interval_coverage_percent.value == 100
    assert result.mean_prediction_interval_width_tokens.value == 40
    assert result.mean_absolute_error_tokens.sample_count == 1


def test_signed_error_positive_is_underprediction_and_negative_is_overprediction():
    result = evaluate_metrics(
        [
            MetricObservation(100, 150, 80, 120),
            MetricObservation(200, 150, 140, 260),
        ]
    )

    assert result.signed_prediction_bias_tokens.value == 0
    assert result.underprediction_rate_percent.value == 50
    assert result.overprediction_rate_percent.value == 50
    assert result.mean_absolute_error_tokens.value == 50


def test_empty_dataset_returns_unavailable_metrics():
    result = evaluate_metrics([])

    for metric in result.to_dict().values():
        assert metric["value"] is None
        assert metric["sample_count"] == 0
        assert metric["unavailable_reason"]


def test_zero_actual_denominator_makes_wape_unavailable():
    result = evaluate_metrics([MetricObservation(0, 0)])

    assert result.mean_absolute_error_tokens.value == 0
    assert result.weighted_absolute_percentage_error_percent.value is None
    assert (
        result.weighted_absolute_percentage_error_percent.unavailable_reason
        == "zero_actual_token_denominator"
    )


def test_missing_actual_is_excluded_and_never_coerced_to_zero():
    result = evaluate_metrics([MetricObservation(0, None), MetricObservation(10, 10)])

    assert result.mean_absolute_error_tokens.value == 0
    assert result.mean_absolute_error_tokens.sample_count == 1
    assert result.mean_absolute_error_tokens.exclusion_counts == {"missing_actual_usage": 1}
    assert result.weighted_absolute_percentage_error_percent.value == 0


def test_invalid_intervals_are_counted_but_do_not_remove_point_metrics():
    result = evaluate_metrics(
        [
            MetricObservation(10, 12, predicted_low_tokens=20, predicted_high_tokens=5),
            MetricObservation(10, 12, predicted_low_tokens=None, predicted_high_tokens=15),
            MetricObservation(10, 12, predicted_low_tokens=0, predicted_high_tokens=20),
        ]
    )

    assert result.mean_absolute_error_tokens.sample_count == 3
    assert result.prediction_interval_coverage_percent.sample_count == 1
    assert result.prediction_interval_coverage_percent.excluded_count == 2
    assert result.prediction_interval_coverage_percent.exclusion_counts == {
        "prediction_interval_low_exceeds_high": 1,
        "missing_prediction_interval": 1,
    }


def test_invalid_negative_interval_bound_is_counted():
    result = evaluate_metrics(
        [MetricObservation(10, 12, predicted_low_tokens=-1, predicted_high_tokens=20)]
    )

    assert result.prediction_interval_coverage_percent.value is None
    assert result.prediction_interval_coverage_percent.exclusion_counts == {
        "invalid_prediction_interval_bound": 1
    }


def test_interval_must_contain_the_expected_prediction():
    result = evaluate_metrics(
        [MetricObservation(10, 12, predicted_low_tokens=0, predicted_high_tokens=9)]
    )

    assert result.mean_absolute_error_tokens.sample_count == 1
    assert result.prediction_interval_coverage_percent.sample_count == 0
    assert result.prediction_interval_coverage_percent.exclusion_counts == {
        "prediction_interval_does_not_contain_expected": 1
    }


def test_p90_uses_nearest_rank_method():
    result = evaluate_metrics([MetricObservation(0, error) for error in range(1, 11)])

    assert result.p90_absolute_error_tokens.value == 9


def test_median_and_p90_preserve_outliers_instead_of_clipping_them():
    result = evaluate_metrics(
        [
            MetricObservation(0, 1),
            MetricObservation(0, 2),
            MetricObservation(0, 3),
            MetricObservation(0, 1000),
        ]
    )

    assert result.mean_absolute_error_tokens.value == 251.5
    assert result.median_absolute_error_tokens.value == 2.5
    assert result.p90_absolute_error_tokens.value == 1000


def test_anthropic_primary_target_uses_raw_cache_exclusive_input_plus_output():
    summary = normalize_anthropic_aggregate(
        event_count=1,
        input_tokens=10,
        output_tokens=4,
        cache_creation_known_sum=3,
        cache_creation_missing_event_count=0,
        cache_read_known_sum=2,
        cache_read_missing_event_count=0,
    )

    assert summary.cache_exclusive_input_tokens == 10
    assert summary.primary_total_tokens == 14
    assert summary.cache_inclusive_input_tokens == 15
    assert summary.cache_inclusive_total_tokens == 19


def test_postgres_numeric_sum_counters_are_normalized_without_precision_loss():
    summary = normalize_anthropic_aggregate(
        event_count=1,
        input_tokens=Decimal("9007199254740993"),
        output_tokens=Decimal("4"),
        cache_creation_known_sum=Decimal("3"),
        cache_creation_missing_event_count=0,
        cache_read_known_sum=Decimal("2"),
        cache_read_missing_event_count=0,
    )

    assert summary.input_tokens == 9007199254740993
    assert summary.primary_total_tokens == 9007199254740997
    assert summary.cache_inclusive_total_tokens == 9007199254741002


def test_null_cache_counters_remain_unknown_without_invalidating_primary_target():
    summary = normalize_anthropic_aggregate(
        event_count=2,
        input_tokens=20,
        output_tokens=6,
        cache_creation_known_sum=3,
        cache_creation_missing_event_count=1,
        cache_read_known_sum=None,
        cache_read_missing_event_count=2,
    )

    assert summary.primary_total_tokens == 26
    assert summary.cache_creation_input_tokens is None
    assert summary.cache_creation_known_sum == 3
    assert summary.cache_creation_missing_event_count == 1
    assert summary.cache_read_input_tokens is None
    assert summary.cache_read_known_sum is None
    assert summary.cache_read_missing_event_count == 2
    assert summary.cache_inclusive_total_tokens is None


def test_negative_usage_counters_are_unavailable():
    summary = normalize_anthropic_aggregate(
        event_count=1,
        input_tokens=10,
        output_tokens=4,
        cache_creation_known_sum=-1,
        cache_creation_missing_event_count=0,
        cache_read_known_sum=0,
        cache_read_missing_event_count=0,
    )

    assert summary.primary_total_tokens is None
    assert summary.unavailable_reason == "negative_or_invalid_usage_counter"


def test_inconsistent_cache_missing_count_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        normalize_anthropic_aggregate(
            event_count=1,
            input_tokens=10,
            output_tokens=4,
            cache_creation_known_sum=0,
            cache_creation_missing_event_count=2,
            cache_read_known_sum=0,
            cache_read_missing_event_count=0,
        )


def test_zero_usage_events_is_missing_not_zero_tokens():
    summary = normalize_anthropic_aggregate(
        event_count=0,
        input_tokens=None,
        output_tokens=None,
        cache_creation_known_sum=None,
        cache_creation_missing_event_count=0,
        cache_read_known_sum=None,
        cache_read_missing_event_count=0,
    )

    assert summary.primary_total_tokens is None
    assert summary.unavailable_reason == "missing_usage_events"
