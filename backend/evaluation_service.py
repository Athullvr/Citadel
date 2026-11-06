"""Read-only execution evaluation queries and report assembly."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from evaluation_metrics import (
    AnthropicUsageSummary,
    MetricObservation,
    evaluate_metrics,
    normalize_anthropic_aggregate,
)
from execution_models import (
    TELEMETRY_PROVENANCE_SYNTHETIC_BENCHMARK,
    TELEMETRY_PROVENANCE_UNVERIFIED_SUBMITTED,
    TELEMETRY_PROVENANCE_VERIFIED_PROVIDER,
    ExecutionRecord,
    PredictionRecord,
    UsageEvent,
)
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

METHODOLOGY_VERSION = "citadel-token-evaluation-v1"
PRIMARY_TOKEN_TARGET = "anthropic.cache_exclusive_input_plus_output.v1"
MIN_INDEPENDENT_TASKS = 30
PROVENANCE_CATEGORIES = (
    TELEMETRY_PROVENANCE_VERIFIED_PROVIDER,
    TELEMETRY_PROVENANCE_UNVERIFIED_SUBMITTED,
    TELEMETRY_PROVENANCE_SYNTHETIC_BENCHMARK,
    "unknown",
)


@dataclass(frozen=True, slots=True)
class EvaluationFilters:
    start_at: datetime | None = None
    end_at: datetime | None = None
    model_alias: str | None = None
    model_artifact_version: str | None = None
    provider: str | None = None
    exact_provider_model: str | None = None
    integration_name: str | None = None
    integration_version: str | None = None


@dataclass(frozen=True, slots=True)
class _EligibleRecord:
    execution_id: UUID
    prediction_id: UUID
    status: str
    started_at: datetime
    provenance: str
    is_synthetic: bool
    task_hash: str | None
    model_alias: str
    model_artifact_version: str | None
    provider: str
    exact_provider_model: str
    integration_name: str
    integration_version: str | None
    metric_observation: MetricObservation
    usage_summary: AnthropicUsageSummary


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Internal execution-level records and exclusions shared by evaluators."""

    total_execution_count: int
    missing_usage_event_execution_count: int
    completeness_counts: dict[str, int]
    provenance_counts: dict[str, int]
    exclusion_counts: dict[str, int]
    completed_records: list[_EligibleRecord]
    failed_records: list[_EligibleRecord]


def _provenance(value: str | None) -> str:
    if value in PROVENANCE_CATEGORIES:
        return value
    return "unknown"


def _metric_class(provenance: str) -> str:
    return {
        TELEMETRY_PROVENANCE_VERIFIED_PROVIDER: "verified_production",
        TELEMETRY_PROVENANCE_UNVERIFIED_SUBMITTED: "descriptive_unverified",
        TELEMETRY_PROVENANCE_SYNTHETIC_BENCHMARK: "synthetic_benchmark",
        "unknown": "descriptive_unknown",
    }[provenance]


def _completeness(expected_count: int | None, actual_count: int) -> str:
    if expected_count is None:
        return "unknown"
    return "complete" if expected_count == actual_count else "incomplete"


def _cache_statistics(records: list[_EligibleRecord]) -> dict[str, Any]:
    if not records:
        return {
            "cache_exclusive_input_tokens": None,
            "output_tokens": None,
            "primary_total_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_creation_known_sum": None,
            "cache_creation_missing_event_count": 0,
            "cache_read_input_tokens": None,
            "cache_read_known_sum": None,
            "cache_read_missing_event_count": 0,
            "cache_inclusive_input_tokens": None,
            "cache_inclusive_total_tokens": None,
        }

    summaries = [record.usage_summary for record in records]

    def aggregate(field: str, known_field: str, missing_field: str) -> dict[str, int | None]:
        known_values = [getattr(summary, known_field) for summary in summaries]
        known_present = [value for value in known_values if value is not None]
        missing_events = sum(getattr(summary, missing_field) for summary in summaries)
        fully_known = all(getattr(summary, missing_field) == 0 for summary in summaries)
        known_sum = sum(known_present) if known_present else None
        return {
            field: known_sum if fully_known and known_sum is not None else None,
            known_field: known_sum,
            missing_field: missing_events,
        }

    creation = aggregate(
        "cache_creation_input_tokens",
        "cache_creation_known_sum",
        "cache_creation_missing_event_count",
    )
    reads = aggregate(
        "cache_read_input_tokens",
        "cache_read_known_sum",
        "cache_read_missing_event_count",
    )
    input_values = [summary.input_tokens for summary in summaries]
    output_values = [summary.output_tokens for summary in summaries]
    input_total = sum(value for value in input_values if value is not None)
    output_total = sum(value for value in output_values if value is not None)
    cache_inclusive_input = None
    cache_inclusive_total = None
    if (
        creation["cache_creation_input_tokens"] is not None
        and reads["cache_read_input_tokens"] is not None
    ):
        cache_inclusive_input = (
            input_total
            + int(creation["cache_creation_input_tokens"])
            + int(reads["cache_read_input_tokens"])
        )
        cache_inclusive_total = cache_inclusive_input + output_total
    return {
        "cache_exclusive_input_tokens": input_total,
        "output_tokens": output_total,
        "primary_total_tokens": input_total + output_total,
        **creation,
        **reads,
        "cache_inclusive_input_tokens": cache_inclusive_input,
        "cache_inclusive_total_tokens": cache_inclusive_total,
    }


def _reliability(records: list[_EligibleRecord]) -> tuple[int, int, str]:
    hashed = [record.task_hash for record in records if record.task_hash is not None]
    unhashed_count = len(records) - len(hashed)
    independent_count = len(set(hashed))
    if not records:
        return independent_count, unhashed_count, "unavailable"
    if unhashed_count:
        return independent_count, unhashed_count, "task_independence_unknown"
    if independent_count < MIN_INDEPENDENT_TASKS:
        return independent_count, unhashed_count, "insufficient_independent_tasks"
    return independent_count, unhashed_count, "provisional"


def _group_report(
    records: list[_EligibleRecord],
    *,
    provenance: str,
    key: dict[str, str | None],
    outcome: str,
) -> dict[str, Any]:
    metrics = evaluate_metrics(record.metric_observation for record in records).to_dict()
    independent_count, unhashed_count, reliability = _reliability(records)
    return {
        "provenance": provenance,
        "metric_class": _metric_class(provenance),
        "outcome": outcome,
        "key": key,
        "sample_count": len(records),
        "independent_task_count": independent_count,
        "task_hash_missing_count": unhashed_count,
        "reliability": reliability,
        "production_accuracy_claim_eligible": (
            provenance == TELEMETRY_PROVENANCE_VERIFIED_PROVIDER
            and reliability == "provisional"
        ),
        "metrics": metrics,
        "cache_usage": _cache_statistics(records),
    }


def _segments(records: list[_EligibleRecord], outcome: str) -> dict[str, list[dict[str, Any]]]:
    dimensions: dict[str, tuple[str, ...]] = {
        "provenance": ("provenance",),
        "model_artifact": ("model_alias", "model_artifact_version"),
        "provider_model": ("provider", "exact_provider_model"),
        "integration": ("integration_name", "integration_version"),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for name, fields in dimensions.items():
        groups: dict[tuple[Any, ...], list[_EligibleRecord]] = defaultdict(list)
        for record in records:
            group_key = (
                record.provenance,
                *(getattr(record, field) for field in fields),
            )
            groups[group_key].append(record)
        reports: list[dict[str, Any]] = []
        for group_key, values in sorted(
            groups.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            group_provenance = str(group_key[0])
            dimension_values = group_key[1:]
            reports.append(
                _group_report(
                    values,
                    provenance=group_provenance,
                    key=dict(zip(fields, dimension_values, strict=True)),
                    outcome=outcome,
                )
            )
        result[name] = reports
    return result


def load_evaluation_dataset(
    db: Session,
    *,
    workspace_id: str,
    filters: EvaluationFilters | None = None,
) -> EvaluationDataset:
    """Load eligible execution observations using the evaluator's canonical rules.

    Usage is aggregated by workspace and execution before the prediction join.
    Profile computation consumes this same dataset so it cannot silently use
    telemetry that the evaluation engine would reject.
    """
    filters = filters or EvaluationFilters()
    if filters.start_at and filters.start_at.tzinfo is None:
        raise ValueError("start_at must include a timezone")
    if filters.end_at and filters.end_at.tzinfo is None:
        raise ValueError("end_at must include a timezone")
    if filters.start_at and filters.end_at and filters.start_at >= filters.end_at:
        raise ValueError("start_at must be earlier than end_at")

    usage = (
        select(
            UsageEvent.workspace_id.label("workspace_id"),
            UsageEvent.execution_id.label("execution_id"),
            func.count(UsageEvent.event_id).label("usage_event_count"),
            func.sum(UsageEvent.input_tokens).label("input_tokens"),
            (
                func.count(UsageEvent.event_id) - func.count(UsageEvent.input_tokens)
            ).label("input_missing_event_count"),
            func.sum(UsageEvent.output_tokens).label("output_tokens"),
            (
                func.count(UsageEvent.event_id) - func.count(UsageEvent.output_tokens)
            ).label("output_missing_event_count"),
            func.sum(UsageEvent.cache_creation_input_tokens).label("cache_creation_known_sum"),
            (
                func.count(UsageEvent.event_id)
                - func.count(UsageEvent.cache_creation_input_tokens)
            ).label("cache_creation_missing_event_count"),
            func.sum(UsageEvent.cache_read_input_tokens).label("cache_read_known_sum"),
            (
                func.count(UsageEvent.event_id)
                - func.count(UsageEvent.cache_read_input_tokens)
            ).label("cache_read_missing_event_count"),
            func.count(func.distinct(UsageEvent.provider)).label("usage_provider_count"),
            func.min(UsageEvent.provider).label("usage_provider"),
            func.count(func.distinct(UsageEvent.exact_provider_model)).label(
                "usage_model_count"
            ),
            func.min(UsageEvent.exact_provider_model).label("usage_exact_provider_model"),
        )
        .where(UsageEvent.workspace_id == workspace_id)
        .group_by(UsageEvent.workspace_id, UsageEvent.execution_id)
        .subquery("usage_by_execution")
    )
    prediction_links = (
        select(
            ExecutionRecord.workspace_id.label("workspace_id"),
            ExecutionRecord.prediction_id.label("prediction_id"),
            func.count(ExecutionRecord.execution_id).label("execution_count"),
        )
        .where(
            ExecutionRecord.workspace_id == workspace_id,
            ExecutionRecord.prediction_id.is_not(None),
        )
        .group_by(ExecutionRecord.workspace_id, ExecutionRecord.prediction_id)
        .subquery("executions_per_prediction")
    )

    query = (
        select(
            ExecutionRecord,
            PredictionRecord,
            usage.c.usage_event_count,
            usage.c.input_tokens,
            usage.c.input_missing_event_count,
            usage.c.output_tokens,
            usage.c.output_missing_event_count,
            usage.c.cache_creation_known_sum,
            usage.c.cache_creation_missing_event_count,
            usage.c.cache_read_known_sum,
            usage.c.cache_read_missing_event_count,
            usage.c.usage_provider_count,
            usage.c.usage_provider,
            usage.c.usage_model_count,
            usage.c.usage_exact_provider_model,
            prediction_links.c.execution_count,
        )
        .outerjoin(
            PredictionRecord,
            and_(
                PredictionRecord.prediction_id == ExecutionRecord.prediction_id,
                PredictionRecord.workspace_id == workspace_id,
            ),
        )
        .outerjoin(
            usage,
            and_(
                usage.c.workspace_id == ExecutionRecord.workspace_id,
                usage.c.execution_id == ExecutionRecord.execution_id,
            ),
        )
        .outerjoin(
            prediction_links,
            and_(
                prediction_links.c.workspace_id == ExecutionRecord.workspace_id,
                prediction_links.c.prediction_id == ExecutionRecord.prediction_id,
            ),
        )
        .where(ExecutionRecord.workspace_id == workspace_id)
    )
    if filters.start_at:
        query = query.where(ExecutionRecord.started_at >= filters.start_at)
    if filters.end_at:
        query = query.where(ExecutionRecord.started_at < filters.end_at)
    if filters.model_alias:
        query = query.where(PredictionRecord.model_alias == filters.model_alias)
    if filters.model_artifact_version:
        query = query.where(
            PredictionRecord.model_artifact_version == filters.model_artifact_version
        )
    if filters.provider:
        query = query.where(ExecutionRecord.provider == filters.provider)
    if filters.exact_provider_model:
        query = query.where(ExecutionRecord.exact_provider_model == filters.exact_provider_model)
    if filters.integration_name:
        query = query.where(ExecutionRecord.integration_name == filters.integration_name)
    if filters.integration_version:
        query = query.where(ExecutionRecord.integration_version == filters.integration_version)
    query = query.order_by(ExecutionRecord.started_at, ExecutionRecord.execution_id)

    rows = list(db.execute(query))
    total_executions = len(rows)
    completeness_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter({category: 0 for category in PROVENANCE_CATEGORIES})
    exclusion_counts: Counter[str] = Counter()
    missing_usage_event_execution_count = 0
    completed_records: list[_EligibleRecord] = []
    failed_records: list[_EligibleRecord] = []

    for row in rows:
        execution: ExecutionRecord = row[0]
        prediction: PredictionRecord | None = row[1]
        event_count = int(row.usage_event_count or 0)
        if event_count == 0:
            missing_usage_event_execution_count += 1
        provenance = _provenance(execution.telemetry_provenance)
        provenance_counts[provenance] += 1
        completeness = _completeness(execution.expected_usage_event_count, event_count)
        completeness_counts[completeness] += 1

        reason: str | None = None
        if execution.status == "started":
            reason = "execution_started"
        elif execution.status == "cancelled":
            reason = "execution_cancelled"
        elif execution.prediction_id is None or prediction is None:
            reason = "prediction_missing"
        elif int(row.execution_count or 0) != 1:
            reason = "prediction_has_multiple_executions"
        elif completeness == "unknown":
            reason = "telemetry_completeness_unknown"
        elif completeness == "incomplete":
            reason = "usage_event_count_mismatch"
        elif event_count == 0:
            reason = "missing_usage_events"
        elif row.usage_provider_count != 1 or row.usage_provider != execution.provider:
            reason = "usage_provider_mismatch"
        elif row.usage_model_count != 1:
            reason = "mixed_usage_models"
        elif row.usage_exact_provider_model != execution.exact_provider_model:
            reason = "usage_model_mismatch"
        elif execution.provider != "anthropic":
            reason = "unsupported_provider_token_semantics"
        elif prediction.model_alias != "claude-sonnet":
            reason = "unsupported_prediction_model"
        elif prediction.predicted_expected_tokens < 0:
            reason = "invalid_prediction"

        if reason:
            exclusion_counts[reason] += 1
            continue

        usage_summary = normalize_anthropic_aggregate(
            event_count=event_count,
            input_tokens=(
                None if int(row.input_missing_event_count or 0) else row.input_tokens
            ),
            output_tokens=(
                None if int(row.output_missing_event_count or 0) else row.output_tokens
            ),
            cache_creation_known_sum=row.cache_creation_known_sum,
            cache_creation_missing_event_count=int(row.cache_creation_missing_event_count or 0),
            cache_read_known_sum=row.cache_read_known_sum,
            cache_read_missing_event_count=int(row.cache_read_missing_event_count or 0),
        )
        if usage_summary.primary_total_tokens is None:
            exclusion_counts[usage_summary.unavailable_reason or "invalid_usage_counters"] += 1
            continue
        assert prediction is not None

        record = _EligibleRecord(
            execution_id=execution.execution_id,
            prediction_id=prediction.prediction_id,
            status=execution.status,
            started_at=execution.started_at,
            provenance=provenance,
            is_synthetic=execution.is_synthetic,
            task_hash=prediction.task_hash,
            model_alias=prediction.model_alias,
            model_artifact_version=prediction.model_artifact_version,
            provider=execution.provider,
            exact_provider_model=execution.exact_provider_model,
            integration_name=execution.integration_name,
            integration_version=execution.integration_version,
            metric_observation=MetricObservation(
                predicted_expected_tokens=prediction.predicted_expected_tokens,
                actual_tokens=usage_summary.primary_total_tokens,
                predicted_low_tokens=prediction.predicted_low_tokens,
                predicted_high_tokens=prediction.predicted_high_tokens,
            ),
            usage_summary=usage_summary,
        )
        if execution.status == "completed":
            completed_records.append(record)
        elif execution.status == "failed":
            failed_records.append(record)
        else:
            exclusion_counts["unsupported_execution_status"] += 1

    return EvaluationDataset(
        total_execution_count=total_executions,
        missing_usage_event_execution_count=missing_usage_event_execution_count,
        completeness_counts=dict(completeness_counts),
        provenance_counts=dict(provenance_counts),
        exclusion_counts=dict(exclusion_counts),
        completed_records=completed_records,
        failed_records=failed_records,
    )


def get_evaluation_summary(
    db: Session,
    *,
    workspace_id: str,
    filters: EvaluationFilters | None = None,
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only summary from execution-level SQL aggregates."""
    filters = filters or EvaluationFilters()
    evaluated_at = evaluated_at or datetime.now(UTC)
    dataset = load_evaluation_dataset(db, workspace_id=workspace_id, filters=filters)
    completed_records = dataset.completed_records
    failed_records = dataset.failed_records
    total_executions = dataset.total_execution_count
    completeness_counts = dataset.completeness_counts
    provenance_counts = dataset.provenance_counts
    exclusion_counts = dataset.exclusion_counts
    missing_usage_event_execution_count = dataset.missing_usage_event_execution_count

    provenance_reports = [
        _group_report(
            [record for record in completed_records if record.provenance == provenance],
            provenance=provenance,
            key={"provenance": provenance},
            outcome="completed",
        )
        for provenance in PROVENANCE_CATEGORIES
    ]
    failed_provenance_reports = [
        _group_report(
            [record for record in failed_records if record.provenance == provenance],
            provenance=provenance,
            key={"provenance": provenance},
            outcome="failed_consumption",
        )
        for provenance in PROVENANCE_CATEGORIES
    ]
    segments = _segments(completed_records, "completed")

    return {
        "evaluation_status": "available" if completed_records or failed_records else "unavailable",
        "methodology_version": METHODOLOGY_VERSION,
        "evaluation_unit": "one prediction linked to exactly one terminal execution",
        "primary_token_target": {
            "id": PRIMARY_TOKEN_TARGET,
            "description": (
                "Anthropic raw input_tokens (cache-exclusive by provider contract) "
                "plus output_tokens; cache counters remain separate."
            ),
        },
        "evaluated_at": evaluated_at,
        "evaluation_window": {
            "start_at": filters.start_at,
            "end_at": filters.end_at or evaluated_at,
            "basis": "execution.started_at; start inclusive, end exclusive",
        },
        "total_execution_count": total_executions,
        "eligible_completed_execution_count": len(completed_records),
        "eligible_failed_execution_count": len(failed_records),
        "excluded_execution_count": total_executions
        - len(completed_records)
        - len(failed_records),
        "missing_usage_event_execution_count": missing_usage_event_execution_count,
        "completeness_distribution": dict(completeness_counts),
        "provenance_distribution": dict(provenance_counts),
        "excluded_execution_counts": dict(exclusion_counts),
        "provenance_groups": provenance_reports,
        "failed_execution_consumption_groups": failed_provenance_reports,
        "segments": segments,
        "task_category": {
            "available": False,
            "reason": "task category is not stored on production prediction or execution records",
        },
        "filters": {
            "model_alias": filters.model_alias,
            "model_artifact_version": filters.model_artifact_version,
            "provider": filters.provider,
            "exact_provider_model": filters.exact_provider_model,
            "integration_name": filters.integration_name,
            "integration_version": filters.integration_version,
        },
        "production_accuracy_claim_eligible": any(
            group["production_accuracy_claim_eligible"] for group in provenance_reports
        ),
    }
