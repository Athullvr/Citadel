"""Authenticated, read-only API for prediction evaluation summaries."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from database import get_db
from evaluation_service import EvaluationFilters, get_evaluation_summary
from execution_feedback import configured_workspace_id, require_execution_auth
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/evaluations", tags=["prediction evaluation"])


class MetricValueResponse(BaseModel):
    value: float | None
    sample_count: int
    excluded_count: int
    exclusion_counts: dict[str, int]
    unavailable_reason: str | None


class EvaluationGroupResponse(BaseModel):
    provenance: str
    metric_class: str
    outcome: str
    key: dict[str, str | None]
    sample_count: int
    independent_task_count: int
    task_hash_missing_count: int
    reliability: str
    production_accuracy_claim_eligible: bool
    metrics: dict[str, MetricValueResponse]
    cache_usage: dict[str, int | None]


class EvaluationSummaryResponse(BaseModel):
    evaluation_status: str
    methodology_version: str
    evaluation_unit: str
    primary_token_target: dict[str, str]
    evaluated_at: datetime
    evaluation_window: dict[str, Any]
    total_execution_count: int
    eligible_completed_execution_count: int
    eligible_failed_execution_count: int
    excluded_execution_count: int
    missing_usage_event_execution_count: int
    completeness_distribution: dict[str, int]
    provenance_distribution: dict[str, int]
    excluded_execution_counts: dict[str, int]
    provenance_groups: list[EvaluationGroupResponse]
    failed_execution_consumption_groups: list[EvaluationGroupResponse]
    segments: dict[str, list[EvaluationGroupResponse]]
    task_category: dict[str, Any]
    filters: dict[str, str | None]
    production_accuracy_claim_eligible: bool


def _utc_datetime(value: datetime | None, field: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must include a timezone",
        )
    return value.astimezone(UTC)


@router.get(
    "/summary",
    response_model=EvaluationSummaryResponse,
    responses={
        401: {"description": "Authentication is required or invalid."},
        422: {"description": "Invalid evaluation filter or time window."},
        503: {"description": "Evaluation authentication or storage is unavailable."},
    },
    summary="Get a workspace-scoped prediction evaluation summary",
)
def evaluation_summary(
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
    from_at: Annotated[datetime | None, Query(alias="from")] = None,
    to_at: Annotated[datetime | None, Query(alias="to")] = None,
    model_alias: Annotated[str | None, Query(max_length=64)] = None,
    model_artifact_version: Annotated[str | None, Query(max_length=128)] = None,
    provider: Annotated[str | None, Query(max_length=64)] = None,
    exact_provider_model: Annotated[str | None, Query(max_length=128)] = None,
    integration_name: Annotated[str | None, Query(max_length=128)] = None,
    integration_version: Annotated[str | None, Query(max_length=64)] = None,
) -> dict[str, Any]:
    """Calculate metrics from immutable prediction and usage records."""
    start_at = _utc_datetime(from_at, "from")
    end_at = _utc_datetime(to_at, "to")
    if start_at and end_at and start_at >= end_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from must be earlier than to",
        )
    try:
        return get_evaluation_summary(
            db,
            workspace_id=configured_workspace_id(),
            filters=EvaluationFilters(
                start_at=start_at,
                end_at=end_at,
                model_alias=model_alias,
                model_artifact_version=model_artifact_version,
                provider=provider,
                exact_provider_model=exact_provider_model,
                integration_name=integration_name,
                integration_version=integration_version,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
