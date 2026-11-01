"""Opt-in prediction and execution feedback API.

This router uses a server-configured workspace and a dedicated fail-closed
authentication dependency. It does not change the legacy prediction endpoint.
"""

import hashlib
import hmac
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from api_schemas import PredictResponse
from database import get_db
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent, utc_now
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from features import FEATURE_NAMES
from predict import DEFAULT_MODEL, SUPPORTED_MODELS, get_bundle_path
from predict import predict as run_predict
from pricing import DEFAULT_INPUT_RATIO, get_pricing
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/executions", tags=["execution feedback"])
_bearer = HTTPBearer(auto_error=False)
_MAX_BIGINT = 2**63 - 1
FEATURE_SCHEMA_VERSION = "citadel-features-v1"
MAX_EXECUTION_BODY_BYTES = 256 * 1024


def require_execution_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),  # noqa: B008
) -> None:
    """Require a configured server API key for every execution route."""
    expected_key = os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution feedback authentication is not configured.",
        )
    if not credentials or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <API_KEY>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(credentials.credentials.strip(), expected_key.strip()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def configured_workspace_id() -> str:
    workspace_id = os.environ.get("CITADEL_EXECUTION_WORKSPACE_ID", "").strip()
    if not workspace_id or len(workspace_id) > 128:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution feedback workspace identity is not configured correctly.",
        )
    return workspace_id


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreatePredictionRequest(StrictRequest):
    task_text: str = Field(min_length=1, max_length=50000)
    tools: list[str] = Field(default_factory=list, max_length=20)
    model_id: str = Field(default=DEFAULT_MODEL, max_length=64)

    @field_validator("tools")
    @classmethod
    def validate_tool_names(cls, value: list[str]) -> list[str]:
        if any(len(tool) > 64 for tool in value):
            raise ValueError("Each tool name must be at most 64 characters")
        return value


class StoredPredictionResponse(PredictResponse):
    prediction_id: UUID
    created_at: datetime


class CreateExecutionRequest(StrictRequest):
    prediction_id: UUID | None = None
    integration_name: str = Field(min_length=1, max_length=128)
    integration_version: str | None = Field(default=None, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    exact_provider_model: str = Field(min_length=1, max_length=128)
    external_run_id: str | None = Field(default=None, max_length=256)
    parent_execution_id: UUID | None = None
    is_synthetic: bool = False


class CreateUsageEventRequest(StrictRequest):
    event_id: UUID
    provider: str = Field(min_length=1, max_length=64)
    exact_provider_model: str = Field(min_length=1, max_length=128)
    input_tokens: int = Field(strict=True, ge=0, le=_MAX_BIGINT)
    output_tokens: int = Field(strict=True, ge=0, le=_MAX_BIGINT)
    cache_creation_input_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    cache_read_input_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    source: str = Field(min_length=1, max_length=128)
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)


class CompleteExecutionRequest(StrictRequest):
    status: Literal["completed", "failed"]
    completed_at: datetime | None = None

    @field_validator("completed_at")
    @classmethod
    def normalize_completed_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must include a timezone")
        return value.astimezone(UTC)


class ExecutionMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    execution_id: UUID
    prediction_id: UUID | None
    integration_name: str
    integration_version: str | None
    provider: str
    exact_provider_model: str
    status: str
    external_run_id: str | None
    parent_execution_id: UUID | None
    is_synthetic: bool
    started_at: datetime
    completed_at: datetime | None


class PredictionSummary(BaseModel):
    prediction_id: UUID
    model_alias: str
    predicted_low_tokens: int
    predicted_expected_tokens: int
    predicted_high_tokens: int
    estimated_low_cost: float | None
    estimated_expected_cost: float | None
    estimated_high_cost: float | None
    created_at: datetime


class UsageAggregate(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_creation_input_tokens_known_sum: int | None
    cache_creation_missing_event_count: int
    cache_read_input_tokens: int | None
    cache_read_input_tokens_known_sum: int | None
    cache_read_missing_event_count: int
    total_input_tokens: int | None
    total_tokens: int | None


class ExecutionDetailResponse(BaseModel):
    execution: ExecutionMetadata
    prediction: PredictionSummary | None
    usage: UsageAggregate
    usage_event_count: int


class UsageEventResponse(BaseModel):
    event_id: UUID
    recorded: bool


def _artifact_digest(model_alias: str) -> str | None:
    try:
        path = get_bundle_path(model_alias)
        digest = hashlib.sha256()
        with Path(path).open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except (OSError, FileNotFoundError):
        return None


def _task_hash(task_text: str) -> str | None:
    key = os.environ.get("CITADEL_TASK_HASH_KEY", "")
    if not key:
        return None
    return hmac.new(key.encode("utf-8"), task_text.encode("utf-8"), hashlib.sha256).hexdigest()


def _prediction_assumptions(model_alias: str) -> tuple[str | None, dict[str, Any]]:
    pricing = get_pricing(model_alias)
    assumptions: dict[str, Any] = {"input_ratio": DEFAULT_INPUT_RATIO}
    if pricing is None:
        return None, assumptions
    # pricing.py has a per-entry verification date but no formal registry
    # release identifier, so persist only the available date as the version.
    pricing_version = pricing.last_updated
    assumptions["pricing"] = {
        "provider": pricing.provider,
        "input_cost_per_1k": pricing.input_cost_per_1k,
        "output_cost_per_1k": pricing.output_cost_per_1k,
        "last_updated": pricing.last_updated,
        "currency": pricing.currency,
        "notes": pricing.notes,
    }
    assumptions["pricing_version"] = pricing_version
    return pricing_version, assumptions


def _http_responses(*codes: int) -> dict[int, dict[str, str]]:
    descriptions = {
        400: "Invalid model or request.",
        401: "Authentication is required or invalid.",
        404: "The requested record does not exist in the configured workspace.",
        409: "The request conflicts with an existing record or invalid state.",
        413: "Request body exceeds the execution API size limit.",
        422: "Request schema validation failed.",
        503: "Execution storage or authentication is not configured or available.",
        500: "An unexpected server error occurred.",
    }
    return {code: {"description": descriptions[code]} for code in codes}


@router.post(
    "/predictions",
    response_model=StoredPredictionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_http_responses(400, 401, 413, 422, 500, 503),
    summary="Create and persist a prediction snapshot",
)
def create_prediction_record(
    payload: CreatePredictionRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> StoredPredictionResponse:
    workspace_id = configured_workspace_id()
    if payload.model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported model_id '{payload.model_id}'. "
                f"Supported models in v1: {list(SUPPORTED_MODELS)}"
            ),
        )

    result = run_predict(payload.task_text, payload.tools, model_id=payload.model_id)
    response = PredictResponse.model_validate(result)
    created_at = utc_now()
    prediction_id = uuid4()
    pricing_version, pricing_assumptions = _prediction_assumptions(payload.model_id)
    snapshot = {
        "response": response.model_dump(mode="json"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "model_artifact_version": _artifact_digest(payload.model_id),
        "pricing_assumptions": pricing_assumptions,
        "input_retention": "task_text_not_stored",
    }
    record = PredictionRecord(
        prediction_id=prediction_id,
        workspace_id=workspace_id,
        created_at=created_at,
        model_alias=payload.model_id,
        model_artifact_version=snapshot["model_artifact_version"],
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        predicted_low_tokens=response.low_tokens,
        predicted_expected_tokens=response.expected_tokens,
        predicted_high_tokens=response.high_tokens,
        estimated_low_cost=response.low_cost_usd,
        estimated_expected_cost=response.expected_cost_usd,
        estimated_high_cost=response.high_cost_usd,
        pricing_version=pricing_version,
        prediction_payload_snapshot=snapshot,
        task_hash=_task_hash(payload.task_text),
    )
    with db.begin():
        db.add(record)
        db.flush()
    return StoredPredictionResponse(
        **response.model_dump(), prediction_id=prediction_id, created_at=created_at
    )


@router.post(
    "",
    response_model=ExecutionMetadata,
    status_code=status.HTTP_201_CREATED,
    responses=_http_responses(401, 404, 413, 422, 500, 503),
    summary="Start an execution",
)
def create_execution(
    payload: CreateExecutionRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> ExecutionMetadata:
    workspace_id = configured_workspace_id()
    record = ExecutionRecord(
        execution_id=uuid4(),
        prediction_id=payload.prediction_id,
        workspace_id=workspace_id,
        integration_name=payload.integration_name,
        integration_version=payload.integration_version,
        provider=payload.provider,
        exact_provider_model=payload.exact_provider_model,
        status="started",
        external_run_id=payload.external_run_id,
        parent_execution_id=payload.parent_execution_id,
        is_synthetic=payload.is_synthetic,
        started_at=utc_now(),
    )
    with db.begin():
        if payload.prediction_id is not None:
            prediction = db.scalar(
                select(PredictionRecord).where(
                    PredictionRecord.prediction_id == payload.prediction_id,
                    PredictionRecord.workspace_id == workspace_id,
                )
            )
            if prediction is None:
                raise HTTPException(status_code=404, detail="Prediction not found")
        if payload.parent_execution_id is not None:
            parent = db.scalar(
                select(ExecutionRecord).where(
                    ExecutionRecord.execution_id == payload.parent_execution_id,
                    ExecutionRecord.workspace_id == workspace_id,
                )
            )
            if parent is None:
                raise HTTPException(status_code=404, detail="Parent execution not found")
        db.add(record)
        db.flush()
    return ExecutionMetadata.model_validate(record)


def _event_matches(
    existing: UsageEvent, execution_id: UUID, payload: CreateUsageEventRequest
) -> bool:
    return (
        existing.execution_id == execution_id
        and existing.provider == payload.provider
        and existing.exact_provider_model == payload.exact_provider_model
        and existing.input_tokens == payload.input_tokens
        and existing.output_tokens == payload.output_tokens
        and existing.cache_creation_input_tokens == payload.cache_creation_input_tokens
        and existing.cache_read_input_tokens == payload.cache_read_input_tokens
        and existing.source == payload.source
        and existing.occurred_at == payload.occurred_at
    )


@router.post(
    "/{execution_id}/usage",
    response_model=UsageEventResponse,
    responses=_http_responses(401, 404, 409, 413, 422, 500, 503),
    summary="Append one provider usage event",
)
def record_usage_event(
    execution_id: UUID,
    payload: CreateUsageEventRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> UsageEventResponse:
    workspace_id = configured_workspace_id()
    event_record = UsageEvent(
        event_id=payload.event_id,
        execution_id=execution_id,
        workspace_id=workspace_id,
        provider=payload.provider,
        exact_provider_model=payload.exact_provider_model,
        input_tokens=payload.input_tokens,
        output_tokens=payload.output_tokens,
        cache_creation_input_tokens=payload.cache_creation_input_tokens,
        cache_read_input_tokens=payload.cache_read_input_tokens,
        source=payload.source,
        occurred_at=payload.occurred_at,
        received_at=utc_now(),
        idempotency_identifier=str(payload.event_id),
    )

    try:
        with db.begin():
            execution = db.scalar(
                select(ExecutionRecord).where(
                    ExecutionRecord.execution_id == execution_id,
                    ExecutionRecord.workspace_id == workspace_id,
                ).with_for_update()
            )
            if execution is None:
                raise HTTPException(status_code=404, detail="Execution not found")
            existing = db.get(UsageEvent, payload.event_id)
            if existing is not None:
                if existing.workspace_id == workspace_id and _event_matches(
                    existing, execution_id, payload
                ):
                    return UsageEventResponse(event_id=payload.event_id, recorded=False)
                raise HTTPException(
                    status_code=409,
                    detail="event_id was already used for different usage data",
                )
            if execution.status != "started":
                raise HTTPException(
                    status_code=409,
                    detail="Usage cannot be added to a completed execution",
                )
            if execution.provider != payload.provider:
                raise HTTPException(
                    status_code=409,
                    detail="Usage provider must match the execution record.",
                )

            db.add(event_record)
            db.flush()
        return UsageEventResponse(event_id=payload.event_id, recorded=True)
    except IntegrityError as exc:
        # A concurrent retry may win the unique-key race after the pre-check.
        db.rollback()
        existing = db.get(UsageEvent, payload.event_id)
        if existing is not None and existing.workspace_id == workspace_id:
            if _event_matches(existing, execution_id, payload):
                return UsageEventResponse(event_id=payload.event_id, recorded=False)
            raise HTTPException(
                status_code=409,
                detail="event_id was already used for different usage data",
            ) from exc
        raise


def _sum_optional_field(events: list[UsageEvent], field: str) -> tuple[int | None, int | None, int]:
    values = [getattr(event, field) for event in events]
    present = [value for value in values if value is not None]
    known_sum = sum(present) if present else None
    missing_count = len(values) - len(present)
    complete_sum = known_sum if events and missing_count == 0 else None
    return complete_sum, known_sum, missing_count


def _aggregate_usage(db: Session, execution: ExecutionRecord) -> tuple[UsageAggregate, int]:
    events = list(
        db.scalars(
            select(UsageEvent)
            .where(
                UsageEvent.execution_id == execution.execution_id,
                UsageEvent.workspace_id == execution.workspace_id,
            )
            .order_by(UsageEvent.occurred_at, UsageEvent.event_id)
        )
    )
    event_count = len(events)
    input_total = sum(event.input_tokens for event in events) if events else None
    output_total = sum(event.output_tokens for event in events) if events else None
    create_total, create_known, create_missing = _sum_optional_field(
        events, "cache_creation_input_tokens"
    )
    read_total, read_known, read_missing = _sum_optional_field(events, "cache_read_input_tokens")

    total_input = None
    total_tokens = None
    if (
        events
        and execution.provider == "anthropic"
        and input_total is not None
        and create_total is not None
        and read_total is not None
        and output_total is not None
    ):
        total_input = input_total + create_total + read_total
        total_tokens = total_input + output_total

    return (
        UsageAggregate(
            input_tokens=input_total,
            output_tokens=output_total,
            cache_creation_input_tokens=create_total,
            cache_creation_input_tokens_known_sum=create_known,
            cache_creation_missing_event_count=create_missing,
            cache_read_input_tokens=read_total,
            cache_read_input_tokens_known_sum=read_known,
            cache_read_missing_event_count=read_missing,
            total_input_tokens=total_input,
            total_tokens=total_tokens,
        ),
        event_count,
    )


def _prediction_summary(db: Session, execution: ExecutionRecord) -> PredictionSummary | None:
    if execution.prediction_id is None:
        return None
    prediction = db.scalar(
        select(PredictionRecord).where(
            PredictionRecord.prediction_id == execution.prediction_id,
            PredictionRecord.workspace_id == execution.workspace_id,
        )
    )
    if prediction is None:
        return None
    return PredictionSummary(
        prediction_id=prediction.prediction_id,
        model_alias=prediction.model_alias,
        predicted_low_tokens=prediction.predicted_low_tokens,
        predicted_expected_tokens=prediction.predicted_expected_tokens,
        predicted_high_tokens=prediction.predicted_high_tokens,
        estimated_low_cost=prediction.estimated_low_cost,
        estimated_expected_cost=prediction.estimated_expected_cost,
        estimated_high_cost=prediction.estimated_high_cost,
        created_at=prediction.created_at,
    )


def _execution_detail(db: Session, execution: ExecutionRecord) -> ExecutionDetailResponse:
    usage, event_count = _aggregate_usage(db, execution)
    return ExecutionDetailResponse(
        execution=ExecutionMetadata.model_validate(execution),
        prediction=_prediction_summary(db, execution),
        usage=usage,
        usage_event_count=event_count,
    )


@router.post(
    "/{execution_id}/complete",
    response_model=ExecutionDetailResponse,
    responses=_http_responses(401, 404, 409, 413, 422, 500, 503),
    summary="Complete or fail an execution",
)
def complete_execution(
    execution_id: UUID,
    payload: CompleteExecutionRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> ExecutionDetailResponse:
    workspace_id = configured_workspace_id()
    with db.begin():
        execution = db.scalar(
            select(ExecutionRecord).where(
                ExecutionRecord.execution_id == execution_id,
                ExecutionRecord.workspace_id == workspace_id,
            ).with_for_update()
        )
        if execution is None:
            raise HTTPException(status_code=404, detail="Execution not found")
        if execution.status != "started":
            raise HTTPException(
                status_code=409,
                detail="Execution has already reached a terminal state",
            )
        completed_at = payload.completed_at or utc_now()
        if completed_at < execution.started_at:
            raise HTTPException(
                status_code=422,
                detail="completed_at cannot be earlier than started_at",
            )
        execution.status = payload.status
        execution.completed_at = completed_at
        db.flush()
    return _execution_detail(db, execution)


@router.get(
    "/{execution_id}",
    response_model=ExecutionDetailResponse,
    responses=_http_responses(401, 404, 422, 500, 503),
    summary="Get an execution and aggregated usage",
)
def get_execution(
    execution_id: UUID,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> ExecutionDetailResponse:
    workspace_id = configured_workspace_id()
    execution = db.scalar(
        select(ExecutionRecord).where(
            ExecutionRecord.execution_id == execution_id,
            ExecutionRecord.workspace_id == workspace_id,
        )
    )
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _execution_detail(db, execution)
