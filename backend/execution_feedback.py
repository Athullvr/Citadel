"""Opt-in prediction and execution feedback API.

This router uses a server-configured workspace and a dedicated fail-closed
authentication dependency. It does not change the legacy prediction endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from api_schemas import PredictResponse
from database import get_db
from execution_models import (
    TELEMETRY_PROVENANCE_SYNTHETIC_BENCHMARK,
    TELEMETRY_PROVENANCE_UNVERIFIED_SUBMITTED,
    ExecutionRecord,
    PredictionRecord,
    UsageEvent,
    utc_now,
)
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from features import FEATURE_NAMES
from predict import DEFAULT_MODEL, SUPPORTED_MODELS, get_bundle_path
from predict import predict as run_predict
from prediction_identity import (
    CLAUDE_TOKEN_TARGET_SEMANTICS,
    IDENTITY_STATUS_LEGACY_UNRESOLVED,
    IDENTITY_STATUS_RESOLVED,
    PredictionIdentity,
    make_prediction_identity,
    prediction_identity_hash_from_record,
    prediction_identity_matches_snapshot,
)
from prediction_registry import (
    resolve_prediction_registration,
    validate_prediction_request,
)
from pricing import DEFAULT_INPUT_RATIO, get_pricing
from provider_registry import (
    normalize_provider_id,
    validate_provider_model_identity,
    validate_reasoning_effort,
)
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from usage_normalization import NormalizedUsage, normalize_usage, validate_counter_map

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
    provider: str | None = Field(default=None, max_length=64)
    exact_provider_model: str | None = Field(default=None, max_length=128)

    @field_validator("tools")
    @classmethod
    def validate_tool_names(cls, value: list[str]) -> list[str]:
        if any(len(tool) > 64 for tool in value):
            raise ValueError("Each tool name must be at most 64 characters")
        return value

    @field_validator("provider")
    @classmethod
    def normalize_prediction_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_provider_id(value)
        if not normalized:
            raise ValueError("provider must not be blank")
        return normalized

    @field_validator("exact_provider_model")
    @classmethod
    def normalize_prediction_exact_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("exact_provider_model must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_provider_compatibility(self) -> CreatePredictionRequest:
        if self.provider is not None or self.exact_provider_model is not None:
            validate_prediction_request(
                self.model_id,
                provider=self.provider,
                exact_provider_model=self.exact_provider_model,
            )
        return self


class PredictionIdentityResponse(BaseModel):
    identity_schema_version: str
    identity_status: Literal["resolved", "legacy_unresolved"]
    provider: str
    exact_provider_model: str | None
    model_alias: str
    token_target_semantics: str
    estimator_artifact_version: str
    estimator_artifact_sha256: str
    pricing_schedule_version: str | None
    prediction_identity_hash: str


class StoredPredictionResponse(PredictResponse):
    prediction_id: UUID
    created_at: datetime
    identity: PredictionIdentityResponse


class CreateExecutionRequest(StrictRequest):
    prediction_id: UUID | None = None
    integration_name: str = Field(min_length=1, max_length=128)
    integration_version: str | None = Field(default=None, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    exact_provider_model: str | None = Field(default=None, max_length=128)
    token_target_semantics: str | None = Field(default=None, max_length=128)
    external_run_id: str | None = Field(default=None, max_length=256)
    parent_execution_id: UUID | None = None
    is_synthetic: bool = False
    reasoning_effort: str | None = Field(default=None, max_length=16)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        normalized = normalize_provider_id(value)
        if not normalized:
            raise ValueError("provider must not be blank")
        return normalized

    @field_validator("exact_provider_model")
    @classmethod
    def require_exact_model_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("exact_provider_model must not be blank")
        return value

    @field_validator("token_target_semantics")
    @classmethod
    def normalize_token_target_semantics(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("token_target_semantics must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_provider_model_configuration(self) -> CreateExecutionRequest:
        if self.exact_provider_model is not None:
            validate_provider_model_identity(self.provider, self.exact_provider_model)
            validate_reasoning_effort(
                self.provider, self.exact_provider_model, self.reasoning_effort
            )
        elif self.reasoning_effort is not None:
            raise ValueError("reasoning_effort requires an exact provider model identifier.")
        return self


class CreateUsageEventRequest(StrictRequest):
    event_id: UUID
    provider: str = Field(min_length=1, max_length=64)
    exact_provider_model: str | None = Field(default=None, max_length=128)
    input_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    output_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    cache_creation_input_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    cache_read_input_tokens: int | None = Field(default=None, strict=True, ge=0, le=_MAX_BIGINT)
    provider_counters: dict[str, StrictInt | None] | None = None
    source: str = Field(min_length=1, max_length=128)
    occurred_at: datetime

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        normalized = normalize_provider_id(value)
        if not normalized:
            raise ValueError("provider must not be blank")
        return normalized

    @field_validator("exact_provider_model")
    @classmethod
    def require_exact_model_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("exact_provider_model must not be blank")
        return value

    @field_validator("provider_counters")
    @classmethod
    def validate_provider_counters(
        cls, value: dict[str, int | None] | None
    ) -> dict[str, int | None] | None:
        return validate_counter_map(value) if value is not None else None

    @model_validator(mode="after")
    def validate_counter_source(self) -> CreateUsageEventRequest:
        if self.exact_provider_model is not None:
            validate_provider_model_identity(self.provider, self.exact_provider_model)
        if self.provider_counters is not None:
            if not self.provider_counters:
                raise ValueError("provider_counters must contain at least one counter.")
            if any(
                getattr(self, field) is not None
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ):
                raise ValueError("Use provider_counters or legacy token fields, not both.")
        elif self.input_tokens is None or self.output_tokens is None:
            raise ValueError("Both input_tokens and output_tokens are required for legacy usage.")
        return self

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(UTC)


class CompleteExecutionRequest(StrictRequest):
    status: Literal["completed", "failed", "cancelled"]
    completed_at: datetime | None = None
    expected_usage_event_count: int | None = Field(
        default=None, strict=True, ge=0, le=_MAX_BIGINT
    )

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
    exact_provider_model: str | None
    identity_schema_version: str | None
    identity_status: str
    token_target_semantics: str | None
    prediction_identity_hash: str | None
    status: str
    external_run_id: str | None
    parent_execution_id: UUID | None
    is_synthetic: bool
    telemetry_provenance: str | None = None
    expected_usage_event_count: int | None
    reasoning_effort: str | None = None
    started_at: datetime
    completed_at: datetime | None


class PredictionSummary(BaseModel):
    prediction_id: UUID
    model_alias: str
    provider: str | None = None
    exact_provider_model: str | None = None
    identity_status: str
    identity_schema_version: str | None = None
    token_target_semantics: str | None = None
    estimator_artifact_version: str | None = None
    estimator_artifact_sha256: str | None = None
    prediction_identity_hash: str | None = None
    model_artifact_version: str | None = None
    predicted_low_tokens: int
    predicted_expected_tokens: int
    predicted_high_tokens: int
    estimated_low_cost: float | None
    estimated_expected_cost: float | None
    estimated_high_cost: float | None
    created_at: datetime


class UsageAggregate(BaseModel):
    input_tokens: int | None
    input_tokens_known_sum: int | None
    input_tokens_missing_event_count: int
    output_tokens: int | None
    output_tokens_known_sum: int | None
    output_tokens_missing_event_count: int
    cache_creation_input_tokens: int | None
    cache_creation_input_tokens_known_sum: int | None
    cache_creation_missing_event_count: int
    cache_read_input_tokens: int | None
    cache_read_input_tokens_known_sum: int | None
    cache_read_missing_event_count: int
    cached_input_tokens: int | None
    cached_input_tokens_known_sum: int | None
    cached_input_tokens_missing_event_count: int
    reasoning_tokens: int | None
    reasoning_tokens_known_sum: int | None
    reasoning_tokens_missing_event_count: int
    tool_use_prompt_tokens: int | None
    tool_use_prompt_tokens_known_sum: int | None
    tool_use_prompt_tokens_missing_event_count: int
    provider_reported_total_tokens: int | None
    provider_reported_total_tokens_known_sum: int | None
    provider_reported_total_tokens_missing_event_count: int
    incompatible_usage_event_count: int
    token_semantics: str
    provider_usage_completeness: dict[str, int]
    total_input_tokens: int | None
    total_tokens: int | None


class ExecutionDetailResponse(BaseModel):
    execution: ExecutionMetadata
    prediction: PredictionSummary | None
    usage: UsageAggregate
    usage_event_count: int
    usage_completeness: Literal["complete", "incomplete", "unknown"]


class UsageEventResponse(BaseModel):
    event_id: UUID
    recorded: bool
    token_semantics: str | None = None
    usage_completeness: Literal["complete", "incomplete", "unknown"] = "unknown"


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


def _prediction_identity(
    payload: CreatePredictionRequest,
) -> PredictionIdentity:
    registration = validate_prediction_request(
        payload.model_id,
        provider=payload.provider,
        exact_provider_model=payload.exact_provider_model,
    )
    artifact_version = _artifact_digest(payload.model_id)
    if artifact_version is None or not artifact_version.startswith("sha256:"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The estimator artifact checksum is unavailable.",
        )
    artifact_sha256 = artifact_version.removeprefix("sha256:")
    if len(artifact_sha256) != 64:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The estimator artifact checksum is invalid.",
        )
    pricing_version, _ = _prediction_assumptions(payload.model_id)
    identity_status = (
        IDENTITY_STATUS_RESOLVED
        if registration.exact_provider_model is not None
        else IDENTITY_STATUS_LEGACY_UNRESOLVED
    )
    if registration.provider != "anthropic":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No prediction token target is configured for this provider.",
        )
    if not registration.estimator_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The selected estimator has no immutable artifact identifier.",
        )
    return make_prediction_identity(
        identity_status=identity_status,
        provider=registration.provider,
        exact_provider_model=registration.exact_provider_model,
        model_alias=registration.model_id,
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=f"{registration.estimator_id}@{artifact_version}",
        estimator_artifact_sha256=artifact_sha256,
        pricing_schedule_version=pricing_version,
    )


def _http_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
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


def persist_prediction_snapshot(
    payload: CreatePredictionRequest,
    result: dict[str, Any],
    *,
    workspace_id: str,
    db: Session,
    identity: PredictionIdentity | None = None,
    profile_adjustment: dict[str, Any] | None = None,
    adjusted_prediction: dict[str, int] | None = None,
) -> StoredPredictionResponse:
    """Persist the unchanged model output and optional immutable profile metadata."""
    response = PredictResponse.model_validate(result)
    created_at = utc_now()
    prediction_id = uuid4()
    pricing_version, pricing_assumptions = _prediction_assumptions(payload.model_id)
    identity = identity or _prediction_identity(payload)
    artifact_version = identity.estimator_artifact_version
    snapshot: dict[str, Any] = {
        "response": response.model_dump(mode="json"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "model_artifact_version": artifact_version,
        "prediction_identity": identity.public_dict(),
        "pricing_assumptions": pricing_assumptions,
        "input_retention": "task_text_not_stored",
    }
    if profile_adjustment is not None:
        snapshot["profile_adjustment"] = profile_adjustment
        if adjusted_prediction is not None:
            snapshot["adjusted_prediction"] = adjusted_prediction
    record = PredictionRecord(
        prediction_id=prediction_id,
        workspace_id=workspace_id,
        created_at=created_at,
        model_alias=payload.model_id,
        provider=identity.provider,
        exact_provider_model=identity.exact_provider_model,
        identity_schema_version=identity.identity_schema_version,
        identity_status=identity.identity_status,
        token_target_semantics=identity.token_target_semantics,
        estimator_artifact_version=identity.estimator_artifact_version,
        estimator_artifact_sha256=identity.estimator_artifact_sha256,
        prediction_identity_hash=identity.prediction_identity_hash,
        model_artifact_version=artifact_version,
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
        **response.model_dump(),
        prediction_id=prediction_id,
        created_at=created_at,
        identity=PredictionIdentityResponse.model_validate(identity.public_dict()),
    )


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
        registration = resolve_prediction_registration(payload.model_id)
        detail = (
            f"Unsupported model_id '{payload.model_id}': no compatible calibrated estimator "
            "is registered; no cross-provider fallback is used."
            if registration is not None
            else (
                f"Unsupported model_id '{payload.model_id}'. "
                f"Supported models in v1: {list(SUPPORTED_MODELS)}"
            )
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
    try:
        identity = _prediction_identity(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = run_predict(payload.task_text, payload.tools, model_id=payload.model_id)
    return persist_prediction_snapshot(
        payload,
        result,
        workspace_id=workspace_id,
        db=db,
        identity=identity,
    )


@router.post(
    "",
    response_model=ExecutionMetadata,
    status_code=status.HTTP_201_CREATED,
    responses=_http_responses(401, 404, 409, 413, 422, 500, 503),
    summary="Start an execution",
)
def create_execution(
    payload: CreateExecutionRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> ExecutionMetadata:
    workspace_id = configured_workspace_id()
    with db.begin():
        prediction: PredictionRecord | None = None
        if payload.prediction_id is not None:
            prediction = db.scalar(
                select(PredictionRecord).where(
                    PredictionRecord.prediction_id == payload.prediction_id,
                    PredictionRecord.workspace_id == workspace_id,
                )
            )
            if prediction is None:
                raise HTTPException(status_code=404, detail="Prediction not found")
        identity_status = "unlinked"
        identity_schema_version = None
        token_target_semantics = None
        prediction_identity_hash = None
        if prediction is not None:
            if prediction.provider is not None and prediction.provider != payload.provider:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "execution_provider_mismatch: provider must match the "
                        "prediction identity."
                    ),
                )
            if (
                prediction.exact_provider_model is not None
                and prediction.exact_provider_model != payload.exact_provider_model
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "execution_model_mismatch: exact_provider_model must match the "
                        "prediction identity."
                    ),
                )
            if (
                payload.token_target_semantics is not None
                and prediction.token_target_semantics is not None
                and payload.token_target_semantics != prediction.token_target_semantics
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "execution_token_target_mismatch: token target semantics must "
                        "match the prediction identity."
                    ),
                )
            identity_status = prediction.identity_status
            identity_schema_version = prediction.identity_schema_version
            token_target_semantics = prediction.token_target_semantics
            prediction_identity_hash = prediction.prediction_identity_hash
            if identity_status == IDENTITY_STATUS_RESOLVED:
                if (
                    not prediction.identity_schema_version
                    or not prediction.token_target_semantics
                    or not prediction.estimator_artifact_sha256
                    or not prediction.prediction_identity_hash
                    or len(prediction.estimator_artifact_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in prediction.estimator_artifact_sha256
                    )
                    or prediction_identity_hash_from_record(prediction)
                    != prediction.prediction_identity_hash
                    or not prediction_identity_matches_snapshot(
                        prediction,
                        prediction.prediction_payload_snapshot.get("prediction_identity"),
                    )
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=(
                            "prediction_identity_invalid: stored identity failed "
                            "integrity checks."
                        ),
                    )
                if (
                    payload.exact_provider_model is None
                    or prediction.exact_provider_model is None
                ):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="prediction_identity_unresolved: exact model identity is required.",
                    )
            elif identity_status != IDENTITY_STATUS_LEGACY_UNRESOLVED:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="prediction_identity_unresolved: prediction identity is not usable.",
                )

        if payload.parent_execution_id is not None:
            parent = db.scalar(
                select(ExecutionRecord).where(
                    ExecutionRecord.execution_id == payload.parent_execution_id,
                    ExecutionRecord.workspace_id == workspace_id,
                )
            )
            if parent is None:
                raise HTTPException(status_code=404, detail="Parent execution not found")
        record = ExecutionRecord(
            execution_id=uuid4(),
            prediction_id=payload.prediction_id,
            workspace_id=workspace_id,
            integration_name=payload.integration_name,
            integration_version=payload.integration_version,
            provider=payload.provider,
            exact_provider_model=payload.exact_provider_model,
            identity_schema_version=identity_schema_version,
            identity_status=identity_status,
            token_target_semantics=token_target_semantics,
            prediction_identity_hash=prediction_identity_hash,
            status="started",
            external_run_id=payload.external_run_id,
            parent_execution_id=payload.parent_execution_id,
            is_synthetic=payload.is_synthetic,
            reasoning_effort=payload.reasoning_effort,
            telemetry_provenance=(
                TELEMETRY_PROVENANCE_SYNTHETIC_BENCHMARK
                if payload.is_synthetic
                else TELEMETRY_PROVENANCE_UNVERIFIED_SUBMITTED
            ),
            started_at=utc_now(),
        )
        db.add(record)
        db.flush()
    return ExecutionMetadata.model_validate(record)


def _event_matches(
    existing: UsageEvent,
    execution_id: UUID,
    payload: CreateUsageEventRequest,
    normalized: NormalizedUsage,
    execution: ExecutionRecord,
) -> bool:
    # Event IDs define logical event identity. An omitted client timestamp can
    # differ on a retry; preserve the first timestamp and compare usage payloads.
    return (
        existing.execution_id == execution_id
        and existing.identity_status == execution.identity_status
        and existing.identity_schema_version == execution.identity_schema_version
        and existing.token_target_semantics == execution.token_target_semantics
        and existing.prediction_identity_hash == execution.prediction_identity_hash
        and existing.provider == payload.provider
        and existing.exact_provider_model == payload.exact_provider_model
        and existing.input_tokens == normalized.input_tokens
        and existing.output_tokens == normalized.output_tokens
        and existing.cache_creation_input_tokens == normalized.cache_creation_input_tokens
        and existing.cache_read_input_tokens == normalized.cache_read_input_tokens
        and existing.cached_input_tokens == normalized.cached_input_tokens
        and existing.reasoning_tokens == normalized.reasoning_tokens
        and existing.tool_use_prompt_tokens == normalized.tool_use_prompt_tokens
        and existing.total_tokens == normalized.total_tokens
        and existing.token_semantics == normalized.token_semantics
        and existing.usage_completeness == normalized.usage_completeness
        and existing.provider_counters == normalized.provider_counters
        and existing.source == payload.source
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
    legacy_counters: dict[str, int | None] = {
        field: getattr(payload, field)
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
        if field in payload.model_fields_set
    }
    normalized = normalize_usage(
        provider=payload.provider,
        exact_provider_model=payload.exact_provider_model,
        provider_counters=payload.provider_counters,
        legacy_counters=legacy_counters if payload.provider_counters is None else None,
        legacy_fields_present=set(payload.model_fields_set),
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
            if execution.provider != payload.provider:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="usage_provider_mismatch: provider must match the execution.",
                )
            if execution.exact_provider_model != payload.exact_provider_model:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "usage_model_mismatch: exact_provider_model must match the execution; "
                        "usage cannot change execution identity."
                    ),
                )
            existing = db.get(UsageEvent, payload.event_id)
            if existing is not None:
                if existing.workspace_id == workspace_id and _event_matches(
                    existing, execution_id, payload, normalized, execution
                ):
                    return UsageEventResponse(
                        event_id=payload.event_id,
                        recorded=False,
                        token_semantics=normalized.token_semantics,
                        usage_completeness=normalized.usage_completeness,
                    )
                raise HTTPException(
                    status_code=409,
                    detail="event_id was already used for different usage data",
                )
            if execution.status != "started":
                raise HTTPException(
                    status_code=409,
                    detail="Usage cannot be added to a completed execution",
                )
            event_record = UsageEvent(
                event_id=payload.event_id,
                execution_id=execution_id,
                workspace_id=workspace_id,
                provider=payload.provider,
                exact_provider_model=payload.exact_provider_model,
                identity_schema_version=execution.identity_schema_version,
                identity_status=execution.identity_status,
                token_target_semantics=execution.token_target_semantics,
                prediction_identity_hash=execution.prediction_identity_hash,
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
                cache_creation_input_tokens=normalized.cache_creation_input_tokens,
                cache_read_input_tokens=normalized.cache_read_input_tokens,
                cached_input_tokens=normalized.cached_input_tokens,
                reasoning_tokens=normalized.reasoning_tokens,
                tool_use_prompt_tokens=normalized.tool_use_prompt_tokens,
                total_tokens=normalized.total_tokens,
                token_semantics=normalized.token_semantics,
                usage_completeness=normalized.usage_completeness,
                provider_counters=normalized.provider_counters,
                source=payload.source,
                occurred_at=payload.occurred_at,
                received_at=utc_now(),
                idempotency_identifier=str(payload.event_id),
            )
            db.add(event_record)
            db.flush()
        return UsageEventResponse(
            event_id=payload.event_id,
            recorded=True,
            token_semantics=normalized.token_semantics,
            usage_completeness=normalized.usage_completeness,
        )
    except IntegrityError as exc:
        # A concurrent retry may win the unique-key race after the pre-check.
        db.rollback()
        existing = db.get(UsageEvent, payload.event_id)
        execution = db.scalar(
            select(ExecutionRecord).where(
                ExecutionRecord.execution_id == execution_id,
                ExecutionRecord.workspace_id == workspace_id,
            )
        )
        if (
            existing is not None
            and existing.workspace_id == workspace_id
            and execution is not None
        ):
            if _event_matches(existing, execution_id, payload, normalized, execution):
                return UsageEventResponse(
                    event_id=payload.event_id,
                    recorded=False,
                    token_semantics=normalized.token_semantics,
                    usage_completeness=normalized.usage_completeness,
                )
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
    incompatible_count = sum(
        event.provider != execution.provider
        or event.exact_provider_model != execution.exact_provider_model
        or event.identity_status != execution.identity_status
        or event.identity_schema_version != execution.identity_schema_version
        or event.token_target_semantics != execution.token_target_semantics
        or event.prediction_identity_hash != execution.prediction_identity_hash
        for event in events
    )
    aggregate_events = [] if incompatible_count else events
    input_total, input_known, input_missing = _sum_optional_field(aggregate_events, "input_tokens")
    output_total, output_known, output_missing = _sum_optional_field(
        aggregate_events, "output_tokens"
    )
    create_total, create_known, create_missing = _sum_optional_field(
        aggregate_events, "cache_creation_input_tokens"
    )
    read_total, read_known, read_missing = _sum_optional_field(
        aggregate_events, "cache_read_input_tokens"
    )
    cached_total, cached_known, cached_missing = _sum_optional_field(
        aggregate_events, "cached_input_tokens"
    )
    reasoning_total, reasoning_known, reasoning_missing = _sum_optional_field(
        aggregate_events, "reasoning_tokens"
    )
    tool_total, tool_known, tool_missing = _sum_optional_field(
        aggregate_events, "tool_use_prompt_tokens"
    )
    provider_total, provider_total_known, provider_total_missing = _sum_optional_field(
        aggregate_events, "total_tokens"
    )
    semantics = {event.token_semantics for event in events}
    non_null_semantics = {value for value in semantics if value is not None}
    if incompatible_count:
        token_semantics = "mixed_or_unknown"
    elif len(non_null_semantics) > 1 or (None in semantics and non_null_semantics):
        token_semantics = "mixed_or_unknown"
    elif non_null_semantics:
        token_semantics = next(iter(non_null_semantics))
    else:
        token_semantics = "unknown_legacy"
    provider_completeness = dict(Counter(event.usage_completeness for event in events))

    total_input = None
    total_tokens = None
    if (
        aggregate_events
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
            input_tokens_known_sum=input_known,
            input_tokens_missing_event_count=(event_count if incompatible_count else input_missing),
            output_tokens=output_total,
            output_tokens_known_sum=output_known,
            output_tokens_missing_event_count=(
                event_count if incompatible_count else output_missing
            ),
            cache_creation_input_tokens=create_total,
            cache_creation_input_tokens_known_sum=create_known,
            cache_creation_missing_event_count=(
                event_count if incompatible_count else create_missing
            ),
            cache_read_input_tokens=read_total,
            cache_read_input_tokens_known_sum=read_known,
            cache_read_missing_event_count=(event_count if incompatible_count else read_missing),
            cached_input_tokens=cached_total,
            cached_input_tokens_known_sum=cached_known,
            cached_input_tokens_missing_event_count=(
                event_count if incompatible_count else cached_missing
            ),
            reasoning_tokens=reasoning_total,
            reasoning_tokens_known_sum=reasoning_known,
            reasoning_tokens_missing_event_count=(
                event_count if incompatible_count else reasoning_missing
            ),
            tool_use_prompt_tokens=tool_total,
            tool_use_prompt_tokens_known_sum=tool_known,
            tool_use_prompt_tokens_missing_event_count=(
                event_count if incompatible_count else tool_missing
            ),
            provider_reported_total_tokens=provider_total,
            provider_reported_total_tokens_known_sum=provider_total_known,
            provider_reported_total_tokens_missing_event_count=(
                event_count if incompatible_count else provider_total_missing
            ),
            incompatible_usage_event_count=incompatible_count,
            token_semantics=token_semantics,
            provider_usage_completeness=provider_completeness,
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
        provider=prediction.provider,
        exact_provider_model=prediction.exact_provider_model,
        identity_status=prediction.identity_status,
        identity_schema_version=prediction.identity_schema_version,
        token_target_semantics=prediction.token_target_semantics,
        estimator_artifact_version=prediction.estimator_artifact_version,
        estimator_artifact_sha256=prediction.estimator_artifact_sha256,
        prediction_identity_hash=prediction.prediction_identity_hash,
        model_artifact_version=prediction.model_artifact_version,
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
    if execution.expected_usage_event_count is None:
        completeness: Literal["complete", "incomplete", "unknown"] = "unknown"
    elif execution.expected_usage_event_count == event_count:
        completeness = "complete"
    else:
        completeness = "incomplete"
    return ExecutionDetailResponse(
        execution=ExecutionMetadata.model_validate(execution),
        prediction=_prediction_summary(db, execution),
        usage=usage,
        usage_event_count=event_count,
        usage_completeness=completeness,
    )


@router.post(
    "/{execution_id}/complete",
    response_model=ExecutionDetailResponse,
    responses=_http_responses(401, 404, 409, 413, 422, 500, 503),
    summary="Complete, fail, or cancel an execution",
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
        execution.expected_usage_event_count = payload.expected_usage_event_count
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
