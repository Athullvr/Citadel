"""Authenticated adaptive prediction endpoint backed by the profile service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from adaptive_profiles import adjusted_token_interval, get_adaptive_profile
from api_schemas import PredictResponse
from database import get_db
from execution_feedback import (
    CreatePredictionRequest,
    StoredPredictionResponse,
    _artifact_digest,
    _http_responses,
    configured_workspace_id,
    persist_prediction_snapshot,
    require_execution_auth,
)
from fastapi import APIRouter, Depends, HTTPException, status
from predict import SUPPORTED_MODELS
from predict import predict as run_predict
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/predictions", tags=["adaptive prediction"])


class AdaptivePredictionRequest(CreatePredictionRequest):
    provider: str = Field(min_length=1, max_length=64)
    exact_provider_model: str = Field(min_length=1, max_length=128)

    @field_validator("provider", "exact_provider_model")
    @classmethod
    def require_nonblank_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider and exact_provider_model must not be blank")
        return normalized


class PublicPredictionResponse(BaseModel):
    """Prediction fields safe for profile-aware output; omits feature snapshots."""

    model_id: str
    low_tokens: int
    expected_tokens: int
    high_tokens: int
    expected_cost_usd: float | None
    low_cost_usd: float | None
    high_cost_usd: float | None
    cost_note: str | None
    driving_factors: list[str]
    confidence: str
    out_of_distribution: bool
    ood_reasons: list[str]
    prediction_id: UUID
    created_at: datetime


class AdjustedTokenEstimate(BaseModel):
    low_tokens: int
    expected_tokens: int
    high_tokens: int
    adjustment_tokens: int
    costs_adjusted: bool = False
    cost_note: str


class AdaptiveProfileResponse(BaseModel):
    status: Literal["unavailable", "insufficient_data", "active", "stale", "incompatible"]
    profile_level: str
    profile_sample_count: int
    profile_execution_sample_count: int
    adjustment_tokens: int
    median_residual_tokens: float | None = None
    median_residual_confidence_interval_tokens: dict[str, Any] | None = None
    adjustment_clipped: bool = False
    fallback_reason: str | None = None
    methodology_version: str
    evaluation_window: dict[str, Any] | None = None
    eligibility: dict[str, Any]
    profiles: dict[str, Any]


class AdaptivePredictionResponse(BaseModel):
    prediction_id: UUID
    created_at: datetime
    original_prediction: PublicPredictionResponse
    adjusted_prediction: AdjustedTokenEstimate
    profile: AdaptiveProfileResponse


@router.post(
    "/adaptive",
    response_model=AdaptivePredictionResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_http_responses(400, 401, 413, 422, 500, 503),
    summary="Create a prediction with a workspace-scoped adaptive profile",
)
def create_adaptive_prediction(
    payload: AdaptivePredictionRequest,
    _auth: None = Depends(require_execution_auth),
    db: Session = Depends(get_db),  # noqa: B008
) -> AdaptivePredictionResponse:
    """Persist the base estimator output and return a safe, derived adjustment."""
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
    original = PredictResponse.model_validate(result)
    artifact_version = _artifact_digest(payload.model_id)
    profile = get_adaptive_profile(
        db,
        workspace_id=workspace_id,
        provider=payload.provider,
        exact_provider_model=payload.exact_provider_model,
        model_alias=payload.model_id,
        model_artifact_version=artifact_version,
        low_tokens=original.low_tokens,
        expected_tokens=original.expected_tokens,
        high_tokens=original.high_tokens,
    )
    adjustment = int(profile["adjustment_tokens"])
    adjusted = adjusted_token_interval(
        original.low_tokens,
        original.expected_tokens,
        original.high_tokens,
        adjustment,
    )
    # The evaluator has opened a read transaction. Close it before the helper
    # starts the durable insert transaction for this immutable prediction.
    db.commit()
    stored: StoredPredictionResponse = persist_prediction_snapshot(
        payload,
        result,
        workspace_id=workspace_id,
        db=db,
        profile_adjustment=profile,
        adjusted_prediction=adjusted,
    )
    return AdaptivePredictionResponse(
        prediction_id=stored.prediction_id,
        created_at=stored.created_at,
        original_prediction=PublicPredictionResponse.model_validate(stored.model_dump()),
        adjusted_prediction=AdjustedTokenEstimate(
            low_tokens=adjusted["low_tokens"],
            expected_tokens=adjusted["expected_tokens"],
            high_tokens=adjusted["high_tokens"],
            adjustment_tokens=adjustment,
            cost_note=(
                "Dollar cost estimates remain the original model estimates because "
                "a total-token residual cannot be allocated between input and output."
            ),
        ),
        profile=AdaptiveProfileResponse.model_validate(profile),
    )
