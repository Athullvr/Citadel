from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import adaptive_api
import pytest
from adaptive_api import AdaptivePredictionRequest
from adaptive_profiles import (
    _adjustment_bounds,
    distribution_free_median_confidence_interval,
    get_adaptive_profile,
)
from alembic import command
from alembic.config import Config
from database import create_database_engine
from execution_feedback import require_execution_auth
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent, utc_now
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from httpx import Response
from main import app
from pydantic import ValidationError
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
API_KEY = "adaptive-profile-test-secret"
WORKSPACE_ID = "adaptive-profile-test-workspace"
EXACT_MODEL = "claude-sonnet-5-20251001"


@pytest.fixture
def adaptive_client(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'adaptive-profile.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)
    monkeypatch.setenv("CITADEL_TASK_HASH_KEY", "stable-test-hmac-key")
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    engine = create_database_engine(database_url)
    yield engine
    engine.dispose()


def _insert_history(
    engine,
    *,
    count: int,
    residual: int,
    provider_model: str = EXACT_MODEL,
    workspace_id: str = WORKSPACE_ID,
    provenance: str = "verified_provider",
    artifact_version: str | None = None,
    age_days: int = 1,
    repeat_task: bool = False,
    synthetic_flag: bool | None = None,
    missing_task_hash: bool = False,
    expected_event_count: int = 1,
    status: str = "completed",
) -> None:
    artifact_version = artifact_version or adaptive_api._artifact_digest("claude-sonnet")
    assert artifact_version is not None
    started_at = utc_now() - timedelta(days=age_days)
    with Session(engine) as db:
        for index in range(count):
            prediction_id = uuid4()
            execution_id = uuid4()
            event_id = uuid4()
            task_hash = None
            if not missing_task_hash:
                task_hash = (
                    "a" * 64
                    if repeat_task
                    else hashlib.sha256(f"{uuid4()}:{index}".encode()).hexdigest()
                )
            actual = 100 + residual
            db.add(
                PredictionRecord(
                    prediction_id=prediction_id,
                    workspace_id=workspace_id,
                    created_at=started_at,
                    model_alias="claude-sonnet",
                    model_artifact_version=artifact_version,
                    feature_schema_version="citadel-features-v1",
                    predicted_low_tokens=80,
                    predicted_expected_tokens=100,
                    predicted_high_tokens=140,
                    estimated_low_cost=None,
                    estimated_expected_cost=None,
                    estimated_high_cost=None,
                    pricing_version=None,
                    prediction_payload_snapshot={"response": {"expected_tokens": 100}},
                    task_hash=task_hash,
                )
            )
            db.flush()
            db.add(
                ExecutionRecord(
                    execution_id=execution_id,
                    prediction_id=prediction_id,
                    workspace_id=workspace_id,
                    integration_name="adaptive-profile-test",
                    integration_version="1.0",
                    provider="anthropic",
                    exact_provider_model=provider_model,
                    status=status,
                    is_synthetic=(
                        provenance == "synthetic_benchmark"
                        if synthetic_flag is None
                        else synthetic_flag
                    ),
                    expected_usage_event_count=expected_event_count,
                    telemetry_provenance=provenance,
                    started_at=started_at,
                    completed_at=started_at + timedelta(minutes=1),
                )
            )
            db.flush()
            db.add(
                UsageEvent(
                    event_id=event_id,
                    execution_id=execution_id,
                    workspace_id=workspace_id,
                    provider="anthropic",
                    exact_provider_model=provider_model,
                    input_tokens=actual,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    source="caller_claimed_verified_provider",
                    occurred_at=started_at + timedelta(seconds=30),
                    received_at=started_at + timedelta(seconds=31),
                    idempotency_identifier=str(event_id),
                )
            )
        db.commit()


def _model_result(**overrides):
    result = {
        "model_id": "claude-sonnet",
        "low_tokens": 90,
        "expected_tokens": 100,
        "high_tokens": 130,
        "expected_cost_usd": 0.01,
        "low_cost_usd": 0.008,
        "high_cost_usd": 0.014,
        "cost_note": "test estimate",
        "driving_factors": ["task length"],
        "features": {"secret_feature_snapshot": "must not be returned"},
        "confidence": "normal",
        "out_of_distribution": False,
        "ood_reasons": [],
    }
    result.update(overrides)
    return result


def _adaptive_request(engine, **overrides):
    payload = {
        "task_text": "Research a bounded public topic.",
        "tools": ["web_search"],
        "model_id": "claude-sonnet",
        "provider": "anthropic",
        "exact_provider_model": EXACT_MODEL,
    }
    payload.update(overrides)
    request = AdaptivePredictionRequest.model_validate(payload)
    with Session(engine, expire_on_commit=False) as db:
        result = adaptive_api.create_adaptive_prediction(
            payload=request,
            _auth=None,
            db=db,
        )
    return Response(status_code=201, json=result.model_dump(mode="json"))


def test_adaptive_endpoint_uses_existing_bearer_auth(monkeypatch):
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    spec = app.openapi()["paths"]["/api/predictions/adaptive"]["post"]
    assert spec["security"]

    with pytest.raises(HTTPException) as exc_info:
        require_execution_auth(None)
    assert exc_info.value.status_code == 401
    require_execution_auth(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=API_KEY)
    )


def test_distribution_free_median_interval_and_insufficient_sample():
    assert distribution_free_median_confidence_interval([1.0] * 30) == (1.0, 1.0)
    assert distribution_free_median_confidence_interval([1.0] * 29) is not None
    assert distribution_free_median_confidence_interval([1.0]) is None


def test_adjustment_bounds_preserve_prediction_interval_and_resist_extreme_residual():
    adjustment, clipped, reason = _adjustment_bounds(
        10_000,
        low_tokens=90,
        expected_tokens=100,
        high_tokens=130,
    )
    assert adjustment == 40
    assert clipped is True
    assert reason is None

    adjustment, clipped, reason = _adjustment_bounds(
        -10_000,
        low_tokens=0,
        expected_tokens=100,
        high_tokens=130,
    )
    assert adjustment == 0
    assert clipped is True
    assert reason == "prediction_interval_does_not_allow_a_safe_adjustment"

    positive_half, _, _ = _adjustment_bounds(
        10.5,
        low_tokens=90,
        expected_tokens=100,
        high_tokens=180,
    )
    negative_half, _, _ = _adjustment_bounds(
        -10.5,
        low_tokens=90,
        expected_tokens=100,
        high_tokens=180,
    )
    assert positive_half == 11
    assert negative_half == -11


def test_adaptive_endpoint_uses_verified_independent_evidence_and_preserves_base(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(engine, count=29, residual=20)
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    insufficient = _adaptive_request(engine)
    assert insufficient.status_code == 201, insufficient.text
    assert insufficient.json()["profile"]["status"] == "insufficient_data"
    assert insufficient.json()["adjusted_prediction"]["expected_tokens"] == 100

    _insert_history(engine, count=1, residual=20)
    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["profile"]["status"] == "active"
    assert data["profile"]["profile_level"] == "provider_model"
    assert data["profile"]["evaluation_window"]["start_at"]
    assert data["profile"]["evaluation_window"]["end_at"]
    assert data["profile"]["profile_sample_count"] == 30
    assert data["profile"]["median_residual_tokens"] == 20
    assert data["original_prediction"]["expected_tokens"] == 100
    assert data["adjusted_prediction"]["expected_tokens"] == 120
    assert data["adjusted_prediction"]["low_tokens"] == 110
    assert "features" not in data["original_prediction"]
    assert "secret_feature_snapshot" not in response.text
    assert WORKSPACE_ID not in response.text
    assert '"task_hash":' not in response.text

    with Session(engine) as db:
        record = db.get(PredictionRecord, UUID(data["prediction_id"]))
        assert record is not None
        assert record.predicted_expected_tokens == 100
        assert record.prediction_payload_snapshot["response"]["expected_tokens"] == 100
        assert record.prediction_payload_snapshot["profile_adjustment"]["adjustment_tokens"] == 20
        assert record.prediction_payload_snapshot["adjusted_prediction"]["expected_tokens"] == 120


def test_unverified_synthetic_incomplete_and_repeated_tasks_do_not_activate_profiles(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=20, provenance="unverified_submitted")
    _insert_history(
        engine,
        count=30,
        residual=20,
        provenance="synthetic_benchmark",
        provider_model=EXACT_MODEL,
    )
    _insert_history(
        engine,
        count=30,
        residual=20,
        provenance="verified_provider",
        provider_model=EXACT_MODEL,
        expected_event_count=2,
    )
    _insert_history(
        engine,
        count=30,
        residual=20,
        provenance="verified_provider",
        provider_model=EXACT_MODEL,
        repeat_task=True,
    )
    _insert_history(
        engine,
        count=30,
        residual=20,
        provenance="verified_provider",
        provider_model=EXACT_MODEL,
        synthetic_flag=True,
    )
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert profile["status"] == "insufficient_data"
    assert profile["adjustment_tokens"] == 0
    assert profile["profiles"]["workspace"]["provenance_counts"]["unverified_submitted"] == 30
    assert profile["eligibility"]["evaluator_excluded_execution_counts"][
        "usage_event_count_mismatch"
    ] == 30
    assert profile["profiles"]["workspace"]["independent_task_count"] == 1
    assert profile["profiles"]["workspace"]["collapsed_repeat_count"] == 29
    assert profile["profiles"]["workspace"]["exclusion_counts"]["is_synthetic"] == 60


def test_workspace_isolation_and_requested_model_workspace_cohort(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(
        engine,
        count=30,
        residual=20,
        workspace_id="another-workspace",
    )
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())
    isolated = _adaptive_request(engine)
    assert isolated.status_code == 201, isolated.text
    assert isolated.json()["profile"]["status"] == "unavailable"
    assert isolated.json()["adjusted_prediction"]["low_tokens"] == 90
    assert isolated.json()["adjusted_prediction"]["expected_tokens"] == 100
    assert isolated.json()["adjusted_prediction"]["high_tokens"] == 130

    _insert_history(
        engine,
        count=30,
        residual=-20,
        provider_model="claude-sonnet-5-other",
    )
    fallback = _adaptive_request(engine)
    assert fallback.status_code == 201, fallback.text
    profile = fallback.json()["profile"]
    assert profile["status"] == "unavailable"
    assert profile["fallback_reason"] == "no_evaluator_eligible_history"
    assert profile["profiles"]["workspace"]["independent_task_count"] == 0
    assert fallback.json()["adjusted_prediction"]["low_tokens"] == 90
    assert fallback.json()["adjusted_prediction"]["expected_tokens"] == 100
    assert fallback.json()["adjusted_prediction"]["high_tokens"] == 130

    _insert_history(engine, count=30, residual=-20)
    requested_model = _adaptive_request(engine)
    assert requested_model.status_code == 201, requested_model.text
    profile = requested_model.json()["profile"]
    assert profile["status"] == "active"
    assert profile["profile_level"] == "provider_model"
    # With no cross-model pooling, workspace and provider/model cohorts have
    # identical eligible evidence; the exact provider/model level is selected.
    assert profile["profiles"]["workspace"]["status"] == "active"
    assert profile["profiles"]["workspace"]["independent_task_count"] == 30
    assert requested_model.json()["adjusted_prediction"]["expected_tokens"] == 80


def test_other_model_cannot_make_requested_model_sample_sufficient(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(engine, count=29, residual=20)
    _insert_history(
        engine,
        count=30,
        residual=-20,
        provider_model="claude-sonnet-5-other",
    )
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert profile["status"] == "insufficient_data"
    assert profile["fallback_reason"] == "insufficient_independent_tasks"
    assert profile["profiles"]["provider_model"]["independent_task_count"] == 29
    assert profile["profiles"]["workspace"]["independent_task_count"] == 29
    assert response.json()["adjusted_prediction"]["low_tokens"] == 90
    assert response.json()["adjusted_prediction"]["expected_tokens"] == 100
    assert response.json()["adjusted_prediction"]["high_tokens"] == 130


def test_missing_or_ambiguous_exact_model_cannot_use_other_model_history(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(
        engine,
        count=30,
        residual=-20,
        provider_model="claude-sonnet-5-other",
    )
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    with pytest.raises(ValidationError):
        AdaptivePredictionRequest.model_validate(
            {
                "task_text": "Review a bounded public topic.",
                "tools": ["web_search"],
                "model_id": "claude-sonnet",
                "provider": "anthropic",
            }
        )

    ambiguous = _adaptive_request(engine, exact_provider_model="claude-sonnet")
    assert ambiguous.status_code == 201, ambiguous.text
    profile = ambiguous.json()["profile"]
    assert profile["status"] == "incompatible"
    assert profile["fallback_reason"] == "exact_provider_model_missing_or_ambiguous"
    assert ambiguous.json()["adjusted_prediction"]["low_tokens"] == 90
    assert ambiguous.json()["adjusted_prediction"]["expected_tokens"] == 100
    assert ambiguous.json()["adjusted_prediction"]["high_tokens"] == 130


def test_exact_model_profile_uses_its_own_residuals(adaptive_client, monkeypatch):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=20)
    _insert_history(engine, count=30, residual=-20, provider_model="claude-sonnet-5-other")
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())
    exact = _adaptive_request(engine)
    assert exact.status_code == 201, exact.text
    assert exact.json()["profile"]["profile_level"] == "provider_model"
    assert exact.json()["profile"]["adjustment_tokens"] == 20


def test_stale_profile_falls_back(adaptive_client, monkeypatch):
    engine = adaptive_client
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())
    _insert_history(engine, count=30, residual=20, age_days=240)
    stale = _adaptive_request(engine)
    assert stale.status_code == 201, stale.text
    assert stale.json()["profile"]["status"] == "stale"
    assert stale.json()["profile"]["adjustment_tokens"] == 0
    assert stale.json()["profile"]["profiles"]["provider_model"][
        "independent_task_count"
    ] == 30


def test_incompatible_artifact_profile_falls_back(adaptive_client, monkeypatch):
    engine = adaptive_client
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())
    _insert_history(
        engine,
        count=30,
        residual=20,
        artifact_version="sha256:old-model-artifact",
    )
    incompatible = _adaptive_request(engine)
    assert incompatible.status_code == 201, incompatible.text
    assert incompatible.json()["profile"]["status"] == "incompatible"
    assert incompatible.json()["profile"]["fallback_reason"] == "prediction_artifact_mismatch"


def test_outlier_does_not_dominate_task_median(adaptive_client, monkeypatch):
    engine = adaptive_client
    _insert_history(engine, count=29, residual=20)
    _insert_history(engine, count=1, residual=100_000)
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert profile["status"] == "active"
    assert profile["median_residual_tokens"] == 20
    assert profile["adjustment_tokens"] == 20


def test_failed_executions_are_reported_but_not_used(adaptive_client, monkeypatch):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=20, status="failed")
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert profile["status"] == "unavailable"
    assert profile["eligibility"]["eligible_failed_execution_count"] == 30
    assert profile["eligibility"]["profile_excluded_execution_counts"][
        "failed_execution_not_used_for_prediction_accuracy"
    ] == 30


def test_missing_task_hashes_cannot_activate_a_profile(adaptive_client, monkeypatch):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=20, missing_task_hash=True)
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())

    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    profile = response.json()["profile"]
    assert profile["status"] == "insufficient_data"
    assert profile["profiles"]["workspace"]["exclusion_counts"]["task_hash_missing"] == 30
    assert profile["adjustment_tokens"] == 0


def test_concurrent_profile_queries_are_read_only_and_deterministic(adaptive_client):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=20)
    artifact_version = adaptive_api._artifact_digest("claude-sonnet")

    def query_profile(_index: int):
        with Session(engine) as db:
            return get_adaptive_profile(
                db,
                workspace_id=WORKSPACE_ID,
                provider="anthropic",
                exact_provider_model=EXACT_MODEL,
                model_alias="claude-sonnet",
                model_artifact_version=artifact_version,
                low_tokens=90,
                expected_tokens=100,
                high_tokens=130,
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(query_profile, range(8)))
    assert all(result["status"] == "active" for result in results)
    assert all(result["adjustment_tokens"] == 20 for result in results)


def test_negative_residual_and_unsupported_hierarchy_metadata_fall_back_safely(
    adaptive_client, monkeypatch
):
    engine = adaptive_client
    _insert_history(engine, count=30, residual=-15)
    monkeypatch.setattr(adaptive_api, "run_predict", lambda *_args, **_kwargs: _model_result())
    response = _adaptive_request(engine)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["profile"]["median_residual_tokens"] == -15
    assert data["adjusted_prediction"]["expected_tokens"] == 85
    assert data["profile"]["profiles"]["project"]["fallback_reason"] == (
        "project_identity_not_stored"
    )
    assert data["profile"]["profiles"]["task_category"]["fallback_reason"] == (
        "task_category_not_stored"
    )

    with pytest.raises(ValidationError):
        _adaptive_request(
            engine,
            tools=[],
            provider="other-provider",
            exact_provider_model="other-model-v1",
        )

    with Session(engine) as db:
        unsupported_profile = get_adaptive_profile(
            db,
            workspace_id=WORKSPACE_ID,
            provider="other-provider",
            exact_provider_model="other-model-v1",
            model_alias="claude-sonnet",
            model_artifact_version=adaptive_api._artifact_digest("claude-sonnet"),
            low_tokens=90,
            expected_tokens=100,
            high_tokens=130,
        )
    assert unsupported_profile["status"] == "incompatible"
    assert unsupported_profile["adjustment_tokens"] == 0

