"""PostgreSQL checks for profile eligibility and workspace isolation."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from adaptive_profiles import get_adaptive_profile
from execution_feedback import _artifact_digest
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent, utc_now
from prediction_identity import CLAUDE_TOKEN_TARGET_SEMANTICS, make_prediction_identity
from sqlalchemy.orm import Session

from postgres.conftest import PostgreSQLWorkers

pytestmark = pytest.mark.postgres

EXACT_MODEL = "test-anthropic-model-v1"


def _insert_verified_examples(
    engine,
    *,
    workspace_id: str,
    count: int,
    residual: int,
    task_prefix: str,
    exact_model: str = EXACT_MODEL,
) -> None:
    artifact_version = _artifact_digest("claude-sonnet")
    assert artifact_version is not None
    identity = make_prediction_identity(
        identity_status="resolved",
        provider="anthropic",
        exact_provider_model=exact_model,
        model_alias="claude-sonnet",
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=f"frozen-claude-sonnet-bundle@{artifact_version}",
        estimator_artifact_sha256=artifact_version.removeprefix("sha256:"),
        pricing_schedule_version=None,
    )
    started_at = utc_now() - timedelta(days=2)
    with Session(engine) as db:
        for index in range(count):
            prediction_id = uuid4()
            execution_id = uuid4()
            event_id = uuid4()
            db.add(
                PredictionRecord(
                    prediction_id=prediction_id,
                    workspace_id=workspace_id,
                    created_at=started_at,
                    model_alias="claude-sonnet",
                    provider=identity.provider,
                    exact_provider_model=identity.exact_provider_model,
                    identity_schema_version=identity.identity_schema_version,
                    identity_status=identity.identity_status,
                    token_target_semantics=identity.token_target_semantics,
                    estimator_artifact_version=identity.estimator_artifact_version,
                    estimator_artifact_sha256=identity.estimator_artifact_sha256,
                    prediction_identity_hash=identity.prediction_identity_hash,
                    model_artifact_version=identity.estimator_artifact_version,
                    feature_schema_version="citadel-features-v1",
                    predicted_low_tokens=80,
                    predicted_expected_tokens=100,
                    predicted_high_tokens=140,
                    prediction_payload_snapshot={
                        "response": {"expected_tokens": 100},
                        "prediction_identity": identity.public_dict(),
                    },
                    task_hash=hashlib.sha256(f"{task_prefix}:{index}".encode()).hexdigest(),
                )
            )
            db.flush()
            db.add(
                ExecutionRecord(
                    execution_id=execution_id,
                    prediction_id=prediction_id,
                    workspace_id=workspace_id,
                    integration_name="postgres-adaptive-profile-test",
                    integration_version="1.0",
                    provider="anthropic",
                    exact_provider_model=exact_model,
                    identity_schema_version=identity.identity_schema_version,
                    identity_status=identity.identity_status,
                    token_target_semantics=identity.token_target_semantics,
                    prediction_identity_hash=identity.prediction_identity_hash,
                    status="completed",
                    is_synthetic=False,
                    expected_usage_event_count=1,
                    telemetry_provenance="verified_provider",
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
                    exact_provider_model=exact_model,
                    identity_schema_version=identity.identity_schema_version,
                    identity_status=identity.identity_status,
                    token_target_semantics=identity.token_target_semantics,
                    prediction_identity_hash=identity.prediction_identity_hash,
                    input_tokens=100 + residual,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    source="direct-postgres-fixture",
                    occurred_at=started_at + timedelta(seconds=30),
                    received_at=started_at + timedelta(seconds=31),
                    idempotency_identifier=str(event_id),
                )
            )
        db.commit()


def test_postgres_profile_uses_eligible_history_and_isolates_workspaces(
    postgres_engine,
    postgres_workers: PostgreSQLWorkers,
):
    workspace_id = postgres_workers.workspace_id
    _insert_verified_examples(
        postgres_engine,
        workspace_id=workspace_id,
        count=30,
        residual=20,
        task_prefix="profile-workspace",
    )
    _insert_verified_examples(
        postgres_engine,
        workspace_id=workspace_id,
        count=30,
        residual=-80,
        task_prefix="profile-other-model",
        exact_model="test-other-anthropic-model-v1",
    )
    _insert_verified_examples(
        postgres_engine,
        workspace_id=f"{workspace_id}-other",
        count=30,
        residual=-80,
        task_prefix="other-workspace",
    )
    artifact_version = _artifact_digest("claude-sonnet")
    assert artifact_version is not None
    query_identity = make_prediction_identity(
        identity_status="resolved",
        provider="anthropic",
        exact_provider_model=EXACT_MODEL,
        model_alias="claude-sonnet",
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=f"frozen-claude-sonnet-bundle@{artifact_version}",
        estimator_artifact_sha256=artifact_version.removeprefix("sha256:"),
        pricing_schedule_version=None,
    )

    def query_profile(_index: int):
        with Session(postgres_engine) as db:
            return get_adaptive_profile(
                db,
                workspace_id=workspace_id,
                identity=query_identity,
                low_tokens=90,
                expected_tokens=100,
                high_tokens=130,
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        profiles = list(executor.map(query_profile, range(8)))
    assert all(profile["status"] == "active" for profile in profiles)
    assert all(profile["profile_sample_count"] == 30 for profile in profiles)
    assert all(profile["median_residual_tokens"] == 20 for profile in profiles)

    response = postgres_workers.request(
        0,
        "POST",
        "/api/predictions/adaptive",
        payload={
            "task_text": "Inspect a small code change and summarize the risk.",
            "tools": ["read_file"],
            "model_id": "claude-sonnet",
            "provider": "anthropic",
            "exact_provider_model": EXACT_MODEL,
        },
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["profile"]["status"] == "incompatible"
    assert result["profile"]["fallback_reason"] == "prediction_identity_unresolved"
    assert result["original_prediction"]["identity"]["exact_provider_model"] is None
    assert result["adjusted_prediction"]["expected_tokens"] == result[
        "original_prediction"
    ]["expected_tokens"]
    assert workspace_id not in response.text
    assert '"task_hash":' not in response.text

