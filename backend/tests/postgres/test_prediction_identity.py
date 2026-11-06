from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from execution_feedback import (
    CreateExecutionRequest,
    CreateUsageEventRequest,
    create_execution,
    record_usage_event,
)
from execution_models import PredictionRecord, UsageEvent
from fastapi import HTTPException
from prediction_identity import CLAUDE_TOKEN_TARGET_SEMANTICS, make_prediction_identity
from sqlalchemy.orm import Session

pytestmark = pytest.mark.postgres
TEST_WORKSPACE_ID = "postgres-identity-test"
TEST_MODEL = "postgres-test-anthropic-model-v1"
TEST_SHA256 = hashlib.sha256(b"postgres-identity-test-artifact").hexdigest()


def _record(db: Session) -> PredictionRecord:
    identity = make_prediction_identity(
        identity_status="resolved",
        provider="anthropic",
        exact_provider_model=TEST_MODEL,
        model_alias="claude-sonnet",
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=f"test-estimator@sha256:{TEST_SHA256}",
        estimator_artifact_sha256=TEST_SHA256,
        pricing_schedule_version=None,
    )
    prediction = PredictionRecord(
        prediction_id=uuid4(),
        workspace_id=TEST_WORKSPACE_ID,
        model_alias=identity.model_alias,
        provider=identity.provider,
        exact_provider_model=identity.exact_provider_model,
        identity_schema_version=identity.identity_schema_version,
        identity_status=identity.identity_status,
        token_target_semantics=identity.token_target_semantics,
        estimator_artifact_version=identity.estimator_artifact_version,
        estimator_artifact_sha256=identity.estimator_artifact_sha256,
        prediction_identity_hash=identity.prediction_identity_hash,
        model_artifact_version=identity.estimator_artifact_version,
        predicted_low_tokens=80,
        predicted_expected_tokens=100,
        predicted_high_tokens=130,
        prediction_payload_snapshot={
            "prediction_identity": identity.public_dict(),
            "response": {"expected_tokens": 100},
        },
    )
    db.add(prediction)
    db.flush()
    return prediction


def test_postgres_enforces_prediction_execution_usage_identity(postgres_engine, monkeypatch):
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", TEST_WORKSPACE_ID)
    monkeypatch.setenv("CITADEL_API_KEY", "postgres-test-key")
    with Session(postgres_engine, expire_on_commit=False) as db:
        prediction = _record(db)
        db.commit()
        mismatched_execution = CreateExecutionRequest(
            prediction_id=prediction.prediction_id,
            integration_name="postgres-identity-test",
            provider="anthropic",
            exact_provider_model="other-postgres-test-model-v1",
        )
        with pytest.raises(HTTPException) as execution_error:
            create_execution(payload=mismatched_execution, _auth=None, db=db)
        assert execution_error.value.status_code == 409
        assert "execution_model_mismatch" in str(execution_error.value.detail)

        execution = create_execution(
            payload=CreateExecutionRequest(
                prediction_id=prediction.prediction_id,
                integration_name="postgres-identity-test",
                provider="anthropic",
                exact_provider_model=TEST_MODEL,
            ),
            _auth=None,
            db=db,
        )
        event_id = uuid4()
        mismatched_usage = CreateUsageEventRequest(
            event_id=event_id,
            provider="anthropic",
            exact_provider_model="other-postgres-test-model-v1",
            input_tokens=10,
            output_tokens=5,
            source="unverified-postgres-test",
            occurred_at=datetime.now(UTC),
        )
        with pytest.raises(HTTPException) as usage_error:
            record_usage_event(
                execution_id=execution.execution_id,
                payload=mismatched_usage,
                _auth=None,
                db=db,
            )
        assert usage_error.value.status_code == 409
        assert "usage_model_mismatch" in str(usage_error.value.detail)

        valid_usage = mismatched_usage.model_copy(
            update={"event_id": uuid4(), "exact_provider_model": TEST_MODEL}
        )
        result = record_usage_event(
            execution_id=execution.execution_id,
            payload=valid_usage,
            _auth=None,
            db=db,
        )
        stored_event = db.get(UsageEvent, valid_usage.event_id)
        assert result.recorded is True
        assert stored_event is not None
        assert stored_event.identity_status == "resolved"
        assert stored_event.prediction_identity_hash == execution.prediction_identity_hash
