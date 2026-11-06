from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import execution_feedback
import main
import pytest
from alembic import command
from alembic.config import Config
from database import create_database_engine
from evaluation_service import EvaluationFilters, load_evaluation_dataset
from execution_feedback import (
    CreateExecutionRequest,
    CreatePredictionRequest,
    CreateUsageEventRequest,
    create_execution,
    record_usage_event,
)
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent, utc_now
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from prediction_identity import (
    CLAUDE_TOKEN_TARGET_SEMANTICS,
    IDENTITY_SCHEMA_VERSION,
    IDENTITY_STATUS_LEGACY_UNRESOLVED,
    IDENTITY_STATUS_RESOLVED,
    PredictionIdentity,
    make_prediction_identity,
)
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ID = "prediction-identity-workspace"
API_KEY = "prediction-identity-test-key"
TEST_EXACT_MODEL = "test-anthropic-model-v1"
TEST_ARTIFACT_SHA256 = hashlib.sha256(b"identity-test-artifact").hexdigest()
TEST_ARTIFACT_VERSION = f"test-estimator@sha256:{TEST_ARTIFACT_SHA256}"


@pytest.fixture
def identity_database(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'identity.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)
    monkeypatch.setenv("CITADEL_TASK_HASH_KEY", "identity-test-hmac-key")
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    engine = create_database_engine(database_url)
    yield engine
    engine.dispose()


def _resolved_identity(
    *,
    provider: str = "anthropic",
    exact_model: str = TEST_EXACT_MODEL,
    artifact_version: str = TEST_ARTIFACT_VERSION,
    artifact_sha256: str = TEST_ARTIFACT_SHA256,
) -> PredictionIdentity:
    return make_prediction_identity(
        identity_status=IDENTITY_STATUS_RESOLVED,
        provider=provider,
        exact_provider_model=exact_model,
        model_alias="claude-sonnet",
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=artifact_version,
        estimator_artifact_sha256=artifact_sha256,
        pricing_schedule_version=None,
    )


def _insert_prediction(
    db: Session,
    *,
    identity: PredictionIdentity | None = None,
    workspace_id: str = WORKSPACE_ID,
    task_hash: str | None = None,
) -> PredictionRecord:
    identity = identity or _resolved_identity()
    record = PredictionRecord(
        prediction_id=uuid4(),
        workspace_id=workspace_id,
        created_at=utc_now(),
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
        feature_schema_version="citadel-features-v1",
        predicted_low_tokens=80,
        predicted_expected_tokens=100,
        predicted_high_tokens=130,
        pricing_version=identity.pricing_schedule_version,
        prediction_payload_snapshot={
            "response": {"expected_tokens": 100},
            "prediction_identity": identity.public_dict(),
        },
        task_hash=task_hash,
    )
    db.add(record)
    db.flush()
    return record


def _execution_request(
    prediction_id,
    *,
    provider: str = "anthropic",
    exact_model: str | None = TEST_EXACT_MODEL,
    token_target_semantics: str | None = None,
) -> CreateExecutionRequest:
    return CreateExecutionRequest(
        prediction_id=prediction_id,
        integration_name="identity-test-agent",
        integration_version="1.0.0",
        provider=provider,
        exact_provider_model=exact_model,
        token_target_semantics=token_target_semantics,
    )


def _usage_request(*, event_id=None, exact_model: str | None = TEST_EXACT_MODEL, occurred_at=None):
    return CreateUsageEventRequest(
        event_id=event_id or uuid4(),
        provider="anthropic",
        exact_provider_model=exact_model,
        provider_counters={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "thinking_tokens": 1,
        },
        source="identity-test-submission",
        occurred_at=occurred_at or datetime.now(UTC),
    )


def test_new_prediction_persists_legacy_identity_and_unchanged_claude_api(
    identity_database, monkeypatch
):
    task_text = "Prompt must never be copied into the prediction snapshot."
    payload = CreatePredictionRequest(task_text=task_text, tools=["web_search"])
    result = {
        "model_id": "claude-sonnet",
        "low_tokens": 80,
        "expected_tokens": 100,
        "high_tokens": 130,
        "expected_cost_usd": 0.01,
        "low_cost_usd": 0.008,
        "high_cost_usd": 0.013,
        "cost_note": "Test estimate",
        "driving_factors": ["task length"],
        "features": {},
        "confidence": "moderate",
        "out_of_distribution": False,
        "ood_reasons": [],
    }
    monkeypatch.setattr(execution_feedback, "run_predict", lambda *_args, **_kwargs: result)
    with Session(identity_database, expire_on_commit=False) as db:
        response = execution_feedback.create_prediction_record(
            payload=payload, _auth=None, db=db
        )
    identity = response.identity
    assert identity.identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED
    assert identity.identity_schema_version == IDENTITY_SCHEMA_VERSION
    assert identity.provider == "anthropic"
    assert identity.exact_provider_model is None
    assert identity.model_alias == "claude-sonnet"
    assert identity.token_target_semantics == CLAUDE_TOKEN_TARGET_SEMANTICS
    assert identity.estimator_artifact_version.startswith(
        "frozen-claude-sonnet-bundle@sha256:"
    )
    assert identity.estimator_artifact_sha256 != TEST_ARTIFACT_SHA256
    assert identity.pricing_schedule_version is not None

    with Session(identity_database) as db:
        stored = db.get(PredictionRecord, response.prediction_id)
        assert stored is not None
        assert stored.prediction_identity_hash == identity.prediction_identity_hash
        assert stored.prediction_payload_snapshot["prediction_identity"] == identity.model_dump()
        assert task_text not in str(stored.prediction_payload_snapshot)
        assert stored.task_hash is not None

    monkeypatch.setattr(main, "run_predict", lambda *_args, **_kwargs: result)

    async def request_legacy_endpoint():
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://citadel.test"
        ) as client:
            return await client.post(
                "/api/predict",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"task_text": "legacy request", "tools": ["web_search"]},
            )

    legacy_response = asyncio.run(request_legacy_endpoint())
    assert legacy_response.status_code == 200
    assert legacy_response.json() == result
    assert "identity" not in legacy_response.json()


def test_execution_registration_rejects_provider_model_and_target_mismatch(identity_database):
    with Session(identity_database, expire_on_commit=False) as db:
        prediction = _insert_prediction(db)
        db.commit()
        cases = (
            (_execution_request(prediction.prediction_id, provider="openai"), "provider_mismatch"),
            (
                _execution_request(prediction.prediction_id, exact_model="other-test-model-v1"),
                "model_mismatch",
            ),
            (
                _execution_request(
                    prediction.prediction_id,
                    token_target_semantics="anthropic.different_target.v1",
                ),
                "token_target_mismatch",
            ),
        )
    for payload, reason in cases:
        with Session(identity_database) as db, pytest.raises(HTTPException) as exc_info:
            create_execution(payload=payload, _auth=None, db=db)
        assert exc_info.value.status_code == 409
        assert reason in str(exc_info.value.detail)


def test_usage_identity_cannot_override_execution_and_retry_keeps_first_timestamp(
    identity_database
):
    with Session(identity_database, expire_on_commit=False) as db:
        prediction = _insert_prediction(db)
        db.commit()
        execution = create_execution(
            payload=_execution_request(prediction.prediction_id), _auth=None, db=db
        )
        execution_id = execution.execution_id

    mismatch = _usage_request(exact_model="different-test-model-v1")
    with Session(identity_database) as db, pytest.raises(HTTPException) as exc_info:
        record_usage_event(
            execution_id=execution_id, payload=mismatch, _auth=None, db=db
        )
    assert exc_info.value.status_code == 409
    assert "usage_model_mismatch" in str(exc_info.value.detail)

    first_timestamp = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    event_id = uuid4()
    first = _usage_request(event_id=event_id, occurred_at=first_timestamp)
    with Session(identity_database) as db:
        first_result = record_usage_event(
            execution_id=execution_id, payload=first, _auth=None, db=db
        )
    assert first_result.recorded is True

    retry = _usage_request(
        event_id=event_id,
        occurred_at=first_timestamp + timedelta(days=1),
    )
    with Session(identity_database) as db:
        retry_result = record_usage_event(
            execution_id=execution_id, payload=retry, _auth=None, db=db
        )
        event = db.get(UsageEvent, event_id)
        execution = db.get(ExecutionRecord, execution_id)
    assert retry_result.recorded is False
    assert event is not None and execution is not None
    assert event.occurred_at == first_timestamp
    assert event.prediction_identity_hash == execution.prediction_identity_hash
    assert event.identity_status == IDENTITY_STATUS_RESOLVED
    assert event.provider_counters == first.provider_counters
    assert event.provider_counters["thinking_tokens"] == 1


def test_legacy_unresolved_is_ineligible_and_workspace_is_server_scoped(
    identity_database, monkeypatch
):
    with Session(identity_database, expire_on_commit=False) as db:
        prediction = _insert_prediction(
            db,
            identity=make_prediction_identity(
                identity_status=IDENTITY_STATUS_LEGACY_UNRESOLVED,
                provider="anthropic",
                exact_provider_model=None,
                model_alias="claude-sonnet",
                token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
                estimator_artifact_version=TEST_ARTIFACT_VERSION,
                estimator_artifact_sha256=TEST_ARTIFACT_SHA256,
                pricing_schedule_version=None,
            ),
        )
        db.commit()

        # A caller cannot select another workspace to retrieve the prediction.
        monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", "different-workspace")
        with pytest.raises(HTTPException) as not_found:
            create_execution(
                payload=_execution_request(prediction.prediction_id),
                _auth=None,
                db=db,
            )
        assert not_found.value.status_code == 404
        monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)

        execution_id = uuid4()
        event_id = uuid4()
        started_at = utc_now() - timedelta(minutes=2)
        db.add(
            ExecutionRecord(
                execution_id=execution_id,
                prediction_id=prediction.prediction_id,
                workspace_id=WORKSPACE_ID,
                integration_name="legacy-identity-test",
                provider="anthropic",
                exact_provider_model=None,
                identity_schema_version=prediction.identity_schema_version,
                identity_status=prediction.identity_status,
                token_target_semantics=prediction.token_target_semantics,
                prediction_identity_hash=prediction.prediction_identity_hash,
                status="completed",
                expected_usage_event_count=1,
                telemetry_provenance="unverified_submitted",
                started_at=started_at,
                completed_at=started_at + timedelta(minutes=1),
            )
        )
        db.flush()
        db.add(
            UsageEvent(
                event_id=event_id,
                execution_id=execution_id,
                workspace_id=WORKSPACE_ID,
                provider="anthropic",
                exact_provider_model=None,
                identity_schema_version=prediction.identity_schema_version,
                identity_status=prediction.identity_status,
                token_target_semantics=prediction.token_target_semantics,
                prediction_identity_hash=prediction.prediction_identity_hash,
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                cached_input_tokens=0,
                total_tokens=15,
                token_semantics="anthropic.messages.cache_exclusive_input.v1",
                usage_completeness="complete",
                provider_counters={"input_tokens": 10, "output_tokens": 5},
                source="untrusted_test_fixture",
                occurred_at=started_at + timedelta(seconds=30),
                received_at=started_at + timedelta(seconds=31),
                idempotency_identifier=str(event_id),
            )
        )
        db.commit()

    with Session(identity_database) as db:
        dataset = load_evaluation_dataset(
            db,
            workspace_id=WORKSPACE_ID,
            filters=EvaluationFilters(),
        )
    assert dataset.completed_records == []
    assert dataset.exclusion_counts["prediction_identity_unresolved"] == 1


def test_evaluation_excludes_inconsistent_usage_identity(identity_database):
    with Session(identity_database) as db:
        prediction = _insert_prediction(db, task_hash="b" * 64)
        identity_hash = prediction.prediction_identity_hash
        execution_id = uuid4()
        started_at = utc_now() - timedelta(minutes=2)
        db.add(
            ExecutionRecord(
                execution_id=execution_id,
                prediction_id=prediction.prediction_id,
                workspace_id=WORKSPACE_ID,
                integration_name="identity-test-agent",
                integration_version="1.0.0",
                provider=prediction.provider,
                exact_provider_model=prediction.exact_provider_model,
                identity_schema_version=prediction.identity_schema_version,
                identity_status=prediction.identity_status,
                token_target_semantics=prediction.token_target_semantics,
                prediction_identity_hash=identity_hash,
                status="completed",
                expected_usage_event_count=1,
                telemetry_provenance="verified_provider",
                started_at=started_at,
                completed_at=started_at + timedelta(minutes=1),
            )
        )
        db.flush()
        db.add(
            UsageEvent(
                event_id=uuid4(),
                execution_id=execution_id,
                workspace_id=WORKSPACE_ID,
                provider=prediction.provider,
                exact_provider_model=prediction.exact_provider_model,
                identity_schema_version=prediction.identity_schema_version,
                identity_status=prediction.identity_status,
                token_target_semantics=prediction.token_target_semantics,
                prediction_identity_hash="0" * 64,
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                cached_input_tokens=0,
                total_tokens=15,
                token_semantics="anthropic.messages.cache_exclusive_input.v1",
                usage_completeness="complete",
                provider_counters={"input_tokens": 10, "output_tokens": 5},
                source="direct_test_fixture",
                occurred_at=started_at + timedelta(seconds=30),
                received_at=started_at + timedelta(seconds=31),
                idempotency_identifier=str(uuid4()),
            )
        )
        db.commit()

    with Session(identity_database) as db:
        dataset = load_evaluation_dataset(
            db,
            workspace_id=WORKSPACE_ID,
            filters=EvaluationFilters(),
        )
    assert dataset.completed_records == []
    assert dataset.exclusion_counts["usage_prediction_identity_mismatch"] == 1


def test_identity_migration_preserves_unknown_history_and_round_trips_sqlite(
    tmp_path, monkeypatch
):
    database_url = f"sqlite:///{(tmp_path / 'identity-migration.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    command.upgrade(config, "0004_provider_usage_metadata")
    engine = create_database_engine(database_url)
    prediction_id = uuid4()
    execution_id = uuid4()
    event_id = uuid4()
    historical_time = datetime(2026, 1, 2, 3, 4, tzinfo=UTC).isoformat()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO execution_predictions "
            "(prediction_id, workspace_id, created_at, model_alias, "
            "predicted_low_tokens, predicted_expected_tokens, predicted_high_tokens, "
            "prediction_payload_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prediction_id.hex,
                WORKSPACE_ID,
                historical_time,
                "claude-sonnet",
                80,
                100,
                130,
                '{"response":{"expected_tokens":100}}',
            ),
        )
        connection.exec_driver_sql(
            "INSERT INTO execution_records "
            "(execution_id, prediction_id, workspace_id, integration_name, provider, "
            "exact_provider_model, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                execution_id.hex,
                prediction_id.hex,
                WORKSPACE_ID,
                "historical-agent",
                "anthropic",
                "caller-reported-legacy-model",
                "completed",
                historical_time,
            ),
        )
        connection.exec_driver_sql(
            "INSERT INTO usage_events "
            "(event_id, execution_id, workspace_id, provider, exact_provider_model, "
            "input_tokens, output_tokens, source, occurred_at, received_at, "
            "idempotency_identifier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id.hex,
                execution_id.hex,
                WORKSPACE_ID,
                "anthropic",
                "caller-reported-legacy-model",
                10,
                5,
                "historical-unverified",
                historical_time,
                historical_time,
                str(event_id),
            ),
        )

    command.upgrade(config, "head")
    with Session(engine) as db:
        prediction = db.get(PredictionRecord, prediction_id)
        execution = db.get(ExecutionRecord, execution_id)
        event = db.get(UsageEvent, event_id)
    assert prediction is not None and execution is not None and event is not None
    assert prediction.identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED
    assert prediction.provider is None and prediction.exact_provider_model is None
    assert prediction.estimator_artifact_version is None
    assert prediction.estimator_artifact_sha256 is None
    assert prediction.prediction_identity_hash is None
    assert execution.identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED
    assert execution.exact_provider_model == "caller-reported-legacy-model"
    assert execution.prediction_identity_hash is None
    assert event.identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED
    assert event.exact_provider_model == "caller-reported-legacy-model"
    assert event.prediction_identity_hash is None

    command.downgrade(config, "0004_provider_usage_metadata")
    command.upgrade(config, "head")
    with Session(engine) as db:
        assert db.get(PredictionRecord, prediction_id).identity_status == (
            IDENTITY_STATUS_LEGACY_UNRESOLVED
        )
        assert db.get(ExecutionRecord, execution_id).identity_status == (
            IDENTITY_STATUS_LEGACY_UNRESOLVED
        )
        assert db.get(UsageEvent, event_id).identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED
    engine.dispose()


def test_identity_migration_refuses_to_drop_new_identity(identity_database):
    with Session(identity_database) as db:
        identity = _resolved_identity()
        prediction = _insert_prediction(db, identity=identity)
        db.commit()
    with pytest.raises(RuntimeError, match="Cannot downgrade prediction identity"):
        command.downgrade(Config(str(BACKEND_DIR / "alembic.ini")), "0004_provider_usage_metadata")
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    with Session(identity_database) as db:
        assert db.get(PredictionRecord, prediction.prediction_id).prediction_identity_hash == (
            identity.prediction_identity_hash
        )
