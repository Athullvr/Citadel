from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from database import create_database_engine
from evaluation_api import EvaluationSummaryResponse, evaluation_summary
from execution_feedback import (
    CompleteExecutionRequest,
    CreateExecutionRequest,
    CreateUsageEventRequest,
    complete_execution,
    create_execution,
    record_usage_event,
    require_execution_auth,
)
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent, utc_now
from fastapi import HTTPException
from main import app
from prediction_identity import CLAUDE_TOKEN_TARGET_SEMANTICS, make_prediction_identity
from pydantic import ValidationError
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
API_KEY = "evaluation-api-test-secret"
WORKSPACE_ID = "citadel-evaluation-test-workspace"


@pytest.fixture
def evaluation_database(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'evaluation.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    engine = create_database_engine(database_url)
    yield engine
    engine.dispose()


def insert_prediction(
    engine,
    *,
    task_hash: str | None = None,
    provider: str = "anthropic",
    provider_model: str = "test-anthropic-model-a-v1",
) -> PredictionRecord:
    artifact_sha256 = "f" * 64
    artifact_version = f"evaluation-test-estimator@sha256:{artifact_sha256}"
    identity = make_prediction_identity(
        identity_status="resolved",
        provider=provider,
        exact_provider_model=provider_model,
        model_alias="claude-sonnet",
        token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS,
        estimator_artifact_version=artifact_version,
        estimator_artifact_sha256=artifact_sha256,
        pricing_schedule_version=None,
    )
    prediction = PredictionRecord(
        prediction_id=uuid4(),
        workspace_id=WORKSPACE_ID,
        created_at=utc_now(),
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
        predicted_high_tokens=130,
        prediction_payload_snapshot={
            "response": {"expected_tokens": 100},
            "prediction_identity": identity.public_dict(),
        },
        task_hash=task_hash,
    )
    with Session(engine, expire_on_commit=False) as db:
        db.add(prediction)
        db.commit()
    return prediction


def insert_execution(
    engine,
    prediction_id: UUID | None,
    *,
    is_synthetic: bool = False,
    integration_name: str = "evaluation-test-agent",
    provider: str = "anthropic",
    provider_model: str = "test-anthropic-model-a-v1",
    workspace_id: str = WORKSPACE_ID,
) -> dict:
    if workspace_id != WORKSPACE_ID:
        execution = ExecutionRecord(
            execution_id=uuid4(),
            prediction_id=prediction_id,
            workspace_id=workspace_id,
            integration_name=integration_name,
            integration_version="1.2.3",
            provider=provider,
            exact_provider_model=provider_model,
            identity_schema_version="citadel.prediction-identity.v1"
            if prediction_id is not None
            else None,
            identity_status="resolved" if prediction_id is not None else "unlinked",
            token_target_semantics=CLAUDE_TOKEN_TARGET_SEMANTICS
            if prediction_id is not None
            else None,
            prediction_identity_hash=None,
            status="started",
            is_synthetic=is_synthetic,
            started_at=utc_now(),
        )
        with Session(engine) as db:
            db.add(execution)
            db.commit()
        return {"execution_id": execution.execution_id, "prediction_id": prediction_id}

    payload = CreateExecutionRequest(
        prediction_id=prediction_id,
        integration_name=integration_name,
        integration_version="1.2.3",
        provider=provider,
        exact_provider_model=provider_model,
        is_synthetic=is_synthetic,
    )
    with Session(engine) as db:
        result = create_execution(payload=payload, _auth=None, db=db)
    return result.model_dump()


def usage_request(
    *,
    event_id: UUID | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    provider: str = "anthropic",
    provider_model: str = "test-anthropic-model-a-v1",
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    source: str = "caller-supplied-source",
) -> CreateUsageEventRequest:
    return CreateUsageEventRequest(
        event_id=event_id or uuid4(),
        provider=provider,
        exact_provider_model=provider_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        source=source,
        occurred_at=datetime.now(UTC),
    )


def submit_usage(engine, execution_id: UUID, payload: CreateUsageEventRequest) -> dict:
    with Session(engine) as db:
        result = record_usage_event(
            execution_id=execution_id,
            payload=payload,
            _auth=None,
            db=db,
        )
    return result.model_dump(mode="json")


def finish_execution(
    engine,
    execution_id: UUID,
    *,
    status: str = "completed",
    expected_count: int | None = None,
):
    payload_data: dict[str, str | int] = {"status": status}
    if expected_count is not None:
        payload_data["expected_usage_event_count"] = expected_count
    payload = CompleteExecutionRequest.model_validate(payload_data)
    with Session(engine) as db:
        return complete_execution(
            execution_id=execution_id,
            payload=payload,
            _auth=None,
            db=db,
        )


def summary(engine) -> dict:
    with Session(engine) as db:
        raw = evaluation_summary(
            _auth=None,
            db=db,
            from_at=None,
            to_at=None,
            model_alias=None,
            model_artifact_version=None,
            provider=None,
            exact_provider_model=None,
            integration_name=None,
            integration_version=None,
        )
    return EvaluationSummaryResponse.model_validate(raw).model_dump(mode="json")


def group(report: dict, provenance: str, outcome: str = "completed") -> dict:
    return next(
        item
        for item in report["provenance_groups"]
        if item["provenance"] == provenance and item["outcome"] == outcome
    )


def test_summary_requires_existing_auth_and_is_read_only(evaluation_database):
    with pytest.raises(HTTPException) as error:
        require_execution_auth(credentials=None)
    assert error.value.status_code == 401

    parameters = app.openapi()["paths"]["/api/evaluations/summary"]["get"].get("parameters", [])
    assert "workspace_id" not in {parameter["name"] for parameter in parameters}
    assert set(app.openapi()["paths"]["/api/evaluations/summary"]) == {"get"}


def test_multiple_events_are_summed_once_and_provenance_is_unverified(evaluation_database):
    prediction = insert_prediction(evaluation_database, task_hash="a" * 64)
    execution = insert_execution(evaluation_database, prediction.prediction_id)
    event_id = uuid4()
    first = usage_request(
        event_id=event_id,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=4,
    )
    second = usage_request(
        input_tokens=20,
        output_tokens=7,
        cache_creation_input_tokens=2,
        cache_read_input_tokens=6,
    )
    assert submit_usage(evaluation_database, execution["execution_id"], first)["recorded"] is True
    retry = usage_request(
        event_id=event_id,
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=4,
    )
    assert submit_usage(evaluation_database, execution["execution_id"], retry)["recorded"] is False
    assert submit_usage(evaluation_database, execution["execution_id"], second)["recorded"] is True
    finish_execution(evaluation_database, execution["execution_id"], expected_count=2)

    report = summary(evaluation_database)
    unverified = group(report, "unverified_submitted")
    assert report["eligible_completed_execution_count"] == 1
    assert unverified["sample_count"] == 1
    assert unverified["metrics"]["mean_absolute_error_tokens"]["value"] == 58
    assert unverified["cache_usage"]["cache_exclusive_input_tokens"] == 30
    assert unverified["cache_usage"]["output_tokens"] == 12
    assert unverified["cache_usage"]["primary_total_tokens"] == 42
    assert unverified["cache_usage"]["cache_creation_input_tokens"] == 5
    assert unverified["cache_usage"]["cache_read_input_tokens"] == 10
    assert report["provenance_distribution"]["verified_provider"] == 0
    assert report["production_accuracy_claim_eligible"] is False
    serialized = str(report)
    assert "a" * 64 not in serialized
    assert "prediction_payload_snapshot" not in serialized


def test_completeness_is_complete_incomplete_or_unknown(evaluation_database):
    complete_prediction = insert_prediction(evaluation_database)
    complete_run = insert_execution(evaluation_database, complete_prediction.prediction_id)
    submit_usage(evaluation_database, complete_run["execution_id"], usage_request())
    complete_result = finish_execution(
        evaluation_database, complete_run["execution_id"], expected_count=1
    )
    assert complete_result.usage_completeness == "complete"

    unknown_prediction = insert_prediction(evaluation_database)
    unknown_run = insert_execution(evaluation_database, unknown_prediction.prediction_id)
    submit_usage(evaluation_database, unknown_run["execution_id"], usage_request())
    unknown_result = finish_execution(evaluation_database, unknown_run["execution_id"])
    assert unknown_result.usage_completeness == "unknown"

    partial_prediction = insert_prediction(evaluation_database)
    partial_run = insert_execution(evaluation_database, partial_prediction.prediction_id)
    submit_usage(evaluation_database, partial_run["execution_id"], usage_request())
    incomplete_result = finish_execution(
        evaluation_database, partial_run["execution_id"], expected_count=2
    )
    assert incomplete_result.usage_completeness == "incomplete"

    report = summary(evaluation_database)
    assert report["completeness_distribution"] == {
        "complete": 1,
        "unknown": 1,
        "incomplete": 1,
    }
    assert report["excluded_execution_counts"]["telemetry_completeness_unknown"] == 1
    assert report["excluded_execution_counts"]["usage_event_count_mismatch"] == 1
    assert report["eligible_completed_execution_count"] == 1


def test_negative_expected_count_is_rejected_and_old_completion_shape_is_valid(
    evaluation_database,
):
    with pytest.raises(ValidationError):
        CompleteExecutionRequest.model_validate(
            {"status": "completed", "expected_usage_event_count": -1}
        )

    prediction = insert_prediction(evaluation_database)
    execution = insert_execution(evaluation_database, prediction.prediction_id)
    detail = finish_execution(evaluation_database, execution["execution_id"])
    assert detail.usage_completeness == "unknown"
    with Session(evaluation_database) as db:
        stored = db.get(ExecutionRecord, execution["execution_id"])
    assert stored is not None
    assert stored.expected_usage_event_count is None
    assert summary(evaluation_database)["completeness_distribution"]["unknown"] == 1


def test_synthetic_and_failed_telemetry_are_separate(evaluation_database):
    synthetic_prediction = insert_prediction(evaluation_database)
    synthetic = insert_execution(
        evaluation_database, synthetic_prediction.prediction_id, is_synthetic=True
    )
    submit_usage(evaluation_database, synthetic["execution_id"], usage_request())
    finish_execution(evaluation_database, synthetic["execution_id"], expected_count=1)

    failed_prediction = insert_prediction(evaluation_database)
    failed = insert_execution(evaluation_database, failed_prediction.prediction_id)
    submit_usage(evaluation_database, failed["execution_id"], usage_request())
    finish_execution(
        evaluation_database,
        failed["execution_id"],
        status="failed",
        expected_count=1,
    )

    report = summary(evaluation_database)
    assert group(report, "synthetic_benchmark")["metric_class"] == "synthetic_benchmark"
    assert group(report, "synthetic_benchmark")["sample_count"] == 1
    failed_group = next(
        item
        for item in report["failed_execution_consumption_groups"]
        if item["provenance"] == "unverified_submitted"
    )
    assert failed_group["sample_count"] == 1
    assert report["eligible_completed_execution_count"] == 1
    assert report["eligible_failed_execution_count"] == 1


def test_cancelled_execution_is_excluded_from_completed_and_failed_metrics(evaluation_database):
    prediction = insert_prediction(evaluation_database)
    execution = insert_execution(evaluation_database, prediction.prediction_id)
    submit_usage(evaluation_database, execution["execution_id"], usage_request())
    detail = finish_execution(
        evaluation_database,
        execution["execution_id"],
        status="cancelled",
        expected_count=1,
    )
    assert detail.execution.status == "cancelled"
    assert detail.usage_completeness == "complete"
    with pytest.raises(HTTPException) as repeated:
        finish_execution(
            evaluation_database,
            execution["execution_id"],
            status="failed",
            expected_count=1,
        )
    assert repeated.value.status_code == 409
    with pytest.raises(HTTPException) as late_event:
        submit_usage(evaluation_database, execution["execution_id"], usage_request())
    assert late_event.value.status_code == 409

    report = summary(evaluation_database)
    assert report["eligible_completed_execution_count"] == 0
    assert report["eligible_failed_execution_count"] == 0
    assert report["excluded_execution_counts"]["execution_cancelled"] == 1


def test_started_and_empty_executions_are_not_scored_as_zero(evaluation_database):
    started_prediction = insert_prediction(evaluation_database)
    insert_execution(evaluation_database, started_prediction.prediction_id)

    empty_prediction = insert_prediction(evaluation_database)
    empty = insert_execution(evaluation_database, empty_prediction.prediction_id)
    finish_execution(evaluation_database, empty["execution_id"], expected_count=0)

    report = summary(evaluation_database)
    assert report["eligible_completed_execution_count"] == 0
    assert report["excluded_execution_count"] == 2
    assert report["missing_usage_event_execution_count"] == 2
    assert report["excluded_execution_counts"]["execution_started"] == 1
    assert report["excluded_execution_counts"]["missing_usage_events"] == 1
    assert group(report, "unverified_submitted")["metrics"][
        "mean_absolute_error_tokens"
    ]["value"] is None


def test_multiple_executions_for_one_prediction_are_ambiguous(evaluation_database):
    prediction = insert_prediction(evaluation_database)
    executions = [
        insert_execution(evaluation_database, prediction.prediction_id),
        insert_execution(evaluation_database, prediction.prediction_id),
    ]
    for execution in executions:
        submit_usage(evaluation_database, execution["execution_id"], usage_request())
        finish_execution(evaluation_database, execution["execution_id"], expected_count=1)

    report = summary(evaluation_database)
    assert report["eligible_completed_execution_count"] == 0
    assert report["excluded_execution_counts"]["prediction_has_multiple_executions"] == 2


def test_provider_and_exact_model_segments_do_not_mix_counter_definitions(
    evaluation_database,
):
    models = ("test-anthropic-model-a-v1", "test-anthropic-model-b-v1")
    for model in models:
        prediction = insert_prediction(evaluation_database, provider_model=model)
        execution = insert_execution(
            evaluation_database, prediction.prediction_id, provider_model=model
        )
        submit_usage(
            evaluation_database,
            execution["execution_id"],
            usage_request(provider_model=model),
        )
        finish_execution(evaluation_database, execution["execution_id"], expected_count=1)

    unsupported_prediction = insert_prediction(
        evaluation_database,
        provider="openai",
        provider_model="test-openai-model-v1",
    )
    unsupported = insert_execution(
        evaluation_database,
        unsupported_prediction.prediction_id,
        provider="openai",
        provider_model="test-openai-model-v1",
    )
    submit_usage(
        evaluation_database,
        unsupported["execution_id"],
        usage_request(provider="openai", provider_model="test-openai-model-v1"),
    )
    finish_execution(evaluation_database, unsupported["execution_id"], expected_count=1)

    report = summary(evaluation_database)
    model_groups = report["segments"]["provider_model"]
    assert report["eligible_completed_execution_count"] == 2
    assert {
        group["key"]["exact_provider_model"] for group in model_groups
    } == set(models)
    assert all(group["sample_count"] == 1 for group in model_groups)
    assert report["excluded_execution_counts"]["unsupported_provider_token_semantics"] == 1


def test_usage_model_mismatch_and_caller_source_label(evaluation_database):
    prediction = insert_prediction(evaluation_database)
    execution = insert_execution(evaluation_database, prediction.prediction_id)
    submit_usage(
        evaluation_database,
        execution["execution_id"],
        usage_request(provider_model="test-anthropic-model-a-v1"),
    )
    with pytest.raises(HTTPException) as model_mismatch:
        submit_usage(
            evaluation_database,
            execution["execution_id"],
            usage_request(provider_model="test-other-anthropic-model-v1"),
        )
    assert model_mismatch.value.status_code == 409
    finish_execution(evaluation_database, execution["execution_id"], expected_count=1)
    report = summary(evaluation_database)
    assert report["eligible_completed_execution_count"] == 1
    another_prediction = insert_prediction(evaluation_database)
    another = insert_execution(evaluation_database, another_prediction.prediction_id)
    with pytest.raises(HTTPException) as error:
        submit_usage(
            evaluation_database,
            another["execution_id"],
            usage_request(provider="openai"),
        )
    assert error.value.status_code == 409
    assert report["provenance_distribution"]["verified_provider"] == 0


def test_missing_anthropic_input_counter_is_excluded_instead_of_summed_as_zero(
    evaluation_database,
):
    prediction = insert_prediction(evaluation_database)
    execution = insert_execution(evaluation_database, prediction.prediction_id)
    partial = CreateUsageEventRequest(
        event_id=uuid4(),
        provider="anthropic",
        exact_provider_model="test-anthropic-model-a-v1",
        provider_counters={"output_tokens": 5},
        source="incomplete-counter-set",
        occurred_at=datetime.now(UTC),
    )
    submit_usage(evaluation_database, execution["execution_id"], partial)
    finish_execution(evaluation_database, execution["execution_id"], expected_count=1)

    report = summary(evaluation_database)
    assert report["eligible_completed_execution_count"] == 0
    assert report["excluded_execution_counts"][
        "missing_required_input_or_output_counter"
    ] == 1


def test_workspace_isolation_and_legacy_unknown_provenance(evaluation_database):
    prediction = insert_prediction(evaluation_database)
    execution = insert_execution(
        evaluation_database,
        prediction.prediction_id,
        integration_name="caller-named-verified-source",
    )
    submit_usage(
        evaluation_database,
        execution["execution_id"],
        usage_request(source="verified_provider"),
    )
    finish_execution(evaluation_database, execution["execution_id"], expected_count=1)

    old_prediction = insert_prediction(evaluation_database)
    old_execution = ExecutionRecord(
        execution_id=uuid4(),
        prediction_id=old_prediction.prediction_id,
        workspace_id=WORKSPACE_ID,
        integration_name="legacy-integration",
        integration_version="0.1.0",
        provider="anthropic",
        exact_provider_model="test-anthropic-model-a-v1",
        identity_schema_version=old_prediction.identity_schema_version,
        identity_status=old_prediction.identity_status,
        token_target_semantics=old_prediction.token_target_semantics,
        prediction_identity_hash=old_prediction.prediction_identity_hash,
        status="completed",
        expected_usage_event_count=1,
        telemetry_provenance=None,
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    with Session(evaluation_database) as db:
        db.add(old_execution)
        db.flush()
        event = usage_request()
        db.add(
            UsageEvent(
                event_id=event.event_id,
                execution_id=old_execution.execution_id,
                workspace_id=WORKSPACE_ID,
                provider=event.provider,
                exact_provider_model=event.exact_provider_model,
                identity_schema_version=old_prediction.identity_schema_version,
                identity_status=old_prediction.identity_status,
                token_target_semantics=old_prediction.token_target_semantics,
                prediction_identity_hash=old_prediction.prediction_identity_hash,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cache_creation_input_tokens=event.cache_creation_input_tokens,
                cache_read_input_tokens=event.cache_read_input_tokens,
                source=event.source,
                occurred_at=event.occurred_at,
                received_at=utc_now(),
                idempotency_identifier=str(event.event_id),
            )
        )
        db.commit()

    report = summary(evaluation_database)
    assert report["provenance_distribution"]["verified_provider"] == 0
    assert group(report, "unverified_submitted")["sample_count"] == 1
    assert group(report, "unknown")["sample_count"] == 1

    # A fully populated row in a second workspace must not enter any aggregate.
    other_prediction = PredictionRecord(
        prediction_id=uuid4(),
        workspace_id="other-evaluation-workspace",
        created_at=utc_now(),
        model_alias="claude-sonnet",
        predicted_low_tokens=10,
        predicted_expected_tokens=20,
        predicted_high_tokens=30,
        prediction_payload_snapshot={"response": {}},
        task_hash="b" * 64,
    )
    other_execution = ExecutionRecord(
        execution_id=uuid4(),
        prediction_id=other_prediction.prediction_id,
        workspace_id=other_prediction.workspace_id,
        integration_name="other-workspace-agent",
        integration_version="1.0.0",
        provider="anthropic",
        exact_provider_model="test-anthropic-model-a-v1",
        status="completed",
        expected_usage_event_count=1,
        telemetry_provenance="verified_provider",
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    other_event_id = uuid4()
    with Session(evaluation_database) as db:
        db.add(other_prediction)
        db.flush()
        db.add(other_execution)
        db.flush()
        db.add(
            UsageEvent(
                event_id=other_event_id,
                execution_id=other_execution.execution_id,
                workspace_id=other_execution.workspace_id,
                provider="anthropic",
                exact_provider_model=other_execution.exact_provider_model,
                identity_schema_version=other_prediction.identity_schema_version,
                identity_status=other_prediction.identity_status,
                token_target_semantics=other_prediction.token_target_semantics,
                prediction_identity_hash=other_prediction.prediction_identity_hash,
                input_tokens=900,
                output_tokens=900,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                source="direct-fixture",
                occurred_at=utc_now(),
                received_at=utc_now(),
                idempotency_identifier=str(other_event_id),
            )
        )
        db.commit()

    isolated_report = summary(evaluation_database)
    assert isolated_report["total_execution_count"] == 2
    assert isolated_report["provenance_distribution"]["verified_provider"] == 0
