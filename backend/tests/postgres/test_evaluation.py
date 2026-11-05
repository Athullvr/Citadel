"""PostgreSQL-backed evaluation query and execution-linkage checks."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent
from sqlalchemy.orm import Session

from postgres.conftest import PostgreSQLWorkers
from postgres.helpers import create_prediction, usage_payload

pytestmark = pytest.mark.postgres


def _create_execution(
    workers: PostgreSQLWorkers,
    prediction_id: str,
    *,
    integration_name: str,
    synthetic: bool = False,
) -> dict:
    response = workers.request(
        0,
        "POST",
        "/api/executions",
        payload={
            "prediction_id": prediction_id,
            "integration_name": integration_name,
            "integration_version": "pg-evaluation-test",
            "provider": "anthropic",
            "exact_provider_model": "claude-sonnet-5-20251001",
            "is_synthetic": synthetic,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _record_usage(
    workers: PostgreSQLWorkers,
    execution_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    event_id=None,
):
    payload = usage_payload(
        event_id or uuid4(), input_tokens=input_tokens, output_tokens=output_tokens
    )
    response = workers.request(
        1,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=payload,
    )
    assert response.status_code == 200, response.text
    return payload, response.json()


def _complete(workers: PostgreSQLWorkers, execution_id: str, expected_count: int):
    response = workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/complete",
        payload={"status": "completed", "expected_usage_event_count": expected_count},
    )
    assert response.status_code == 200, response.text
    assert response.json()["usage_completeness"] == "complete"


def test_postgres_evaluation_sums_events_once_and_preserves_synthetic_cohort(
    postgres_workers: PostgreSQLWorkers,
):
    prediction = create_prediction(postgres_workers)
    execution = _create_execution(
        postgres_workers,
        prediction["prediction_id"],
        integration_name=f"pg-evaluation-sum-{uuid4().hex}",
        synthetic=True,
    )
    execution_id = execution["execution_id"]
    event_id = uuid4()
    first_payload, first = _record_usage(
        postgres_workers,
        execution_id,
        input_tokens=120,
        output_tokens=30,
        event_id=event_id,
    )
    duplicate_response = postgres_workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=first_payload,
    )
    second_payload, second = _record_usage(
        postgres_workers,
        execution_id,
        input_tokens=80,
        output_tokens=20,
    )
    assert first["recorded"] is True
    assert duplicate_response.status_code == 200, duplicate_response.text
    assert duplicate_response.json()["recorded"] is False
    assert second["recorded"] is True
    _complete(postgres_workers, execution_id, expected_count=2)

    report_response = postgres_workers.request(
        0,
        "GET",
        f"/api/evaluations/summary?integration_name={execution['integration_name']}",
    )
    assert report_response.status_code == 200, report_response.text
    report = report_response.json()
    synthetic = next(
        item
        for item in report["provenance_groups"]
        if item["provenance"] == "synthetic_benchmark"
    )
    expected = prediction["expected_tokens"]
    assert report["total_execution_count"] == 1
    assert report["eligible_completed_execution_count"] == 1
    assert report["completeness_distribution"] == {"complete": 1}
    assert synthetic["sample_count"] == 1
    assert synthetic["metrics"]["mean_absolute_error_tokens"]["value"] == abs(
        250 - expected
    )
    assert report["provenance_distribution"]["verified_provider"] == 0
    assert second_payload["input_tokens"] == 80


def test_postgres_evaluation_is_workspace_scoped_and_excludes_prediction_reuse(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    integration_name = f"pg-evaluation-isolation-{uuid4().hex}"
    prediction = create_prediction(postgres_workers)
    executions = [
        _create_execution(
            postgres_workers,
            prediction["prediction_id"],
            integration_name=integration_name,
        )
        for _ in range(2)
    ]
    for execution in executions:
        _record_usage(
            postgres_workers,
            execution["execution_id"],
            input_tokens=10,
            output_tokens=5,
        )
        _complete(postgres_workers, execution["execution_id"], expected_count=1)

    # A fully populated record in another workspace must not leak through any join.
    other_prediction_id = uuid4()
    other_execution_id = uuid4()
    other_event_id = uuid4()
    now = datetime.now(UTC)
    with Session(postgres_engine) as db:
        db.add(
            PredictionRecord(
                prediction_id=other_prediction_id,
                workspace_id="separate-evaluation-workspace",
                created_at=now,
                model_alias="claude-sonnet",
                predicted_low_tokens=10,
                predicted_expected_tokens=20,
                predicted_high_tokens=30,
                prediction_payload_snapshot={"response": {}},
            )
        )
        db.flush()
        db.add(
            ExecutionRecord(
                execution_id=other_execution_id,
                prediction_id=other_prediction_id,
                workspace_id="separate-evaluation-workspace",
                integration_name=integration_name,
                integration_version="pg-evaluation-test",
                provider="anthropic",
                exact_provider_model="claude-sonnet-5-20251001",
                status="completed",
                expected_usage_event_count=1,
                telemetry_provenance="verified_provider",
                started_at=now,
                completed_at=now,
            )
        )
        db.flush()
        db.add(
            UsageEvent(
                event_id=other_event_id,
                execution_id=other_execution_id,
                workspace_id="separate-evaluation-workspace",
                provider="anthropic",
                exact_provider_model="claude-sonnet-5-20251001",
                input_tokens=900,
                output_tokens=900,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                source="direct-fixture",
                occurred_at=now,
                received_at=now,
                idempotency_identifier=str(other_event_id),
            )
        )
        db.commit()

    report_response = postgres_workers.request(
        0,
        "GET",
        f"/api/evaluations/summary?integration_name={integration_name}",
    )
    assert report_response.status_code == 200, report_response.text
    report = report_response.json()
    assert report["total_execution_count"] == 2
    assert report["eligible_completed_execution_count"] == 0
    assert report["excluded_execution_counts"]["prediction_has_multiple_executions"] == 2
    assert report["provenance_distribution"]["verified_provider"] == 0
