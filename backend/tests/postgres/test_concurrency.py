"""Concurrent HTTP requests across independent FastAPI processes."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent
from fastapi.testclient import TestClient
from main import app
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from postgres.conftest import PostgreSQLDatabase, PostgreSQLWorkers
from postgres.helpers import create_execution, create_prediction, run_simultaneously, usage_payload

pytestmark = pytest.mark.postgres


def test_twenty_concurrent_identical_usage_requests_are_counted_once(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    execution = create_execution(postgres_workers)
    execution_id = UUID(execution["execution_id"])
    event_id = uuid4()
    payload = usage_payload(event_id)

    responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            f"/api/executions/{execution_id}/usage",
            payload=payload,
        ),
    )

    assert all(response.status_code == 200 for response in responses)
    response_bodies = [response.json() for response in responses]
    assert sum(body["recorded"] is True for body in response_bodies) == 1
    assert sum(body["recorded"] is False for body in response_bodies) == 19
    with Session(postgres_engine) as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(UsageEvent)
                .where(UsageEvent.execution_id == execution_id)
            )
            == 1
        )

    detail = postgres_workers.request(
        1, "GET", f"/api/executions/{execution_id}"
    ).json()
    assert detail["usage_event_count"] == 1
    assert detail["usage"]["input_tokens"] == payload["input_tokens"]
    assert detail["usage"]["output_tokens"] == payload["output_tokens"]

    retry = postgres_workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=payload,
    )
    assert retry.status_code == 200
    assert retry.json() == {"event_id": str(event_id), "recorded": False}


def test_concurrent_conflicting_usage_payloads_preserve_one_accepted_value(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    execution = create_execution(postgres_workers)
    execution_id = UUID(execution["execution_id"])
    event_id = uuid4()
    payloads = [usage_payload(event_id, output_tokens=100 + index) for index in range(20)]

    indexed_responses = run_simultaneously(
        20,
        lambda index: (
            index,
            postgres_workers.request(
                index % 2,
                "POST",
                f"/api/executions/{execution_id}/usage",
                payload=payloads[index],
            ),
        ),
    )

    accepted = [
        (index, response)
        for index, response in indexed_responses
        if response.status_code == 200
    ]
    rejected = [response for _, response in indexed_responses if response.status_code == 409]
    unexpected = [
        response.text
        for _, response in indexed_responses
        if response.status_code not in {200, 409}
    ]
    assert not unexpected
    assert len(accepted) == 1
    assert len(rejected) == 19
    assert accepted[0][1].json()["recorded"] is True

    accepted_payload = payloads[accepted[0][0]]
    with Session(postgres_engine) as db:
        event = db.get(UsageEvent, event_id)
        assert event is not None
        assert event.execution_id == execution_id
        assert event.input_tokens == accepted_payload["input_tokens"]
        assert event.output_tokens == accepted_payload["output_tokens"]
        assert event.cache_creation_input_tokens == accepted_payload["cache_creation_input_tokens"]
        assert event.cache_read_input_tokens == accepted_payload["cache_read_input_tokens"]
        assert (
            db.scalar(
                select(func.count())
                .select_from(UsageEvent)
                .where(UsageEvent.execution_id == execution_id)
            )
            == 1
        )


def test_event_id_collision_across_executions_rejects_losing_transaction(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    executions = [create_execution(postgres_workers, index) for index in range(2)]
    event_id = uuid4()
    payload = usage_payload(event_id)

    indexed_responses = run_simultaneously(
        2,
        lambda index: (
            index,
            postgres_workers.request(
                index,
                "POST",
                f"/api/executions/{executions[index]['execution_id']}/usage",
                payload=payload,
            ),
        ),
    )

    accepted = [
        (index, response)
        for index, response in indexed_responses
        if response.status_code == 200
    ]
    rejected = [response for _, response in indexed_responses if response.status_code == 409]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0][1].json()["recorded"] is True
    winning_execution_id = UUID(executions[accepted[0][0]]["execution_id"])

    with Session(postgres_engine) as db:
        event = db.get(UsageEvent, event_id)
        assert event is not None
        assert event.execution_id == winning_execution_id
        assert (
            db.scalar(
                select(func.count())
                .select_from(UsageEvent)
                .where(UsageEvent.event_id == event_id)
            )
            == 1
        )


def test_concurrent_prediction_and_execution_creation(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    prediction_responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            "/api/executions/predictions",
            payload={
                "task_text": f"Concurrent prediction request {index}.",
                "tools": ["web_search"],
                "model_id": "claude-sonnet",
            },
        ),
    )
    assert all(response.status_code == 201 for response in prediction_responses)
    prediction_ids = [UUID(response.json()["prediction_id"]) for response in prediction_responses]
    assert len(set(prediction_ids)) == 20

    execution_responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            "/api/executions",
            payload={
                "prediction_id": str(prediction_ids[index]),
                "integration_name": "postgres-release-gate",
                "integration_version": "0.3.0-test",
                "provider": "anthropic",
                "exact_provider_model": "test-anthropic-model-v1",
            },
        ),
    )
    assert all(response.status_code == 201 for response in execution_responses)
    execution_ids = [UUID(response.json()["execution_id"]) for response in execution_responses]
    assert len(set(execution_ids)) == 20

    with Session(postgres_engine) as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(PredictionRecord)
                .where(PredictionRecord.prediction_id.in_(prediction_ids))
            )
            == 20
        )
        assert (
            db.scalar(
                select(func.count())
                .select_from(ExecutionRecord)
                .where(ExecutionRecord.execution_id.in_(execution_ids))
            )
            == 20
        )


def test_concurrent_completion_requests_have_one_terminal_transition(
    postgres_workers: PostgreSQLWorkers,
):
    execution = create_execution(postgres_workers)
    execution_id = UUID(execution["execution_id"])

    responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            f"/api/executions/{execution_id}/complete",
            payload={"status": "completed"},
        ),
    )
    assert sum(response.status_code == 200 for response in responses) == 1
    assert sum(response.status_code == 409 for response in responses) == 19
    success = next(response for response in responses if response.status_code == 200)
    assert success.json()["execution"]["status"] == "completed"
    detail = postgres_workers.request(
        1, "GET", f"/api/executions/{execution_id}"
    ).json()
    assert detail["execution"]["status"] == "completed"


def test_concurrent_usage_for_different_executions_is_isolated(
    postgres_workers: PostgreSQLWorkers,
):
    execution_responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            "/api/executions",
            payload={
                "integration_name": "postgres-release-gate",
                "provider": "anthropic",
                "exact_provider_model": "test-anthropic-model-v1",
            },
        ),
    )
    assert all(response.status_code == 201 for response in execution_responses)
    execution_ids = [response.json()["execution_id"] for response in execution_responses]
    payloads = [
        usage_payload(input_tokens=200 + index, output_tokens=50 + index)
        for index in range(20)
    ]

    responses = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            index % 2,
            "POST",
            f"/api/executions/{execution_ids[index]}/usage",
            payload=payloads[index],
        ),
    )
    assert all(
        response.status_code == 200 and response.json()["recorded"]
        for response in responses
    )

    details = run_simultaneously(
        20,
        lambda index: postgres_workers.request(
            (index + 1) % 2,
            "GET",
            f"/api/executions/{execution_ids[index]}",
        ),
    )
    for index, response in enumerate(details):
        assert response.status_code == 200
        detail = response.json()
        assert detail["usage_event_count"] == 1
        assert detail["usage"]["input_tokens"] == payloads[index]["input_tokens"]
        assert detail["usage"]["output_tokens"] == payloads[index]["output_tokens"]


def test_records_are_shared_between_workers_and_workspace_filtering_remains_active(
    postgres_workers: PostgreSQLWorkers,
    postgres_database: PostgreSQLDatabase,
    monkeypatch,
):
    prediction = create_prediction(postgres_workers, worker_index=0)
    prediction_id = prediction["prediction_id"]
    execution = create_execution(postgres_workers, worker_index=1, prediction_id=prediction_id)
    execution_id = execution["execution_id"]

    read_from_worker_a = postgres_workers.request(0, "GET", f"/api/executions/{execution_id}")
    assert read_from_worker_a.status_code == 200
    assert read_from_worker_a.json()["prediction"]["prediction_id"] == prediction_id

    event = usage_payload()
    ingested_by_worker_b = postgres_workers.request(
        1,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=event,
    )
    assert ingested_by_worker_b.status_code == 200
    read_from_worker_a = postgres_workers.request(0, "GET", f"/api/executions/{execution_id}")
    assert read_from_worker_a.json()["usage_event_count"] == 1

    monkeypatch.setenv("CITADEL_DATABASE_URL", postgres_database.database_url)
    monkeypatch.setenv("CITADEL_API_KEY", postgres_workers.api_key)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", "different-test-workspace")
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {postgres_workers.api_key}"}
        assert client.get(f"/api/executions/{execution_id}", headers=headers).status_code == 404
        assert (
            client.post(
                "/api/executions",
                json={
                    "prediction_id": prediction_id,
                    "integration_name": "postgres-release-gate",
                    "provider": "anthropic",
                    "exact_provider_model": "test-anthropic-model-v1",
                },
                headers=headers,
            ).status_code
            == 404
        )
        assert (
            client.post(
                f"/api/executions/{execution_id}/usage",
                json=usage_payload(),
                headers=headers,
            ).status_code
            == 404
        )
