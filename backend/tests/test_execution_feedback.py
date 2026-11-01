from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import execution_feedback
import pytest
from alembic import command
from alembic.config import Config
from database import create_database_engine
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent
from fastapi.testclient import TestClient
from main import app
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
API_KEY = "execution-feedback-test-secret"
WORKSPACE_ID = "citadel-test-workspace"


@pytest.fixture
def execution_client(tmp_path, monkeypatch):
    """Run each API test against its own migrated SQLite database file."""
    db_path = tmp_path / "citadel-execution-test.sqlite"
    database_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)

    config = Config(str(BACKEND_DIR / "alembic.ini"))
    command.upgrade(config, "head")

    with TestClient(app) as client:
        yield client


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def prediction_request(task_text: str = "Research a bounded public topic.") -> dict:
    return {"task_text": task_text, "tools": ["web_search"], "model_id": "claude-sonnet"}


def execution_request(prediction_id: str | None = None) -> dict:
    return {
        "prediction_id": prediction_id,
        "integration_name": "test-agent",
        "integration_version": "0.1.0",
        "provider": "anthropic",
        "exact_provider_model": "claude-sonnet-5",
    }


def usage_request(event_id: UUID | None = None, **overrides) -> dict:
    payload = {
        "event_id": str(event_id or uuid4()),
        "provider": "anthropic",
        "exact_provider_model": "claude-sonnet-5",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "source": "anthropic_messages_api",
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    payload.update(overrides)
    return payload


def _db_session(monkeypatch) -> Session:
    database_url = __import__("os").environ["CITADEL_DATABASE_URL"]
    engine = create_database_engine(database_url)
    return Session(engine, expire_on_commit=False)


def test_prediction_snapshot_is_persisted_without_raw_task(execution_client, monkeypatch):
    task_text = "Highly specific private prompt that must not be retained."
    response = execution_client.post(
        "/api/executions/predictions", json=prediction_request(task_text), headers=auth_headers()
    )

    assert response.status_code == 201
    data = response.json()
    assert data["prediction_id"]
    assert data["created_at"]
    assert data["model_id"] == "claude-sonnet"
    assert data["expected_tokens"] > 0

    with _db_session(monkeypatch) as db:
        record = db.get(PredictionRecord, UUID(data["prediction_id"]))
        assert record is not None
        assert record.workspace_id == WORKSPACE_ID
        assert record.predicted_expected_tokens == data["expected_tokens"]
        assert record.estimated_expected_cost == data["expected_cost_usd"]
        assert record.model_alias == "claude-sonnet"
        assert record.model_artifact_version.startswith("sha256:")
        assert record.feature_schema_version == "citadel-features-v1"
        assert (
            record.prediction_payload_snapshot["response"]["expected_tokens"]
            == data["expected_tokens"]
        )
        assert record.prediction_payload_snapshot["pricing_assumptions"]["input_ratio"] == 0.75
        assert task_text not in str(record.prediction_payload_snapshot)


def test_prediction_snapshot_is_immutable(execution_client, monkeypatch):
    created = execution_client.post(
        "/api/executions/predictions", json=prediction_request(), headers=auth_headers()
    ).json()

    with _db_session(monkeypatch) as db:
        record = db.get(PredictionRecord, UUID(created["prediction_id"]))
        record.model_alias = "tampered"
        with pytest.raises(SQLAlchemyError):
            db.commit()
        db.rollback()
        assert (
            db.get(PredictionRecord, UUID(created["prediction_id"])).model_alias
            == "claude-sonnet"
        )


def test_execution_can_be_created_with_or_without_prediction(execution_client):
    no_prediction = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    )
    assert no_prediction.status_code == 201
    assert no_prediction.json()["prediction_id"] is None
    assert no_prediction.json()["status"] == "started"

    prediction = execution_client.post(
        "/api/executions/predictions", json=prediction_request(), headers=auth_headers()
    ).json()
    linked = execution_client.post(
        "/api/executions",
        json=execution_request(prediction["prediction_id"]),
        headers=auth_headers(),
    )
    assert linked.status_code == 201
    assert linked.json()["prediction_id"] == prediction["prediction_id"]


def test_prediction_and_execution_reads_are_workspace_scoped(execution_client, monkeypatch):
    prediction = execution_client.post(
        "/api/executions/predictions", json=prediction_request(), headers=auth_headers()
    ).json()
    execution = execution_client.post(
        "/api/executions",
        json=execution_request(prediction["prediction_id"]),
        headers=auth_headers(),
    ).json()

    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", "different-server-workspace")
    linked = execution_client.post(
        "/api/executions",
        json=execution_request(prediction["prediction_id"]),
        headers=auth_headers(),
    )
    read = execution_client.get(
        f"/api/executions/{execution['execution_id']}", headers=auth_headers()
    )
    assert linked.status_code == 404
    assert read.status_code == 404


def test_usage_ingestion_is_idempotent_and_rejects_conflicts(execution_client):
    execution = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    ).json()
    event_id = uuid4()
    payload = usage_request(event_id)
    first = execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage", json=payload, headers=auth_headers()
    )
    duplicate = execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage", json=payload, headers=auth_headers()
    )
    conflicting = execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage",
        json={**payload, "output_tokens": 6},
        headers=auth_headers(),
    )

    assert first.status_code == 200 and first.json()["recorded"] is True
    assert duplicate.status_code == 200 and duplicate.json()["recorded"] is False
    assert conflicting.status_code == 409


def test_usage_aggregates_multiple_anthropic_events_and_cache_tokens(execution_client):
    execution = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    ).json()
    first = usage_request(
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
    for payload in (first, second):
        response = execution_client.post(
            f"/api/executions/{execution['execution_id']}/usage",
            json=payload,
            headers=auth_headers(),
        )
        assert response.status_code == 200

    detail = execution_client.get(
        f"/api/executions/{execution['execution_id']}", headers=auth_headers()
    ).json()
    assert detail["usage_event_count"] == 2
    assert detail["usage"] == {
        "input_tokens": 30,
        "output_tokens": 12,
        "cache_creation_input_tokens": 5,
        "cache_creation_input_tokens_known_sum": 5,
        "cache_creation_missing_event_count": 0,
        "cache_read_input_tokens": 10,
        "cache_read_input_tokens_known_sum": 10,
        "cache_read_missing_event_count": 0,
        "total_input_tokens": 45,
        "total_tokens": 57,
    }


def test_usage_distinguishes_missing_cache_fields_from_zero(execution_client):
    execution = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    ).json()
    payload = usage_request(cache_creation_input_tokens=None, cache_read_input_tokens=0)
    response = execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage", json=payload, headers=auth_headers()
    )
    detail = execution_client.get(
        f"/api/executions/{execution['execution_id']}", headers=auth_headers()
    ).json()

    assert response.status_code == 200
    assert detail["usage"]["cache_creation_input_tokens"] is None
    assert detail["usage"]["cache_creation_input_tokens_known_sum"] is None
    assert detail["usage"]["cache_creation_missing_event_count"] == 1
    assert detail["usage"]["cache_read_input_tokens"] == 0
    assert detail["usage"]["total_input_tokens"] is None
    assert detail["usage"]["total_tokens"] is None


def test_execution_completion_is_terminal_and_returns_aggregate(execution_client):
    execution = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    ).json()
    completed_at = datetime.now(UTC).isoformat()
    result = execution_client.post(
        f"/api/executions/{execution['execution_id']}/complete",
        json={"status": "completed", "completed_at": completed_at},
        headers=auth_headers(),
    )
    second = execution_client.post(
        f"/api/executions/{execution['execution_id']}/complete",
        json={"status": "failed"},
        headers=auth_headers(),
    )
    late_usage = execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage",
        json=usage_request(),
        headers=auth_headers(),
    )

    assert result.status_code == 200
    assert result.json()["execution"]["status"] == "completed"
    assert second.status_code == 409
    assert late_usage.status_code == 409


def test_authentication_fails_closed_for_feedback_endpoints(execution_client, monkeypatch):
    monkeypatch.delenv("CITADEL_API_KEY", raising=False)
    no_configured_key = execution_client.post("/api/executions", json=execution_request())
    monkeypatch.setenv("CITADEL_API_KEY", API_KEY)
    missing_bearer = execution_client.post("/api/executions", json=execution_request())

    assert no_configured_key.status_code == 503
    assert missing_bearer.status_code == 401
    assert missing_bearer.headers["www-authenticate"] == "Bearer"


def test_request_workspace_id_is_rejected(execution_client):
    response = execution_client.post(
        "/api/executions",
        json={**execution_request(), "workspace_id": "attacker-workspace"},
        headers=auth_headers(),
    )
    assert response.status_code == 422


def test_failed_execution_insert_rolls_back_transaction(execution_client, monkeypatch):
    fixed_execution_id = uuid4()
    monkeypatch.setattr(execution_feedback, "uuid4", lambda: fixed_execution_id)
    first = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    )
    collision = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    )

    assert first.status_code == 201
    assert collision.status_code == 503
    with _db_session(monkeypatch) as db:
        assert db.scalar(select(func.count()).select_from(ExecutionRecord)) == 1


def test_usage_event_rows_are_append_only(execution_client, monkeypatch):
    execution = execution_client.post(
        "/api/executions", json=execution_request(), headers=auth_headers()
    ).json()
    payload = usage_request()
    execution_client.post(
        f"/api/executions/{execution['execution_id']}/usage", json=payload, headers=auth_headers()
    )

    with _db_session(monkeypatch) as db:
        event = db.get(UsageEvent, UUID(payload["event_id"]))
        event.output_tokens = 99
        with pytest.raises(SQLAlchemyError):
            db.commit()
        db.rollback()
        assert db.get(UsageEvent, UUID(payload["event_id"])).output_tokens == 5


def test_legacy_prediction_contract_is_unchanged(execution_client):
    legacy = execution_client.post(
        "/api/predict",
        json={"task_text": "Calculate a small sum.", "tools": ["calculator"]},
        headers=auth_headers(),
    )
    assert legacy.status_code == 200
    assert "prediction_id" not in legacy.json()
    assert legacy.json()["model_id"] == "claude-sonnet"

    openapi = execution_client.get("/openapi.json").json()
    response_schema = openapi["paths"]["/api/predict"]["post"]["responses"]["200"]
    response_schema = response_schema["content"]["application/json"]["schema"]["$ref"]
    assert response_schema.endswith("PredictResponse")

