"""PostgreSQL constraint, rollback, connection-loss, and restart checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from execution_models import ExecutionRecord
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from postgres.conftest import PostgreSQLWorkers
from postgres.helpers import create_execution

pytestmark = pytest.mark.postgres


def test_completion_constraint_failure_rolls_back_status_and_timestamp(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    execution = create_execution(postgres_workers)
    execution_id = UUID(execution["execution_id"])

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE FUNCTION citadel_test_reject_completion() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN "
                "RAISE EXCEPTION 'injected completion failure'; END; $$"
            )
        )
        connection.execute(
            text(
                "CREATE TRIGGER citadel_test_reject_completion "
                "BEFORE UPDATE OF status ON execution_records FOR EACH ROW "
                "EXECUTE FUNCTION citadel_test_reject_completion()"
            )
        )

    try:
        failed = postgres_workers.request(
            0,
            "POST",
            f"/api/executions/{execution_id}/complete",
            payload={"status": "completed"},
        )
        assert failed.status_code == 503, failed.text
        with Session(postgres_engine) as db:
            record = db.get(ExecutionRecord, execution_id)
            assert record is not None
            assert record.status == "started"
            assert record.completed_at is None
    finally:
        with postgres_engine.begin() as connection:
            connection.execute(
                text(
                    "DROP TRIGGER IF EXISTS citadel_test_reject_completion "
                    "ON execution_records"
                )
            )
            connection.execute(
                text("DROP FUNCTION IF EXISTS citadel_test_reject_completion()")
            )

    recovered = postgres_workers.request(
        1,
        "POST",
        f"/api/executions/{execution_id}/complete",
        payload={"status": "completed", "completed_at": datetime.now(UTC).isoformat()},
    )
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["execution"]["status"] == "completed"
    with Session(postgres_engine) as db:
        record = db.scalar(
            select(ExecutionRecord).where(ExecutionRecord.execution_id == execution_id)
        )
        assert record is not None and record.completed_at is not None


def test_workers_recover_after_their_database_connections_are_terminated(
    postgres_workers: PostgreSQLWorkers,
    postgres_engine,
):
    executions = [create_execution(postgres_workers, index) for index in range(2)]
    for index, execution in enumerate(executions):
        response = postgres_workers.request(
            index, "GET", f"/api/executions/{execution['execution_id']}"
        )
        assert response.status_code == 200

    application_names = ["citadel_pg_gate_worker_0", "citadel_pg_gate_worker_1"]
    with postgres_engine.begin() as connection:
        active = connection.execute(
            text(
                "SELECT pid, application_name FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "AND application_name IN (:worker_a, :worker_b) "
                "AND pid <> pg_backend_pid()"
            ),
            {"worker_a": application_names[0], "worker_b": application_names[1]},
        ).all()
        assert set(row.application_name for row in active) == set(application_names)
        terminated = connection.execute(
            text(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() "
                "AND application_name IN (:worker_a, :worker_b) "
                "AND pid <> pg_backend_pid()"
            ),
            {"worker_a": application_names[0], "worker_b": application_names[1]},
        ).scalars().all()
    assert len(terminated) >= 2 and all(terminated)

    for index, execution in enumerate(executions):
        recovered = postgres_workers.request(
            index, "GET", f"/api/executions/{execution['execution_id']}"
        )
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["execution"]["status"] == "started"


def test_disposable_postgres_container_restart_preserves_data_and_pool_recovery(
    postgres_workers: PostgreSQLWorkers,
):
    service_id = os.environ.get("CITADEL_TEST_POSTGRES_SERVICE_ID", "").strip()
    if os.environ.get("GITHUB_ACTIONS") != "true" or not service_id:
        pytest.skip("Physical PostgreSQL restart is exercised by the GitHub Actions service job")
    if not re.fullmatch(r"[0-9a-f]{64}", service_id):
        pytest.fail("CITADEL_TEST_POSTGRES_SERVICE_ID is not a Docker container ID")

    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config}}", service_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    config = json.loads(inspected.stdout)
    container_environment = set(config.get("Env", []))
    expected_container = config.get("Image") == "postgres:16" and (
        "POSTGRES_DB=citadel_test" in container_environment
    )
    if not expected_container:
        pytest.fail(
            "Refusing to restart a container outside the disposable PostgreSQL test service"
        )

    execution = create_execution(postgres_workers)
    execution_id = execution["execution_id"]
    event_id = uuid4()
    event_payload = {
        "event_id": str(event_id),
        "provider": "anthropic",
        "exact_provider_model": "claude-sonnet-5-20251001",
        "input_tokens": 321,
        "output_tokens": 87,
        "cache_creation_input_tokens": 21,
        "cache_read_input_tokens": 13,
        "source": "postgres_restart_test",
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    recorded = postgres_workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=event_payload,
    )
    assert recorded.status_code == 200 and recorded.json()["recorded"] is True

    subprocess.run(["docker", "restart", service_id], check=True, timeout=90)

    deadline = time.monotonic() + 60
    detail_response: httpx.Response | None = None
    while time.monotonic() < deadline:
        try:
            detail_response = postgres_workers.request(
                1, "GET", f"/api/executions/{execution_id}", timeout=5
            )
            if detail_response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)

    assert detail_response is not None and detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["execution"]["execution_id"] == execution_id
    assert detail["usage_event_count"] == 1
    assert detail["usage"]["input_tokens"] == 321
    assert detail["usage"]["output_tokens"] == 87

    duplicate = postgres_workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/usage",
        payload=event_payload,
    )
    assert duplicate.status_code == 200
    assert duplicate.json() == {"event_id": str(event_id), "recorded": False}

    completed = postgres_workers.request(
        0,
        "POST",
        f"/api/executions/{execution_id}/complete",
        payload={"status": "completed"},
    )
    assert completed.status_code == 200, completed.text
    reread = postgres_workers.request(1, "GET", f"/api/executions/{execution_id}")
    assert reread.status_code == 200
    assert reread.json()["execution"]["status"] == "completed"
