"""Shared HTTP and concurrency helpers for PostgreSQL integration tests."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from uuid import UUID, uuid4

from postgres.conftest import PostgreSQLWorkers


def run_simultaneously[T](count: int, operation: Callable[[int], T]) -> list[T]:
    """Release all independent HTTP requests from separate threads together."""
    barrier = Barrier(count)

    def start(index: int) -> T:
        barrier.wait(timeout=30)
        return operation(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(start, index) for index in range(count)]
        return [future.result(timeout=90) for future in futures]


def create_execution(
    workers: PostgreSQLWorkers,
    worker_index: int = 0,
    prediction_id: str | None = None,
) -> dict:
    response = workers.request(
        worker_index,
        "POST",
        "/api/executions",
        payload={
            "prediction_id": prediction_id,
            "integration_name": "postgres-release-gate",
            "integration_version": "0.3.0-test",
            "provider": "anthropic",
            "exact_provider_model": "claude-sonnet-5-20251001",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_prediction(workers: PostgreSQLWorkers, worker_index: int = 0) -> dict:
    response = workers.request(
        worker_index,
        "POST",
        "/api/executions/predictions",
        payload={
            "task_text": "Research a bounded public topic for PostgreSQL integration testing.",
            "tools": ["web_search"],
            "model_id": "claude-sonnet",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def usage_payload(
    event_id: UUID | None = None,
    *,
    input_tokens: int = 120,
    output_tokens: int = 30,
) -> dict:
    return {
        "event_id": str(event_id or uuid4()),
        "provider": "anthropic",
        "exact_provider_model": "claude-sonnet-5-20251001",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 10,
        "source": "postgres-release-gate",
        "occurred_at": datetime.now(UTC).isoformat(),
    }
