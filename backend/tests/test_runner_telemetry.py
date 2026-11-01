from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from agent_runner import run_task
from telemetry import CitadelTelemetry


class FakeMessages:
    def __init__(self, responses):
        self._responses = iter(responses)

    def create(self, **_kwargs):
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def fake_response(response_id: str, input_tokens: int, output_tokens: int, **usage_fields):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens, **usage_fields)
    return SimpleNamespace(
        id=response_id,
        model="claude-sonnet-5-20251001",
        usage=usage,
        content=[],
        stop_reason="end_turn",
    )


def test_runner_preserves_legacy_token_fields_and_exposes_missing_cache_data():
    response = fake_response("msg_1", 10, 4, cache_read_input_tokens=0)
    result = run_task(
        FakeClient([response]),
        {"id": "t1", "text": "Calculate 2 + 2", "tools": [], "category": "simple"},
    )

    assert result["total_input_tokens"] == 10
    assert result["total_output_tokens"] == 4
    assert result["total_tokens"] == 14
    assert result["total_processed_input_tokens"] is None
    assert result["turns"][0]["cache_creation_input_tokens"] is None
    assert result["turns"][0]["cache_read_input_tokens"] == 0


def test_runner_cache_inclusive_total_does_not_mutate_legacy_total():
    response = fake_response(
        "msg_2",
        10,
        4,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=2,
    )
    result = run_task(
        FakeClient([response]),
        {"id": "t1", "text": "Calculate 2 + 2", "tools": [], "category": "simple"},
    )

    assert result["total_input_tokens"] == 10
    assert result["total_tokens"] == 14
    assert result["total_processed_input_tokens"] == 15
    assert result["total_processed_input_tokens"] + result["total_output_tokens"] == 19


def test_telemetry_uses_stable_event_id_and_provider_reported_fields(monkeypatch):
    telemetry = CitadelTelemetry("https://citadel.example", "test-key")
    execution_id = UUID("39f04c63-65c3-4d84-9e6d-1bd9d9d26553")
    event = SimpleNamespace(execution_id=execution_id)
    response = fake_response(
        "msg_anthropic_123",
        10,
        4,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=2,
    )
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"event_id": payload["event_id"], "recorded": True}

    monkeypatch.setattr(telemetry, "_post", fake_post)
    telemetry.record_response(event, response)
    telemetry.record_response(event, response)
    telemetry.close()

    assert calls[0][0] == f"/api/executions/{execution_id}/usage"
    assert calls[0][1]["event_id"] == calls[1][1]["event_id"]
    assert calls[0][1]["provider"] == "anthropic"
    assert calls[0][1]["exact_provider_model"] == "claude-sonnet-5-20251001"
    assert calls[0][1]["input_tokens"] == 10
    assert calls[0][1]["output_tokens"] == 4
    assert calls[0][1]["cache_creation_input_tokens"] == 3
    assert calls[0][1]["cache_read_input_tokens"] == 2


def test_telemetry_marks_benchmark_execution_synthetic(monkeypatch):
    telemetry = CitadelTelemetry("https://citadel.example", "test-key")
    prediction_id = "c42ddf02-06ec-4d98-8ec1-3cdbf5c0d125"
    execution_id = "8082fc3a-c30d-4d5e-a81b-f1ea5f65b624"
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        if path.endswith("/predictions"):
            return {"prediction_id": prediction_id}
        return {"execution_id": execution_id}

    monkeypatch.setattr(telemetry, "_post", fake_post)
    execution = telemetry.start_execution(
        {"text": "Synthetic benchmark task", "tools": ["calculator"]},
        "claude-sonnet-5",
    )
    telemetry.close()

    assert execution.prediction_id == UUID(prediction_id)
    assert execution.execution_id == UUID(execution_id)
    assert calls[0][1]["model_id"] == "claude-sonnet"
    assert calls[1][1]["exact_provider_model"] == "claude-sonnet-5"
    assert calls[1][1]["is_synthetic"] is True
