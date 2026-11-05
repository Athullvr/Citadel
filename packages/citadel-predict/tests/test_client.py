import json
from unittest.mock import patch

import httpx
from citadel_predict.client import CitadelClient, predict_cost

MOCK_PREDICT_RESPONSE = {
    "model_id": "claude-sonnet",
    "low_tokens": 1500,
    "expected_tokens": 3200,
    "high_tokens": 5800,
    "expected_cost_usd": 0.0192,
    "low_cost_usd": 0.009,
    "high_cost_usd": 0.0348,
    "cost_note": "Cost estimated using Anthropic published rates",
    "driving_factors": ["Tool count (2)", "Research action verb"],
    "features": {"char_len": 45, "word_count": 8, "num_tools": 2},
    "confidence": "high",
    "out_of_distribution": False,
    "ood_reasons": [],
}


def test_predict_happy_path():
    captured_request = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=MOCK_PREDICT_RESPONSE)

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(
            api_key="citadel_sec_12345",
            base_url="http://api.citadel.internal",
            http_client=http_client,
        )
        result = client.predict(
            task_text="Research competitor pricing",
            tools=["web_search", "draft_document"],
            model_id="claude-sonnet",
        )

        # Exact schema preservation
        assert result == MOCK_PREDICT_RESPONSE
        assert result["expected_tokens"] == 3200
        assert result["low_tokens"] == 1500
        assert result["high_tokens"] == 5800
        assert result["expected_cost_usd"] == 0.0192
        assert result["low_cost_usd"] == 0.009
        assert result["high_cost_usd"] == 0.0348
        assert result["confidence"] == "high"
        assert result["out_of_distribution"] is False

        # Verify request sent
        assert captured_request["url"] == "http://api.citadel.internal/api/predict"
        assert captured_request["headers"]["authorization"] == "Bearer citadel_sec_12345"
        assert captured_request["headers"]["content-type"] == "application/json"
        assert captured_request["body"] == {
            "task_text": "Research competitor pricing",
            "tools": ["web_search", "draft_document"],
            "model_id": "claude-sonnet",
        }


def test_predict_without_auth_header():
    captured_request = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_request["headers"] = dict(request.headers)
        return httpx.Response(200, json=MOCK_PREDICT_RESPONSE)

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://localhost:8000", http_client=http_client)
        result = client.predict(task_text="Simple calculation")
        assert result["expected_tokens"] == 3200
        assert "authorization" not in captured_request["headers"]


def test_predict_cost_function_wrapper(monkeypatch):
    def mock_post(url, json=None, headers=None, timeout=None):
        class MockResp:
            status_code = 200

            def json(self):
                return MOCK_PREDICT_RESPONSE

        return MockResp()

    with patch("httpx.Client.post", side_effect=mock_post):
        result = predict_cost(
            task_text="Analyze market trends",
            tools=["search"],
            api_key="test-key",
            api_url="http://mock-api:8000",
        )
        assert result == MOCK_PREDICT_RESPONSE


def test_client_health():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "ok", "supported_models": ["claude-sonnet"], "auth_enabled": True},
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        health = client.health()
        assert health["status"] == "ok"
        assert "claude-sonnet" in health["supported_models"]


def test_execution_feedback_methods_use_existing_authenticated_routes():
    requests = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8")) if request.content else None
        requests.append((request.method, request.url.path, payload))
        if request.url.path.endswith("/predictions"):
            return httpx.Response(201, json={"prediction_id": "prediction-1"})
        if request.url.path == "/api/executions":
            return httpx.Response(201, json={"execution_id": "execution-1"})
        if request.url.path.endswith("/usage"):
            return httpx.Response(200, json={"event_id": "event-1", "recorded": True})
        if request.url.path.endswith("/complete"):
            return httpx.Response(200, json={"execution": {"status": "cancelled"}})
        return httpx.Response(200, json={"execution": {"status": "started"}})

    with httpx.Client(transport=httpx.MockTransport(mock_handler)) as http_client:
        client = CitadelClient(
            api_key="execution-key",
            base_url="http://citadel.test",
            http_client=http_client,
        )
        client.create_execution_prediction("task text", tools=["search"])
        client.create_execution(
            prediction_id="prediction-1",
            integration_name="mcp-agent",
            integration_version="1.0",
            provider="anthropic",
            exact_provider_model="claude-sonnet-5",
        )
        client.record_execution_usage(
            "execution-1",
            event_id="event-1",
            provider="anthropic",
            exact_provider_model="claude-sonnet-5",
            input_tokens=10,
            output_tokens=2,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
            occurred_at="2026-11-05T00:00:00+00:00",
        )
        client.complete_execution(
            "execution-1", status="cancelled", expected_usage_event_count=1
        )
        client.get_execution_summary("execution-1")

    assert [item[:2] for item in requests] == [
        ("POST", "/api/executions/predictions"),
        ("POST", "/api/executions"),
        ("POST", "/api/executions/execution-1/usage"),
        ("POST", "/api/executions/execution-1/complete"),
        ("GET", "/api/executions/execution-1"),
    ]
    usage_body = requests[2][2]
    assert usage_body["source"] == "mcp_tool_submission"
    assert usage_body["cache_creation_input_tokens"] is None
    assert "workspace_id" not in requests[1][2]
    assert requests[3][2] == {
        "status": "cancelled",
        "expected_usage_event_count": 1,
    }
