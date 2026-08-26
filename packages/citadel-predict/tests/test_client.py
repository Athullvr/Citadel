import json
from unittest.mock import patch
import httpx
import pytest

from citadel_predict.client import CitadelClient, predict_cost


MOCK_PREDICT_RESPONSE = {
    "model_id": "claude-sonnet",
    "low_tokens": 1500,
    "expected_tokens": 3200,
    "high_tokens": 5800,
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
