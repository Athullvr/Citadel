import pytest
from fastapi.testclient import TestClient
from main import app, limiter


@pytest.fixture
def client():
    # Clear rate limiter state before each test
    limiter.reset()
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "claude-sonnet" in data["supported_models"]


def test_list_tools_endpoint(client):
    res = client.get("/api/tools")
    assert res.status_code == 200
    tools = res.json()
    assert len(tools) == 7
    tool_names = [t["name"] for t in tools]
    assert "web_search" in tool_names
    assert "calculator" in tool_names


def test_validation_examples_endpoint(client):
    res = client.get("/api/validation-examples")
    assert res.status_code == 200
    examples = res.json()
    assert len(examples) == 4
    for ex in examples:
        assert "task_id" in ex
        assert "verdict" in ex
        assert "actual_tokens_observed" in ex


def test_predict_happy_path(client):
    payload = {
        "task_text": "Research 3 sources and summarize the findings.",
        "tools": ["web_search", "draft_document"],
        "model_id": "claude-sonnet",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model_id"] == "claude-sonnet"
    assert data["low_tokens"] < data["expected_tokens"] <= data["high_tokens"]
    assert len(data["driving_factors"]) > 0
    assert "confidence" in data
    assert isinstance(data["out_of_distribution"], bool)


def test_predict_ood_flags(client):
    # Unusually long task with no tools -> triggers OOD
    payload = {
        "task_text": "A" * 700,
        "tools": [],
        "model_id": "claude-sonnet",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["out_of_distribution"] is True
    assert data["confidence"] == "low"
    assert len(data["ood_reasons"]) >= 1


def test_predict_unsupported_model_id(client):
    payload = {
        "task_text": "Do something",
        "tools": [],
        "model_id": "non-existent-model-xyz",
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 400
    assert "Unsupported model_id" in res.json()["detail"]


def test_predict_validation_errors(client):
    # Empty task_text
    res = client.post("/api/predict", json={"task_text": "", "tools": []})
    assert res.status_code == 422

    # Exceeds max_length 4000
    res = client.post("/api/predict", json={"task_text": "X" * 4001, "tools": []})
    assert res.status_code == 422

    # Tool name too long
    res = client.post("/api/predict", json={"task_text": "valid text", "tools": ["T" * 65]})
    assert res.status_code == 422

    # Malformed JSON
    res = client.post(
        "/api/predict",
        content="bad-json-string",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422


def test_auth_enforcement_when_configured(client, monkeypatch):
    monkeypatch.setenv("CITADEL_API_KEY", "secret-test-key-12345")

    payload = {"task_text": "Research 5 competitors", "tools": ["web_search"]}

    # Missing auth header -> 401
    res_no_auth = client.post("/api/predict", json=payload)
    assert res_no_auth.status_code == 401
    assert "Missing or malformed" in res_no_auth.json()["detail"]

    # Wrong auth header -> 401
    res_bad_auth = client.post(
        "/api/predict",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert res_bad_auth.status_code == 401
    assert "Invalid API key" in res_bad_auth.json()["detail"]

    # Valid auth header -> 200
    res_good_auth = client.post(
        "/api/predict",
        json=payload,
        headers={"Authorization": "Bearer secret-test-key-12345"},
    )
    assert res_good_auth.status_code == 200
    assert res_good_auth.json()["expected_tokens"] > 0


def test_rate_limiting(client):
    limiter.reset()
    payload = {"task_text": "Calculate 2+2", "tools": ["calculator"]}

    # Hit 30 requests rapidly
    responses = [client.post("/api/predict", json=payload) for _ in range(35)]
    status_codes = [r.status_code for r in responses]

    # At least one 429 should be present
    assert 429 in status_codes
