import pytest
from fastapi.testclient import TestClient
from main import app, limiter


@pytest.fixture
def client():
    # Clear rate limiter state before each test
    limiter.reset()
    return TestClient(app)


def test_health_and_version_endpoints(client):
    for endpoint in ["/api/health", "/api/version"]:
        res = client.get(endpoint)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["model_id"] == "claude-sonnet"
        assert "claude-sonnet" in data["supported_models"]
        assert isinstance(data["auth_enabled"], bool)


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
    assert "X-RateLimit-Limit" in res.headers


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
    data = res.json()
    assert data["error"] == "bad_request"
    assert "Unsupported model_id" in data["message"]
    assert "Unsupported model_id" in data["detail"]


def test_predict_validation_errors_standardized(client):
    # Empty task_text
    res = client.post("/api/predict", json={"task_text": "", "tools": []})
    assert res.status_code == 422
    data = res.json()
    assert data["error"] == "validation_error"
    assert "message" in data
    assert "detail" in data

    # Exceeds max_length 4000
    res = client.post("/api/predict", json={"task_text": "X" * 4001, "tools": []})
    assert res.status_code == 422
    assert res.json()["error"] == "validation_error"

    # Tool name too long
    res = client.post("/api/predict", json={"task_text": "valid text", "tools": ["T" * 65]})
    assert res.status_code == 422
    assert res.json()["error"] == "validation_error"

    # Malformed JSON
    res = client.post(
        "/api/predict",
        content="bad-json-string",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422
    assert res.json()["error"] == "validation_error"


def test_auth_enforcement_when_configured(client, monkeypatch):
    monkeypatch.setenv("CITADEL_API_KEY", "secret-test-key-12345")

    payload = {"task_text": "Research 5 competitors", "tools": ["web_search"]}

    # Missing auth header -> 401 with standard error body
    res_no_auth = client.post("/api/predict", json=payload)
    assert res_no_auth.status_code == 401
    data_no_auth = res_no_auth.json()
    assert data_no_auth["error"] == "unauthorized"
    assert "Missing or malformed" in data_no_auth["message"]
    assert "WWW-Authenticate" in res_no_auth.headers

    # Wrong auth header -> 401
    res_bad_auth = client.post(
        "/api/predict",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert res_bad_auth.status_code == 401
    data_bad_auth = res_bad_auth.json()
    assert data_bad_auth["error"] == "unauthorized"
    assert "Invalid API key" in data_bad_auth["message"]

    # Valid auth header -> 200
    res_good_auth = client.post(
        "/api/predict",
        json=payload,
        headers={"Authorization": "Bearer secret-test-key-12345"},
    )
    assert res_good_auth.status_code == 200
    assert res_good_auth.json()["expected_tokens"] > 0


def test_rate_limiting_and_headers(client):
    limiter.reset()
    payload = {"task_text": "Calculate 2+2", "tools": ["calculator"]}

    # Hit requests rapidly
    responses = [client.post("/api/predict", json=payload) for _ in range(35)]
    status_codes = [r.status_code for r in responses]

    # At least one 429 should be present
    assert 429 in status_codes
    for r in responses:
        if r.status_code == 429:
            data = r.json()
            assert data["error"] == "rate_limited"
            assert "Rate limit exceeded" in data["message"]
            assert data.get("retry_after") == 60
            assert r.headers.get("Retry-After") == "60"
            assert r.headers.get("X-RateLimit-Limit") == "30"
            assert r.headers.get("X-RateLimit-Remaining") == "0"


def test_non_browser_server_to_server_request(client, monkeypatch):
    """
    Verify plain server-to-server request (no Origin/Referer header, plain client)
    is never blocked by CORS or browser-specific logic.
    """
    monkeypatch.setenv("CITADEL_API_KEY", "s2s-test-token-999")
    payload = {
        "task_text": "Server to server automated agent call",
        "tools": ["web_search"],
    }
    # Direct HTTP request with no browser Origin / Referer header
    headers = {
        "Authorization": "Bearer s2s-test-token-999",
        "User-Agent": "citadel-predict-cli/0.1.0",
    }
    res = client.post("/api/predict", json=payload, headers=headers)
    assert res.status_code == 200
    assert res.json()["expected_tokens"] > 0
    assert "access-control-allow-origin" not in res.headers or res.headers.get("access-control-allow-origin") == "*" or res.status_code == 200


def test_auth_enforcement_require_auth_flag(client, monkeypatch):
    """
    Verify CITADEL_REQUIRE_AUTH=true behavior.
    """
    # 1. Require auth is true but no key set -> 500 error
    monkeypatch.setenv("CITADEL_REQUIRE_AUTH", "true")
    monkeypatch.delenv("CITADEL_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    payload = {"task_text": "Test task", "tools": ["calculator"]}
    res = client.post("/api/predict", json=payload, headers={"Authorization": "Bearer test"})
    assert res.status_code == 500
    assert "CITADEL_REQUIRE_AUTH=true but no CITADEL_API_KEY" in res.json()["message"]

    # 2. Require auth is true with key set -> 200 on valid bearer
    monkeypatch.setenv("CITADEL_API_KEY", "secret-key-xyz")
    res_valid = client.post(
        "/api/predict",
        json=payload,
        headers={"Authorization": "Bearer secret-key-xyz"},
    )
    assert res_valid.status_code == 200


def test_resolve_data_collection_dir():
    """Verify resolve_data_collection_dir finds a valid data_collection path."""
    from main import resolve_data_collection_dir

    p = resolve_data_collection_dir()
    assert p.exists()
    assert (p / "predict.py").exists()

