import os
import sys
from pathlib import Path
import httpx
import pytest

# Ensure backend can be imported for integration testing
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "data_collection"))

try:
    from main import app
except ImportError:
    app = None

from citadel_predict.client import CitadelClient
from citadel_predict.errors import CitadelAuthError, CitadelBadRequestError, CitadelValidationError


@pytest.mark.skipif(app is None, reason="FastAPI backend app not importable")
def test_integration_live_backend_happy_path():
    # Use ASGI transport to test directly against backend FastAPI app
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client = CitadelClient(
            base_url="http://testserver",
            http_client=http_client,
        )
        res = client.predict(
            task_text="Research 3 competitor pricing models and draft a summary.",
            tools=["web_search", "draft_document"],
            model_id="claude-sonnet",
        )

        assert res["model_id"] == "claude-sonnet"
        assert res["low_tokens"] < res["expected_tokens"] <= res["high_tokens"]
        assert len(res["driving_factors"]) > 0
        assert isinstance(res["out_of_distribution"], bool)
        assert res["confidence"] in ["high", "medium", "low"]


@pytest.mark.skipif(app is None, reason="FastAPI backend app not importable")
def test_integration_live_backend_auth(monkeypatch):
    monkeypatch.setenv("CITADEL_API_KEY", "prod-secret-key-12345")
    transport = httpx.ASGITransport(app=app)

    # 1. Without auth -> should raise CitadelAuthError
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client_no_auth = CitadelClient(
            api_key=None,
            base_url="http://testserver",
            http_client=http_client,
        )
        with pytest.raises(CitadelAuthError):
            client_no_auth.predict(task_text="Run analysis", tools=["calculator"])

    # 2. With invalid key -> should raise CitadelAuthError
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client_bad_auth = CitadelClient(
            api_key="wrong-key",
            base_url="http://testserver",
            http_client=http_client,
        )
        with pytest.raises(CitadelAuthError):
            client_bad_auth.predict(task_text="Run analysis", tools=["calculator"])

    # 3. With valid key -> succeeds
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client_good_auth = CitadelClient(
            api_key="prod-secret-key-12345",
            base_url="http://testserver",
            http_client=http_client,
        )
        res = client_good_auth.predict(task_text="Run analysis", tools=["calculator"])
        assert res["expected_tokens"] > 0


@pytest.mark.skipif(app is None, reason="FastAPI backend app not importable")
def test_integration_live_backend_validation():
    transport = httpx.ASGITransport(app=app)
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client = CitadelClient(
            base_url="http://testserver",
            http_client=http_client,
        )
        with pytest.raises(CitadelValidationError):
            client.predict(task_text="")  # Empty string rejected by Pydantic min_length=1

        with pytest.raises(CitadelBadRequestError):
            client.predict(task_text="Valid task", model_id="non-existent-model")
