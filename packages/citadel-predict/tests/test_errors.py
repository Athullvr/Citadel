import httpx
import pytest

from citadel_predict.client import CitadelClient, predict_cost
from citadel_predict.errors import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
)


def test_error_401_unauthorized():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"detail": "Missing or malformed Authorization header."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelAuthError) as exc_info:
            client.predict(task_text="test task", tools=["search"])

        err = exc_info.value
        assert err.status_code == 401
        assert "Authentication failed" in err.message
        assert "CITADEL_API_KEY" in err.message


def test_error_429_rate_limited():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"detail": "Rate limit exceeded. 30 per 1 minute"},
            headers={"Retry-After": "60"},
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelRateLimitError) as exc_info:
            client.predict(task_text="test task")

        err = exc_info.value
        assert err.status_code == 429
        assert err.retry_after == 60
        assert "Retry-After: 60s" in str(err)


def test_error_422_validation():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "task_text"],
                        "msg": "String should have at least 1 character",
                        "type": "string_too_short",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelValidationError) as exc_info:
            client.predict(task_text="")

        err = exc_info.value
        assert err.status_code == 422
        assert "task_text" in err.message


def test_error_400_bad_request():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": "Unsupported model_id 'unknown-model'"},
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelBadRequestError) as exc_info:
            client.predict(task_text="test task", model_id="unknown-model")

        err = exc_info.value
        assert err.status_code == 400
        assert "Unsupported model_id" in err.message


def test_error_500_server_error():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"detail": "Model weights could not be loaded"},
        )

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelServerError) as exc_info:
            client.predict(task_text="test task")

        err = exc_info.value
        assert err.status_code == 500
        assert "Server error" in err.message


def test_error_network_timeout():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as http_client:
        client = CitadelClient(base_url="http://test", http_client=http_client)
        with pytest.raises(CitadelNetworkError) as exc_info:
            client.predict(task_text="test task")

        err = exc_info.value
        assert "Network error" in err.message
        assert isinstance(err.cause, httpx.ReadTimeout)
