import json
from unittest.mock import patch

import pytest
from citadel_predict.cli import (
    EXIT_AUTH_ERROR,
    EXIT_GENERAL_ERROR,
    EXIT_NETWORK_ERROR,
    EXIT_RATE_LIMIT_ERROR,
    EXIT_SERVER_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
    main,
)
from citadel_predict.errors import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
)

SAMPLE_RESULT = {
    "model_id": "claude-sonnet",
    "low_tokens": 1200,
    "expected_tokens": 3000,
    "high_tokens": 5500,
    "driving_factors": ["Tool count (2)", "Action verbs"],
    "features": {"num_tools": 2},
    "confidence": "high",
    "out_of_distribution": False,
    "ood_reasons": [],
}


def test_cli_success_formatted(capsys):
    with patch("citadel_predict.cli.CitadelClient.predict", return_value=SAMPLE_RESULT):
        code = main(["--task", "Research competitors", "--tools", "web_search,fetch_url"])
        assert code == EXIT_SUCCESS
        captured = capsys.readouterr()
        assert "CITADEL PREDICT — TOKEN BUDGET ESTIMATE" in captured.out
        assert "3,000 tokens" in captured.out
        assert "1,200 – 5,500 tokens" in captured.out
        assert "Tool count (2)" in captured.out


def test_cli_success_json(capsys):
    with patch("citadel_predict.cli.CitadelClient.predict", return_value=SAMPLE_RESULT):
        code = main(["--task", "Research competitors", "--json"])
        assert code == EXIT_SUCCESS
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["expected_tokens"] == 3000
        assert parsed["model_id"] == "claude-sonnet"


def test_cli_missing_task(capsys):
    code = main([])
    assert code == EXIT_VALIDATION_ERROR
    captured = capsys.readouterr()
    assert "Task description is required" in captured.err


def test_cli_auth_error(capsys):
    with patch(
        "citadel_predict.cli.CitadelClient.predict",
        side_effect=CitadelAuthError("Invalid API key"),
    ):
        code = main(["--task", "test", "--api-key", "invalid-key"])
        assert code == EXIT_AUTH_ERROR
        captured = capsys.readouterr()
        assert "Authentication Error" in captured.err


def test_cli_rate_limit_error(capsys):
    with patch(
        "citadel_predict.cli.CitadelClient.predict",
        side_effect=CitadelRateLimitError("Rate limit reached", retry_after=30),
    ):
        code = main(["--task", "test"])
        assert code == EXIT_RATE_LIMIT_ERROR
        captured = capsys.readouterr()
        assert "Rate Limit Error" in captured.err
        assert "Retry after 30s" in captured.err


def test_cli_validation_error(capsys):
    with patch(
        "citadel_predict.cli.CitadelClient.predict",
        side_effect=CitadelValidationError("Empty task text"),
    ):
        code = main(["--task", "   "])
        assert code == EXIT_VALIDATION_ERROR
        captured = capsys.readouterr()
        assert "Validation Error" in captured.err


def test_cli_server_error(capsys):
    with patch(
        "citadel_predict.cli.CitadelClient.predict",
        side_effect=CitadelServerError("Internal server failure"),
    ):
        code = main(["--task", "test"])
        assert code == EXIT_SERVER_ERROR
        captured = capsys.readouterr()
        assert "Server Error" in captured.err


def test_cli_network_error(capsys):
    with patch(
        "citadel_predict.cli.CitadelClient.predict",
        side_effect=CitadelNetworkError("Connection refused"),
    ):
        code = main(["--task", "test"])
        assert code == EXIT_NETWORK_ERROR
        captured = capsys.readouterr()
        assert "Network Error" in captured.err
