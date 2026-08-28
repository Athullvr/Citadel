"""
Unit tests for the Citadel Predict MCP server.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from citadel_predict.errors import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
)

from citadel_predict_mcp.server import create_server, main


@pytest.fixture
def mcp_server():
    return create_server()


def test_tool_registration(mcp_server):
    async def _test():
        tools = await mcp_server.list_tools()
        tool_names = [t.name for t in tools]
        assert "estimate_agent_cost" in tool_names

        tool_def = next(t for t in tools if t.name == "estimate_agent_cost")
        assert "BEFORE running it" in tool_def.description
        assert "token cost and usage range" in tool_def.description

    asyncio.run(_test())


def test_estimate_agent_cost_success(mcp_server):
    async def _test():
        mock_prediction = {
            "model_id": "claude-sonnet",
            "expected_tokens": 3200,
            "low_tokens": 1500,
            "high_tokens": 5800,
            "out_of_distribution": False,
            "ood_reasons": [],
            "confidence": "high",
            "driving_factors": ["task_complexity", "tools_count"],
        }

        with patch("citadel_predict_mcp.server.predict_cost", return_value=mock_prediction) as mock_predict:
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {
                    "task_text": "Audit repository and draft report",
                    "tools": ["list_files", "draft_document"],
                    "num_tools": 2,
                    "model_id": "claude-sonnet",
                },
            )

            assert not result.is_error
            content_text = result.content[0].text
            data = json.loads(content_text)

            assert data["success"] is True
            assert data["expected_tokens"] == 3200
            assert data["low_tokens"] == 1500
            assert data["high_tokens"] == 5800
            assert data["out_of_distribution"] is False
            assert "Expected: 3,200 tokens" in data["summary"]

            mock_predict.assert_called_once_with(
                task_text="Audit repository and draft report",
                tools=["list_files", "draft_document"],
                num_tools=2,
                model_id="claude-sonnet",
            )

    asyncio.run(_test())


def test_auth_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelAuthError("Invalid API key"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Analyze data"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "AuthenticationError"
            assert "CITADEL_API_KEY" in data["message"]

    asyncio.run(_test())


def test_rate_limit_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelRateLimitError("Rate limited", retry_after=45),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Analyze data"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "RateLimitError"
            assert data["retry_after"] == 45
            assert "retry after 45s" in data["message"]

    asyncio.run(_test())


def test_validation_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelValidationError("task_text exceeds maximum length"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "A" * 5000},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "ValidationError"
            assert "Validation failed" in data["message"]

    asyncio.run(_test())


def test_bad_request_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelBadRequestError("Unsupported model id"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Task", "model_id": "unsupported-model"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "BadRequestError"
            assert "Unsupported model id" in data["message"]

    asyncio.run(_test())


def test_server_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelServerError("Internal database failure", status_code=500),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Task"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "ServerError"
            assert "server error (500)" in data["message"]

    asyncio.run(_test())


def test_network_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelNetworkError("ConnectTimeout"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Task"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "NetworkError"
            assert "Unable to reach the Citadel Predict API" in data["message"]

    asyncio.run(_test())


def test_generic_citadel_error_translation(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=CitadelError("Generic client failure"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Task"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "CitadelError"
            assert "Generic client failure" in data["message"]

    asyncio.run(_test())


def test_unexpected_exception_handling(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.predict_cost",
            side_effect=RuntimeError("Unexpected OS crash"),
        ):
            result = await mcp_server.call_tool(
                "estimate_agent_cost",
                {"task_text": "Task"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "UnexpectedError"
            assert "Unexpected OS crash" in data["details"]

    asyncio.run(_test())


def test_main_entrypoint():
    with patch("sys.argv", ["citadel-predict-mcp"]), patch("citadel_predict_mcp.server.server.run") as mock_run:
        main()
        mock_run.assert_called_once_with(transport="stdio")


def test_main_help(capsys):
    with patch("sys.argv", ["citadel-predict-mcp", "--help"]):
        main()
        captured = capsys.readouterr()
        assert "Citadel Predict MCP Server" in captured.out
        assert "claude_desktop_config.json" in captured.out


def test_main_version(capsys):
    with patch("sys.argv", ["citadel-predict-mcp", "--version"]):
        main()
        captured = capsys.readouterr()
        assert "citadel-predict-mcp v0.1.0" in captured.out

