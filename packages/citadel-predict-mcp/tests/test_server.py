"""
Unit tests for the Citadel Predict MCP server.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

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


async def _invoke_registered_handler(server, name, arguments):
    """Exercise a registered sync handler without the sandbox's broken thread pool."""
    tool = server._tool_manager.get_tool(name)
    validated = tool.fn_metadata.validate_arguments(arguments)
    result = tool.fn(**validated)
    return SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(result))],
        is_error=not result.get("success", True),
    )


def test_tool_registration(mcp_server):
    async def _test():
        tools = await mcp_server.list_tools()
        tool_names = [t.name for t in tools]
        assert "estimate_agent_cost" in tool_names
        assert {
            "begin_execution",
            "record_execution_usage",
            "complete_execution",
            "get_execution_summary",
        } <= set(tool_names)

        tool_def = next(t for t in tools if t.name == "estimate_agent_cost")
        assert "BEFORE running it" in tool_def.description
        assert "token cost and usage range" in tool_def.description

        schemas = {
            tool.name: getattr(tool, "input_schema", None) or tool.inputSchema for tool in tools
        }
        begin_required = schemas["begin_execution"].get("required", [])
        assert "exact_provider_model" not in begin_required
        assert "exact_provider_model" in schemas["begin_execution"]["properties"]
        assert {"task_text", "prediction_id"} <= set(
            schemas["begin_execution"]["properties"]
        )
        assert "use_adaptive_profile" in schemas["begin_execution"]["properties"]
        assert "use_adaptive_profile" not in begin_required
        assert "workspace_id" not in schemas["begin_execution"]["properties"]
        assert {"execution_id", "event_id"} <= set(
            schemas["record_execution_usage"]["required"]
        )
        usage_required = schemas["record_execution_usage"].get("required", [])
        assert "exact_provider_model" not in usage_required
        assert "input_tokens" not in usage_required
        assert "provider_counters" in schemas["record_execution_usage"]["properties"]
        assert "reasoning_effort" in schemas["begin_execution"]["properties"]
        assert "source" not in schemas["record_execution_usage"]["properties"]
        assert schemas["complete_execution"]["properties"]["status"]["enum"] == [
            "completed",
            "failed",
            "cancelled",
        ]
        assert schemas["get_execution_summary"]["required"] == ["execution_id"]

    asyncio.run(_test())


class _FakeExecutionClient:
    detail: ClassVar[dict] = {
        "execution": {
            "execution_id": "2c1071c4-2eec-4eac-b0ea-286c341c885f",
            "workspace_id": "must-not-leak",
            "status": "started",
            "integration_name": "mcp-agent",
            "integration_version": "1.0",
            "provider": "anthropic",
            "exact_provider_model": "test-anthropic-model-v1",
            "started_at": "2026-11-05T00:00:00Z",
            "completed_at": None,
            "telemetry_provenance": "unverified_submitted",
            "external_run_id": "private-run-id",
        },
        "prediction": {
            "prediction_id": "a1cb1c59-3a89-49fa-9e6c-9d371f0802ed",
            "model_alias": "claude-sonnet",
            "predicted_low_tokens": 10,
            "predicted_expected_tokens": 20,
            "predicted_high_tokens": 30,
            "estimated_low_cost": None,
            "estimated_expected_cost": None,
            "estimated_high_cost": None,
            "created_at": "2026-11-05T00:00:00Z",
            "task_hash": "must-not-leak",
            "feature_snapshot": {"secret": True},
        },
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_creation_input_tokens_known_sum": None,
            "cache_creation_missing_event_count": 0,
            "cache_read_input_tokens": None,
            "cache_read_input_tokens_known_sum": None,
            "cache_read_missing_event_count": 0,
            "total_input_tokens": None,
            "total_tokens": None,
        },
        "usage_event_count": 0,
        "usage_completeness": "unknown",
    }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def create_execution_prediction(self, **_kwargs):
        return {"prediction_id": self.detail["prediction"]["prediction_id"]}

    def create_execution(self, **_kwargs):
        return {"execution_id": self.detail["execution"]["execution_id"]}

    def begin_execution(self, **_kwargs):
        if _kwargs.get("task_text") is not None:
            self.create_execution_prediction()
        return self.detail

    def get_execution_summary(self, _execution_id):
        return self.detail

    def record_execution_usage(self, _execution_id, **kwargs):
        return {"event_id": kwargs["event_id"], "recorded": True}

    def complete_execution(self, _execution_id, *, status, **_kwargs):
        return {
            **self.detail,
            "execution": {**self.detail["execution"], "status": status},
            "usage_completeness": "complete",
        }


def _tool_json(result):
    return json.loads(result.content[0].text)


def test_begin_execution_creates_prediction_and_returns_safe_summary(mcp_server):
    async def _test():
        with patch("citadel_predict_mcp.server.CitadelClient", _FakeExecutionClient):
            result = await _invoke_registered_handler(
                mcp_server,
                "begin_execution",
                {
                    "task_text": "Do a private task",
                    "exact_provider_model": "test-anthropic-model-v1",
                    "integration_name": "mcp-agent",
                },
            )
        data = _tool_json(result)
        assert data["success"] is True
        assert data["execution_id"] == _FakeExecutionClient.detail["execution"]["execution_id"]
        assert data["prediction"]["predicted_expected_tokens"] == 20
        assert "workspace_id" not in json.dumps(data)
        assert "task_hash" not in json.dumps(data)
        assert "feature_snapshot" not in json.dumps(data)
        assert "external_run_id" not in json.dumps(data)

    asyncio.run(_test())


def test_legacy_estimate_handler_preserves_its_response(mcp_server):
    async def _test():
        prediction = {
            "model_id": "claude-sonnet",
            "expected_tokens": 3200,
            "low_tokens": 1500,
            "high_tokens": 5800,
            "expected_cost_usd": 0.02,
            "low_cost_usd": 0.01,
            "high_cost_usd": 0.04,
            "driving_factors": [],
        }
        with patch("citadel_predict_mcp.server.predict_cost", return_value=prediction):
            result = await _invoke_registered_handler(
                mcp_server,
                "estimate_agent_cost",
                {"task_text": "A legacy prediction request"},
            )
        data = _tool_json(result)
        assert data["success"] is True
        assert data["expected_tokens"] == 3200
        assert data["model_id"] == "claude-sonnet"

    asyncio.run(_test())


def test_begin_execution_can_use_existing_prediction_reference(mcp_server):
    class ExistingPredictionClient(_FakeExecutionClient):
        def create_execution_prediction(self, **_kwargs):
            raise AssertionError("an existing prediction must not be recreated")

    async def _test():
        with patch("citadel_predict_mcp.server.CitadelClient", ExistingPredictionClient):
            result = await _invoke_registered_handler(
                mcp_server,
                "begin_execution",
                {
                    "prediction_id": _FakeExecutionClient.detail["prediction"]["prediction_id"],
                    "exact_provider_model": "test-anthropic-model-v1",
                },
            )
        data = _tool_json(result)
        assert data["success"] is True
        assert data["prediction"]["prediction_id"] == _FakeExecutionClient.detail[
            "prediction"
        ]["prediction_id"]

    asyncio.run(_test())


def test_begin_execution_can_return_safe_adaptive_profile_summary(mcp_server):
    class AdaptiveExecutionClient(_FakeExecutionClient):
        def begin_execution(self, **kwargs):
            assert kwargs["use_adaptive_profile"] is True
            return {
                **self.detail,
                "adaptive_prediction": {
                    "original_prediction": {"expected_tokens": 100},
                    "adjusted_prediction": {"expected_tokens": 120},
                    "profile": {"status": "active", "adjustment_tokens": 20},
                },
            }

    async def _test():
        with patch("citadel_predict_mcp.server.CitadelClient", AdaptiveExecutionClient):
            result = await _invoke_registered_handler(
                mcp_server,
                "begin_execution",
                {
                    "task_text": "Review a public release",
                    "exact_provider_model": "test-anthropic-model-v1",
                    "use_adaptive_profile": True,
                },
            )
        data = _tool_json(result)
        assert data["success"] is True
        assert data["adaptive_prediction"]["profile"]["status"] == "active"
        assert data["adaptive_prediction"]["adjusted_prediction"]["expected_tokens"] == 120
        assert "workspace_id" not in json.dumps(data)
        assert '"task_hash":' not in json.dumps(data)

    asyncio.run(_test())


def test_record_usage_keeps_cache_null_and_reports_unverified_source(mcp_server):
    async def _test():
        event_id = "c115e9ac-d80b-4e31-b497-96cc1a85a8b0"
        with patch("citadel_predict_mcp.server.CitadelClient", _FakeExecutionClient):
            result = await _invoke_registered_handler(
                mcp_server,
                "record_execution_usage",
                {
                    "execution_id": _FakeExecutionClient.detail["execution"]["execution_id"],
                    "event_id": event_id,
                    "exact_provider_model": "test-anthropic-model-v1",
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": 0,
                },
            )
        data = _tool_json(result)
        assert data == {
            "success": True,
            "event_id": event_id,
            "recorded": True,
            "telemetry_provenance": "unverified_submitted",
        }

    asyncio.run(_test())


def test_execution_tools_validate_provider_counters_and_completion(mcp_server):
    async def _test():
        invalid = await _invoke_registered_handler(
            mcp_server,
            "record_execution_usage",
            {
                "execution_id": _FakeExecutionClient.detail["execution"]["execution_id"],
                "event_id": "c115e9ac-d80b-4e31-b497-96cc1a85a8b0",
                "exact_provider_model": "test-anthropic-model-v1",
                "input_tokens": -1,
                "output_tokens": 3,
            },
        )
        with patch("citadel_predict_mcp.server.CitadelClient", _FakeExecutionClient):
            provider_specific = await _invoke_registered_handler(
                mcp_server,
                "record_execution_usage",
                {
                    "execution_id": _FakeExecutionClient.detail["execution"]["execution_id"],
                    "event_id": "b115e9ac-d80b-4e31-b497-96cc1a85a8b0",
                    "exact_provider_model": "gpt-6-luna",
                    "provider": "openai",
                    "provider_counters": {
                        "input_tokens": 1,
                        "input_tokens_details.cached_tokens": 0,
                        "output_tokens": 3,
                        "output_tokens_details.reasoning_tokens": 0,
                        "total_tokens": 4,
                    },
                },
            )
            completed = await _invoke_registered_handler(
                mcp_server,
                "complete_execution",
                {
                    "execution_id": _FakeExecutionClient.detail["execution"]["execution_id"],
                    "status": "cancelled",
                    "expected_usage_event_count": 0,
                },
            )
        assert _tool_json(invalid)["error_type"] == "ValidationError"
        assert _tool_json(provider_specific)["success"] is True
        assert _tool_json(completed)["status"] == "cancelled"
        assert _tool_json(completed)["usage_completeness"] == "complete"

    asyncio.run(_test())


def test_get_execution_summary_maps_not_found_without_backend_details(mcp_server):
    async def _test():
        with patch(
            "citadel_predict_mcp.server.CitadelClient",
            side_effect=CitadelError("database password leaked", status_code=404),
        ):
            result = await _invoke_registered_handler(
                mcp_server,
                "get_execution_summary",
                {"execution_id": _FakeExecutionClient.detail["execution"]["execution_id"]},
            )
        data = _tool_json(result)
        assert data["error_type"] == "NotFoundError"
        assert "database password" not in json.dumps(data)

    asyncio.run(_test())


def test_get_execution_summary_returns_safe_detail(mcp_server):
    async def _test():
        with patch("citadel_predict_mcp.server.CitadelClient", _FakeExecutionClient):
            result = await _invoke_registered_handler(
                mcp_server,
                "get_execution_summary",
                {"execution_id": _FakeExecutionClient.detail["execution"]["execution_id"]},
            )
        data = _tool_json(result)
        assert data["success"] is True
        assert data["usage_completeness"] == "unknown"
        assert data["telemetry_provenance"] == "unverified_submitted"
        assert data["usage"]["input_tokens"] is None

    asyncio.run(_test())


def test_estimate_agent_cost_success(mcp_server):
    async def _test():
        mock_prediction = {
            "model_id": "claude-sonnet",
            "expected_tokens": 3200,
            "low_tokens": 1500,
            "high_tokens": 5800,
            "expected_cost_usd": 0.0192,
            "low_cost_usd": 0.009,
            "high_cost_usd": 0.0348,
            "cost_note": "Cost estimated using Anthropic published rates",
            "out_of_distribution": False,
            "ood_reasons": [],
            "confidence": "high",
            "driving_factors": ["task_complexity", "tools_count"],
        }

        with patch(
            "citadel_predict_mcp.server.predict_cost",
            return_value=mock_prediction,
        ) as mock_predict:
            result = await _invoke_registered_handler(
                mcp_server,
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
            assert data["expected_cost_usd"] == 0.0192
            assert data["low_cost_usd"] == 0.009
            assert data["high_cost_usd"] == 0.0348
            assert data["out_of_distribution"] is False
            assert "Expected: 3,200 tokens (~$0.02)" in data["summary"]
            assert "(Range: 1,500 \u2013 5,800 tokens (~$0.01 \u2013 $0.03))" in data["summary"]

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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
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
            result = await _invoke_registered_handler(
                mcp_server,
                "estimate_agent_cost",
                {"task_text": "Task"},
            )
            data = json.loads(result.content[0].text)
            assert data["success"] is False
            assert data["error_type"] == "UnexpectedError"
            assert "Unexpected OS crash" in data["details"]

    asyncio.run(_test())


def test_main_entrypoint():
    with (
        patch("sys.argv", ["citadel-predict-mcp"]),
        patch("citadel_predict_mcp.server.server.run") as mock_run,
    ):
        main()
        mock_run.assert_called_once_with(transport="stdio")


def test_main_help(capsys):
    with patch("sys.argv", ["citadel-predict-mcp", "--help"]):
        main()
        captured = capsys.readouterr()
        assert "Citadel Predict MCP Server" in captured.out
        assert "claude_desktop_config.json" in captured.out


def test_main_version(capsys):
    from citadel_predict_mcp import __version__

    with patch("sys.argv", ["citadel-predict-mcp", "--version"]):
        main()
        captured = capsys.readouterr()
        assert f"citadel-predict-mcp v{__version__}" in captured.out

