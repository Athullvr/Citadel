"""
Model Context Protocol (MCP) server for Citadel Predict.

Exposes pre-execution token and cost prediction tools for AI agent workflows
over stdio transport for Claude Desktop and Claude Code.
"""

import sys
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from citadel_predict import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelClient,
    CitadelError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
    predict_cost,
)

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[attr-defined,no-redef]


_MAX_BIGINT = 2**63 - 1


def _safe_execution_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Return only execution fields that are safe for an MCP client to receive."""
    execution = data.get("execution") or {}
    prediction = data.get("prediction")
    usage = data.get("usage") or {}
    safe_prediction = None
    if prediction:
        safe_prediction = {
            key: prediction.get(key)
            for key in (
                "prediction_id",
                "model_alias",
                "provider",
                "exact_provider_model",
                "identity_schema_version",
                "identity_status",
                "token_target_semantics",
                "model_artifact_version",
                "estimator_artifact_version",
                "estimator_artifact_sha256",
                "prediction_identity_hash",
                "predicted_low_tokens",
                "predicted_expected_tokens",
                "predicted_high_tokens",
                "estimated_low_cost",
                "estimated_expected_cost",
                "estimated_high_cost",
                "created_at",
            )
        }
    result = {
        "success": True,
        "execution_id": execution.get("execution_id"),
        "status": execution.get("status"),
        "integration_name": execution.get("integration_name"),
        "integration_version": execution.get("integration_version"),
        "provider": execution.get("provider"),
        "exact_provider_model": execution.get("exact_provider_model"),
        "identity_schema_version": execution.get("identity_schema_version"),
        "identity_status": execution.get("identity_status"),
        "token_target_semantics": execution.get("token_target_semantics"),
        "prediction_identity_hash": execution.get("prediction_identity_hash"),
        "reasoning_effort": execution.get("reasoning_effort"),
        "started_at": execution.get("started_at"),
        "completed_at": execution.get("completed_at"),
        "usage_completeness": data.get("usage_completeness", "unknown"),
        "usage_event_count": data.get("usage_event_count", 0),
        "telemetry_provenance": execution.get("telemetry_provenance") or "unknown",
        "prediction": safe_prediction,
        "usage": {
            key: usage.get(key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_creation_input_tokens_known_sum",
                "cache_creation_missing_event_count",
                "cache_read_input_tokens",
                "cache_read_input_tokens_known_sum",
                "cache_read_missing_event_count",
                "cached_input_tokens",
                "cached_input_tokens_known_sum",
                "cached_input_tokens_missing_event_count",
                "reasoning_tokens",
                "reasoning_tokens_known_sum",
                "reasoning_tokens_missing_event_count",
                "tool_use_prompt_tokens",
                "tool_use_prompt_tokens_known_sum",
                "tool_use_prompt_tokens_missing_event_count",
                "provider_reported_total_tokens",
                "provider_reported_total_tokens_known_sum",
                "provider_reported_total_tokens_missing_event_count",
                "incompatible_usage_event_count",
                "token_semantics",
                "provider_usage_completeness",
                "total_input_tokens",
                "total_tokens",
            )
        },
    }
    adaptive_prediction = data.get("adaptive_prediction")
    if isinstance(adaptive_prediction, dict):
        result["adaptive_prediction"] = {
            "original_prediction": adaptive_prediction.get("original_prediction"),
            "adjusted_prediction": adaptive_prediction.get("adjusted_prediction"),
            "profile": adaptive_prediction.get("profile"),
        }
    return result


def _execution_error(operation: str, exc: Exception) -> dict[str, Any]:
    """Map failures to stable MCP errors without exposing backend details."""
    if isinstance(exc, CitadelAuthError):
        error_type, message = "AuthenticationError", "Citadel API authentication failed."
    elif isinstance(exc, CitadelRateLimitError):
        error_type, message = "RateLimitError", "Citadel API rate limit exceeded."
    elif isinstance(exc, CitadelNetworkError):
        error_type, message = "NetworkError", "Unable to reach the Citadel API."
    elif isinstance(exc, CitadelValidationError):
        error_type, message = "ValidationError", "Citadel rejected the request parameters."
    elif isinstance(exc, CitadelBadRequestError):
        error_type, message = "BadRequestError", "Citadel rejected the request."
    elif isinstance(exc, CitadelError) and exc.status_code == 404:
        error_type, message = "NotFoundError", "The execution or prediction was not found."
    elif isinstance(exc, CitadelError) and exc.status_code == 409:
        error_type, message = "ConflictError", "The requested state conflicts with the execution."
    elif isinstance(exc, CitadelServerError):
        error_type, message = "ServerError", "Citadel could not complete the request."
    elif isinstance(exc, CitadelError):
        error_type, message = "CitadelError", "Citadel could not complete the request."
    else:
        error_type, message = "UnexpectedError", "The MCP execution request could not be completed."
    result: dict[str, Any] = {
        "success": False,
        "error_type": error_type,
        "message": f"{operation}: {message}",
    }
    if isinstance(exc, CitadelRateLimitError) and exc.retry_after is not None:
        result["retry_after"] = exc.retry_after
    return result


def _validate_counter(name: str, value: int | None, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is not int or value < 0 or value > _MAX_BIGINT:
        raise ValueError(f"{name} must be a non-negative integer within the BIGINT range.")


def create_server() -> MCPServer:
    """
    Factory function to instantiate and configure the Citadel Predict MCP server.
    """
    app = MCPServer(
        name="citadel-predict",
        instructions=(
            "Citadel Predict MCP Server provides pre-execution token budget "
            "and cost estimation, plus explicit provider-aware execution lifecycle tracking. "
            "Submitted usage is unverified unless received through a separately authenticated "
            "provider integration."
        ),
    )

    @app.tool(
        name="estimate_agent_cost",
        description=(
            "Estimate token cost and usage range for an AI agent task BEFORE running it. "
            "Use this when the user is about to execute, dispatch, or run a multi-step agent "
            "task and cost/budget matters."
        ),
    )
    def estimate_agent_cost(
        task_text: str,
        tools: list[str] | None = None,
        num_tools: int | None = None,
        model_id: str = "claude-sonnet",
    ) -> dict[str, Any]:
        """
        Estimate token usage and cost bounds for an AI agent task.

        Args:
            task_text: Natural language task description (1-4000 characters).
            tools: Optional tool names available to the agent, such as
                ['web_search', 'draft_doc'].
            num_tools: Optional tool count if tool names are unspecified.
            model_id: Model calibration identifier (default: 'claude-sonnet').

        Returns:
            Dictionary containing token estimates (expected_tokens, low_tokens, high_tokens),
            out-of-distribution flags, and diagnostic reasons.
        """
        try:
            result = predict_cost(
                task_text=task_text,
                tools=tools,
                num_tools=num_tools,
                model_id=model_id,
            )
            expected_cost = result.get("expected_cost_usd")
            low_cost = result.get("low_cost_usd")
            high_cost = result.get("high_cost_usd")

            cost_str = f" (~${expected_cost:.2f})" if expected_cost is not None else ""
            cost_range_str = (
                f" (~${low_cost:.2f} \u2013 ${high_cost:.2f})"
                if low_cost is not None and high_cost is not None
                else ""
            )

            return {
                "success": True,
                "model_id": result.get("model_id", model_id),
                "expected_tokens": result.get("expected_tokens"),
                "low_tokens": result.get("low_tokens"),
                "high_tokens": result.get("high_tokens"),
                "expected_cost_usd": expected_cost,
                "low_cost_usd": low_cost,
                "high_cost_usd": high_cost,
                "cost_note": result.get("cost_note"),
                "out_of_distribution": result.get("out_of_distribution", False),
                "ood_reasons": result.get("ood_reasons", []),
                "confidence": result.get("confidence", "normal"),
                "driving_factors": result.get("driving_factors", []),
                "summary": (
                    f"Expected: {result.get('expected_tokens', 0):,} tokens{cost_str} "
                    f"(Range: {result.get('low_tokens', 0):,} \u2013 "
                    f"{result.get('high_tokens', 0):,} tokens{cost_range_str})"
                ),
            }
        except CitadelAuthError as exc:
            return {
                "success": False,
                "error_type": "AuthenticationError",
                "message": (
                    "Authentication failed: Missing or invalid Citadel API key. "
                    "Ensure CITADEL_API_KEY environment variable is configured in your "
                    "Claude Desktop or Claude Code configuration, or in ~/.citadel/config.toml."
                ),
                "details": str(exc),
            }
        except CitadelRateLimitError as exc:
            retry_note = f" (retry after {exc.retry_after}s)" if exc.retry_after else ""
            return {
                "success": False,
                "error_type": "RateLimitError",
                "message": (
                    f"Citadel Predict API rate limit exceeded{retry_note}. "
                    "Please wait a moment before sending more prediction requests."
                ),
                "retry_after": exc.retry_after,
            }
        except CitadelValidationError as exc:
            return {
                "success": False,
                "error_type": "ValidationError",
                "message": (
                    f"Validation failed for task or tool parameters: {exc.message}. "
                    "Ensure task_text is between 1 and 4000 characters."
                ),
                "details": exc.message,
            }
        except CitadelBadRequestError as exc:
            return {
                "success": False,
                "error_type": "BadRequestError",
                "message": f"Bad request rejected by Citadel Predict API: {exc.message}",
                "details": exc.message,
            }
        except CitadelServerError as exc:
            return {
                "success": False,
                "error_type": "ServerError",
                "message": (
                    f"Citadel Predict server error ({exc.status_code}): {exc.message}. "
                    "Please try again shortly."
                ),
                "details": exc.message,
            }
        except CitadelNetworkError as exc:
            return {
                "success": False,
                "error_type": "NetworkError",
                "message": (
                    "Unable to reach the Citadel Predict API. "
                    "Please verify your internet connection or API endpoint status."
                ),
                "details": str(exc),
            }
        except CitadelError as exc:
            return {
                "success": False,
                "error_type": "CitadelError",
                "message": f"Citadel Predict request error: {exc.message}",
                "details": str(exc),
            }
        # Preserve the MCP tool's structured error contract for any unclassified failure.
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error_type": "UnexpectedError",
                "message": f"Unexpected error during cost estimation: {exc}",
                "details": str(exc),
            }

    @app.tool(
        name="begin_execution",
        description=(
            "Create or attach a compatible prediction and register a provider execution "
            "before it runs. "
            "Optionally apply an eligible workspace execution profile when creating a new "
            "prediction. Returns an execution ID for usage recording and completion."
        ),
    )
    def begin_execution(
        exact_provider_model: str | None = None,
        task_text: str | None = None,
        prediction_id: str | None = None,
        tools: list[str] | None = None,
        model_id: str = "claude-sonnet",
        provider: str = "anthropic",
        integration_name: str = "citadel-predict-mcp",
        integration_version: str | None = None,
        use_adaptive_profile: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Create a prediction if needed and register the execution."""
        try:
            if (task_text is None) == (prediction_id is None):
                raise ValueError("Provide exactly one of task_text or prediction_id.")
            if exact_provider_model is not None and not exact_provider_model.strip():
                raise ValueError("exact_provider_model must not be blank when supplied.")
            if prediction_id is not None:
                UUID(prediction_id)
            if task_text is not None and not task_text.strip():
                raise ValueError("task_text must not be empty.")

            with CitadelClient() as client:
                detail = client.begin_execution(
                    prediction_id=prediction_id,
                    task_text=task_text,
                    tools=tools,
                    model_id=model_id,
                    integration_name=integration_name,
                    integration_version=integration_version,
                    provider=provider,
                    exact_provider_model=exact_provider_model,
                    use_adaptive_profile=use_adaptive_profile,
                    reasoning_effort=reasoning_effort,
                )
            return _safe_execution_summary(detail)
        except (CitadelError, ValueError, KeyError) as exc:
            if isinstance(exc, CitadelError):
                return _execution_error("Unable to begin execution", exc)
            return {"success": False, "error_type": "ValidationError", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return _execution_error("Unable to begin execution", exc)

    @app.tool(
        name="record_execution_usage",
        description=(
            "Append one idempotent provider usage event to a started execution. "
            "Supply the same event_id when retrying the same event."
        ),
    )
    def record_execution_usage(
        execution_id: str,
        event_id: str,
        exact_provider_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        provider: str = "anthropic",
        occurred_at: str | None = None,
        provider_counters: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        """Record provider counters as unverified submitted telemetry."""
        try:
            UUID(execution_id)
            UUID(event_id)
            _validate_counter("input_tokens", input_tokens, nullable=True)
            _validate_counter("output_tokens", output_tokens, nullable=True)
            _validate_counter(
                "cache_creation_input_tokens", cache_creation_input_tokens, nullable=True
            )
            _validate_counter("cache_read_input_tokens", cache_read_input_tokens, nullable=True)
            if provider_counters is not None:
                if not isinstance(provider_counters, dict) or not provider_counters:
                    raise ValueError("provider_counters must be a non-empty counter map.")
                if len(provider_counters) > 32:
                    raise ValueError("provider_counters may contain at most 32 counters.")
                for name, value in provider_counters.items():
                    if not isinstance(name, str) or not name.strip() or len(name) > 96:
                        raise ValueError(
                            "Provider counter names must be non-empty and at most 96 characters."
                        )
                    _validate_counter(f"provider_counters.{name}", value, nullable=True)
            if exact_provider_model is not None and not exact_provider_model.strip():
                raise ValueError("exact_provider_model must not be blank when supplied.")
            if occurred_at is not None:
                parsed = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError("occurred_at must include a timezone.")
            with CitadelClient() as client:
                result = client.record_execution_usage(
                    execution_id,
                    event_id=event_id,
                    provider=provider,
                    exact_provider_model=exact_provider_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    occurred_at=occurred_at,
                    provider_counters=provider_counters,
                )
            return {
                "success": True,
                "event_id": result.get("event_id", event_id),
                "recorded": result.get("recorded"),
                "telemetry_provenance": "unverified_submitted",
            }
        except (CitadelError, ValueError) as exc:
            if isinstance(exc, CitadelError):
                return _execution_error("Unable to record usage", exc)
            return {"success": False, "error_type": "ValidationError", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return _execution_error("Unable to record usage", exc)

    @app.tool(
        name="complete_execution",
        description=(
            "Mark a started execution completed, failed, or cancelled. "
            "An omitted expected event count leaves telemetry completeness unknown."
        ),
    )
    def complete_execution(
        execution_id: str,
        status: Literal["completed", "failed", "cancelled"],
        expected_usage_event_count: int | None = None,
    ) -> dict[str, Any]:
        """Transition an execution exactly once into a terminal state."""
        try:
            UUID(execution_id)
            if status not in {"completed", "failed", "cancelled"}:
                raise ValueError("status must be completed, failed, or cancelled.")
            _validate_counter(
                "expected_usage_event_count", expected_usage_event_count, nullable=True
            )
            with CitadelClient() as client:
                detail = client.complete_execution(
                    execution_id,
                    status=status,
                    expected_usage_event_count=expected_usage_event_count,
                )
            return _safe_execution_summary(detail)
        except (CitadelError, ValueError) as exc:
            if isinstance(exc, CitadelError):
                return _execution_error("Unable to complete execution", exc)
            return {"success": False, "error_type": "ValidationError", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return _execution_error("Unable to complete execution", exc)

    @app.tool(
        name="get_execution_summary",
        description="Read a safe summary of an execution, prediction, usage, and completeness.",
    )
    def get_execution_summary(execution_id: str) -> dict[str, Any]:
        """Read a workspace-scoped execution summary without prompts or feature snapshots."""
        try:
            UUID(execution_id)
            with CitadelClient() as client:
                detail = client.get_execution_summary(execution_id)
            return _safe_execution_summary(detail)
        except (CitadelError, ValueError) as exc:
            if isinstance(exc, CitadelError):
                return _execution_error("Unable to read execution summary", exc)
            return {"success": False, "error_type": "ValidationError", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return _execution_error("Unable to read execution summary", exc)

    return app


server = create_server()


def main() -> None:
    """Entrypoint for the citadel-predict-mcp CLI command (runs over stdio transport)."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help", "-v", "--version"):
        if sys.argv[1] in ("-v", "--version"):
            from . import __version__

            print(f"citadel-predict-mcp v{__version__}")
            return

        print("Citadel Predict MCP Server (stdio transport)")
        print("\nUsage:")
        print(
            "  citadel-predict-mcp            Run as an MCP stdio server "
            "(used by Claude Desktop/Code)"
        )
        print("  citadel-predict-mcp --help     Show this help message")
        print("  citadel-predict-mcp --version  Show version")
        print("\nConfiguration for Claude Desktop (claude_desktop_config.json):")
        print("  {")
        print('    "mcpServers": {')
        print('      "citadel-predict": {')
        print('        "command": "citadel-predict-mcp",')
        print('        "env": { "CITADEL_API_KEY": "cp_live_..." }')
        print("      }")
        print("    }")
        print("  }")
        return

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
