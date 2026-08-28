"""
Model Context Protocol (MCP) server for Citadel Predict.

Exposes pre-execution token and cost prediction tools for AI agent workflows
over stdio transport for Claude Desktop and Claude Code.
"""

from typing import Any, Optional

from citadel_predict import (
    CitadelAuthError,
    CitadelBadRequestError,
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
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]


def create_server() -> MCPServer:
    """
    Factory function to instantiate and configure the Citadel Predict MCP server.
    """
    app = MCPServer(
        name="citadel-predict",
        instructions=(
            "Citadel Predict MCP Server provides pre-execution token budget "
            "and cost estimation for AI agent tasks before running them."
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
        tools: Optional[list[str]] = None,
        num_tools: Optional[int] = None,
        model_id: str = "claude-sonnet",
    ) -> dict[str, Any]:
        """
        Estimate token usage and cost bounds for an AI agent task.

        Args:
            task_text: Natural language task description (1-4000 characters).
            tools: Optional list of tool names available to the agent (e.g. ['web_search', 'draft_doc']).
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
            return {
                "success": True,
                "model_id": result.get("model_id", model_id),
                "expected_tokens": result.get("expected_tokens"),
                "low_tokens": result.get("low_tokens"),
                "high_tokens": result.get("high_tokens"),
                "out_of_distribution": result.get("out_of_distribution", False),
                "ood_reasons": result.get("ood_reasons", []),
                "confidence": result.get("confidence", "normal"),
                "driving_factors": result.get("driving_factors", []),
                "summary": (
                    f"Expected: {result.get('expected_tokens', 0):,} tokens "
                    f"(Range: {result.get('low_tokens', 0):,} – {result.get('high_tokens', 0):,})"
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
        except Exception as exc:
            return {
                "success": False,
                "error_type": "UnexpectedError",
                "message": f"Unexpected error during cost estimation: {exc}",
                "details": str(exc),
            }

    return app


server = create_server()


def main() -> None:
    """Entrypoint for the citadel-predict-mcp CLI command (runs over stdio transport)."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
