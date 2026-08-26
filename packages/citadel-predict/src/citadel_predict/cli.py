"""
Command-line interface for Citadel Predict.

Usage:
    citadel-predict --task "Research 3 competitors" --tools web_search,draft_document
"""

import argparse
import json
import sys
from typing import Optional, Sequence

from .client import CitadelClient
from .errors import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
)

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_AUTH_ERROR = 3
EXIT_RATE_LIMIT_ERROR = 4
EXIT_SERVER_ERROR = 5
EXIT_NETWORK_ERROR = 6


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="citadel-predict",
        description="Predict AI agent token budgets and costs before execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task description string for the agent run.",
    )
    parser.add_argument(
        "positional_task",
        nargs="?",
        type=str,
        help="Task description (if --task is not passed).",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="",
        help="Comma-separated list of tool names (e.g. 'web_search,fetch_url,draft_document').",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        default="claude-sonnet",
        help="Model calibration identifier.",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="Citadel API key (overrides CITADEL_API_KEY env var and config file).",
    )
    parser.add_argument(
        "--api-url",
        "-u",
        type=str,
        default=None,
        help="Citadel API base URL (default: http://localhost:8000 or CITADEL_API_URL).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text.",
    )
    return parser.parse_args(args)


def format_pretty_output(result: dict) -> str:
    lines = [
        "=" * 50,
        "  CITADEL PREDICT — TOKEN BUDGET ESTIMATE",
        "=" * 50,
        f"Model:           {result.get('model_id')}",
        f"Expected Tokens: {result.get('expected_tokens', 0):,} tokens",
        f"Predicted Range: {result.get('low_tokens', 0):,} – {result.get('high_tokens', 0):,} tokens",
        f"Confidence:      {result.get('confidence', 'unknown').upper()}",
    ]

    if result.get("out_of_distribution"):
        lines.append(f"OOD Warning:     YES ({', '.join(result.get('ood_reasons', []))})")
    else:
        lines.append("OOD Warning:     No (In-Distribution)")

    factors = result.get("driving_factors", [])
    if factors:
        lines.append("Driving Factors:")
        for factor in factors:
            lines.append(f"  • {factor}")

    lines.append("=" * 50)
    return "\n".join(lines)


def main(args: Optional[Sequence[str]] = None) -> int:
    parsed = parse_args(args)

    task = parsed.task or parsed.positional_task
    if not task:
        sys.stderr.write("Error: Task description is required. Use --task '...' or pass as argument.\n")
        return EXIT_VALIDATION_ERROR

    tools = [t.strip() for t in parsed.tools.split(",") if t.strip()] if parsed.tools else []

    try:
        with CitadelClient(
            api_key=parsed.api_key,
            base_url=parsed.api_url,
            timeout=parsed.timeout,
        ) as client:
            result = client.predict(
                task_text=task,
                tools=tools,
                model_id=parsed.model_id,
            )

        if parsed.json:
            print(json.dumps(result, indent=2))
        else:
            print(format_pretty_output(result))
        return EXIT_SUCCESS

    except CitadelAuthError as exc:
        sys.stderr.write(f"Authentication Error: {exc.message}\n")
        return EXIT_AUTH_ERROR

    except CitadelRateLimitError as exc:
        msg = f"Rate Limit Error: {exc.message}"
        if exc.retry_after is not None:
            msg += f" (Retry after {exc.retry_after}s)"
        sys.stderr.write(f"{msg}\n")
        return EXIT_RATE_LIMIT_ERROR

    except (CitadelValidationError, CitadelBadRequestError) as exc:
        sys.stderr.write(f"Validation Error: {exc.message}\n")
        return EXIT_VALIDATION_ERROR

    except CitadelServerError as exc:
        sys.stderr.write(f"Server Error: {exc.message}\n")
        return EXIT_SERVER_ERROR

    except CitadelNetworkError as exc:
        sys.stderr.write(f"Network Error: {exc.message}\n")
        return EXIT_NETWORK_ERROR

    except CitadelError as exc:
        sys.stderr.write(f"Citadel Error: {exc.message}\n")
        return EXIT_GENERAL_ERROR

    except Exception as exc:
        sys.stderr.write(f"Unexpected Error: {exc}\n")
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
