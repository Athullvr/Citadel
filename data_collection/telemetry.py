"""Opt-in reporting client for the Citadel execution feedback API."""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

logger = logging.getLogger("citadel.runner.telemetry")


@dataclass(frozen=True)
class TelemetryExecution:
    prediction_id: UUID
    execution_id: UUID


class CitadelTelemetry:
    """Best-effort Citadel reporting for an explicitly enabled runner."""

    def __init__(self, api_url: str, api_key: str, timeout: float = 10.0) -> None:
        self.api_url = api_url.rstrip("/")
        parsed_url = urlparse(self.api_url)
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed_url.scheme != "https" and not (
            parsed_url.scheme == "http" and parsed_url.hostname in local_hosts
        ):
            raise ValueError(
                "Citadel telemetry requires HTTPS; HTTP is allowed only for localhost development."
            )
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "citadel-data-collection-runner/0.3.0",
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(f"{self.api_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()

    def start_execution(self, task: dict[str, Any], model: str) -> TelemetryExecution | None:
        """Create the immutable prediction and execution records best-effort."""
        try:
            prediction = self._post(
                "/api/executions/predictions",
                {
                    "task_text": task["text"],
                    "tools": task.get("tools", []),
                    "model_id": "claude-sonnet",
                },
            )
            prediction_id = UUID(prediction["prediction_id"])
            execution = self._post(
                "/api/executions",
                {
                    "prediction_id": str(prediction_id),
                    "integration_name": "citadel-data-collection-agent-runner",
                    "integration_version": "0.3.0",
                    "provider": "anthropic",
                    "exact_provider_model": model,
                    "is_synthetic": True,
                },
            )
            return TelemetryExecution(
                prediction_id=prediction_id,
                execution_id=UUID(execution["execution_id"]),
            )
        except (httpx.HTTPError, KeyError, ValueError):
            logger.warning(
                "Could not start Citadel telemetry; the agent run will continue without it."
            )
            return None

    @staticmethod
    def _event_id(execution_id: UUID, provider_response_id: str) -> UUID:
        """Derive a stable retry key from the execution and Anthropic response ID."""
        return uuid5(NAMESPACE_URL, f"citadel:{execution_id}:anthropic:{provider_response_id}")

    def record_response(self, execution: TelemetryExecution, response: Any) -> None:
        """Report one Anthropic Messages response, retrying with its stable event ID."""
        usage = response.usage
        event_id = self._event_id(execution.execution_id, str(response.id))
        payload = {
            "event_id": str(event_id),
            "provider": "anthropic",
            "exact_provider_model": str(response.model),
            "input_tokens": int(usage.input_tokens),
            "output_tokens": int(usage.output_tokens),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "source": "anthropic_messages_api_synthetic_tools",
            "occurred_at": datetime.now(UTC).isoformat(),
        }
        for attempt in range(3):
            try:
                result = self._post(f"/api/executions/{execution.execution_id}/usage", payload)
                if result.get("event_id") != str(event_id):
                    raise ValueError("Citadel returned a different usage event ID")
                return
            except (httpx.HTTPError, ValueError):
                if attempt < 2:
                    time.sleep(0.2 * (2**attempt))
        logger.warning("Could not report one Anthropic usage event; the agent run will continue.")

    def complete_execution(self, execution: TelemetryExecution, succeeded: bool) -> None:
        try:
            self._post(
                f"/api/executions/{execution.execution_id}/complete",
                {
                    "status": "completed" if succeeded else "failed",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        except (httpx.HTTPError, ValueError):
            logger.warning(
                "Could not complete Citadel telemetry; the agent run result is unchanged."
            )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CitadelTelemetry":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
