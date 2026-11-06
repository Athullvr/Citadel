"""
HTTP client for the Citadel Predict API.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .config import resolve_api_key, resolve_api_url
from .errors import (
    CitadelAuthError,
    CitadelBadRequestError,
    CitadelError,
    CitadelNetworkError,
    CitadelRateLimitError,
    CitadelServerError,
    CitadelValidationError,
)


class CitadelClient:
    """
    Client for interacting with the Citadel Predict API.

    Handles authentication, error mapping, timeouts, and request dispatch.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
        config_path: Path | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key, config_path=config_path)
        self.base_url = resolve_api_url(base_url, config_path=config_path)
        self.timeout = timeout
        self._custom_client = http_client is not None
        self._client = http_client or httpx.Client(timeout=self.timeout)

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "citadel-predict-python/0.2.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_error_response(self, response: httpx.Response) -> None:
        status_code = response.status_code
        try:
            data = response.json()
            detail = data.get("detail")
            if isinstance(detail, list):
                # Pydantic validation error list format
                msg = "; ".join(
                    (
                        f"{err.get('loc', [])}: {err.get('msg', '')}"
                        if isinstance(err, dict)
                        else str(err)
                    )
                    for err in detail
                )
            elif detail:
                msg = str(detail)
            else:
                msg = response.text or f"HTTP error {status_code}"
        except ValueError:
            data = None
            msg = response.text or f"HTTP error {status_code}"

        if status_code == 401:
            raise CitadelAuthError(
                message=(
                    f"Authentication failed ({msg}). "
                    "Ensure your Citadel API key is set via api_key argument, "
                    "CITADEL_API_KEY environment variable, or ~/.citadel/config.toml"
                ),
                status_code=401,
                response_data=data,
            )

        if status_code == 429:
            retry_after: int | None = None
            raw_retry = response.headers.get("Retry-After")
            if raw_retry and raw_retry.isdigit():
                retry_after = int(raw_retry)
            raise CitadelRateLimitError(
                message=f"Rate limit exceeded: {msg}",
                status_code=429,
                response_data=data,
                retry_after=retry_after,
            )

        if status_code == 422:
            raise CitadelValidationError(
                message=f"Validation error: {msg}",
                status_code=422,
                response_data=data,
            )

        if status_code == 400:
            raise CitadelBadRequestError(
                message=f"Bad request: {msg}",
                status_code=400,
                response_data=data,
            )

        if 500 <= status_code < 600:
            raise CitadelServerError(
                message=f"Server error ({status_code}): {msg}",
                status_code=status_code,
                response_data=data,
            )

        raise CitadelError(
            message=f"API request failed with status {status_code}: {msg}",
            status_code=status_code,
            response_data=data,
        )

    def predict(
        self,
        task_text: str,
        tools: list[str] | None = None,
        num_tools: int | None = None,
        model_id: str = "claude-sonnet",
    ) -> dict[str, Any]:
        """
        Predict token budget and cost ranges for a given agent task description and tools.

        Parameters:
            task_text: Natural language task description (1-4000 chars).
            tools: List of tool names available to the agent.
            num_tools: Optional integer count (tools list takes precedence if provided).
            model_id: Model calibration identifier (default: 'claude-sonnet').

        Returns:
            Dictionary matching the API response schema:
            {
                "model_id": "claude-sonnet",
                "low_tokens": int,
                "expected_tokens": int,
                "high_tokens": int,
                "expected_cost_usd": Optional[float],
                "low_cost_usd": Optional[float],
                "high_cost_usd": Optional[float],
                "cost_note": Optional[str],
                "driving_factors": list[str],
                "features": dict,
                "confidence": str,
                "out_of_distribution": bool,
                "ood_reasons": list[str]
            }
        """
        tool_list = list(tools) if tools is not None else []
        url = f"{self.base_url}/api/predict"
        payload = {
            "task_text": task_text,
            "tools": tool_list,
            "model_id": model_id,
        }

        try:
            response = self._client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
            raise CitadelNetworkError(
                message=f"Network error communicating with Citadel API at {url}: {exc}",
                cause=exc,
            ) from exc

        if response.status_code != 200:
            self._handle_error_response(response)

        try:
            return response.json()
        except Exception as exc:
            raise CitadelServerError(
                message="Failed to parse JSON response from Citadel Predict API",
                status_code=response.status_code,
            ) from exc

    def health(self) -> dict[str, Any]:
        """Check health status and supported models on the Citadel Predict API."""
        url = f"{self.base_url}/api/health"
        try:
            response = self._client.get(url, headers=self._get_headers())
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
            raise CitadelNetworkError(
                message=f"Network error communicating with Citadel API at {url}: {exc}",
                cause=exc,
            ) from exc

        if response.status_code != 200:
            self._handle_error_response(response)

        return response.json()

    def _execution_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        expected_status: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        """Send one authenticated request to the opt-in execution feedback API."""
        url = f"{self.base_url}{path}"
        try:
            response = self._client.request(
                method,
                url,
                json=payload,
                headers=self._get_headers(),
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RequestError) as exc:
            raise CitadelNetworkError(
                message=f"Network error communicating with Citadel API at {url}: {exc}",
                cause=exc,
            ) from exc

        if response.status_code not in expected_status:
            self._handle_error_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise CitadelServerError(
                message="Failed to parse JSON response from Citadel execution API",
                status_code=response.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise CitadelServerError(
                message="Citadel execution API returned an invalid response",
                status_code=response.status_code,
            )
        return data

    def create_execution_prediction(
        self,
        task_text: str,
        tools: list[str] | None = None,
        model_id: str = "claude-sonnet",
        *,
        provider: str | None = None,
        exact_provider_model: str | None = None,
    ) -> dict[str, Any]:
        """Create a persisted prediction snapshot for a forthcoming execution."""
        payload: dict[str, Any] = {
            "task_text": task_text,
            "tools": list(tools) if tools is not None else [],
            "model_id": model_id,
        }
        if provider is not None:
            payload["provider"] = provider
        if exact_provider_model is not None:
            payload["exact_provider_model"] = exact_provider_model
        return self._execution_request(
            "POST",
            "/api/executions/predictions",
            payload,
            expected_status=(201,),
        )

    def create_adaptive_prediction(
        self,
        task_text: str,
        *,
        provider: str,
        exact_provider_model: str,
        tools: list[str] | None = None,
        model_id: str = "claude-sonnet",
    ) -> dict[str, Any]:
        """Persist a base prediction and return its workspace profile adjustment."""
        return self._execution_request(
            "POST",
            "/api/predictions/adaptive",
            {
                "task_text": task_text,
                "tools": list(tools) if tools is not None else [],
                "model_id": model_id,
                "provider": provider,
                "exact_provider_model": exact_provider_model,
            },
            expected_status=(201,),
        )

    def create_execution(
        self,
        *,
        prediction_id: str | None,
        integration_name: str,
        integration_version: str | None,
        provider: str,
        exact_provider_model: str,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Register a new execution under the configured workspace."""
        payload: dict[str, Any] = {
            "prediction_id": prediction_id,
            "integration_name": integration_name,
            "integration_version": integration_version,
            "provider": provider,
            "exact_provider_model": exact_provider_model,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return self._execution_request(
            "POST",
            "/api/executions",
            payload,
            expected_status=(201,),
        )

    def begin_execution(
        self,
        *,
        task_text: str | None = None,
        prediction_id: str | None = None,
        tools: list[str] | None = None,
        model_id: str = "claude-sonnet",
        integration_name: str = "citadel-predict-sdk",
        integration_version: str | None = None,
        provider: str = "anthropic",
        exact_provider_model: str,
        use_adaptive_profile: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Create or reuse a prediction, start its execution, and return the safe detail record."""
        if (task_text is None) == (prediction_id is None):
            raise ValueError("Provide exactly one of task_text or prediction_id.")
        if use_adaptive_profile and task_text is None:
            raise ValueError(
                "Adaptive profiles require task_text so Citadel can create a prediction."
            )
        adaptive_prediction: dict[str, Any] | None = None
        if task_text is not None and use_adaptive_profile:
            adaptive_prediction = self.create_adaptive_prediction(
                task_text,
                tools=tools,
                model_id=model_id,
                provider=provider,
                exact_provider_model=exact_provider_model,
            )
            prediction_id = str(adaptive_prediction["prediction_id"])
        elif task_text is not None:
            prediction = self.create_execution_prediction(
                task_text,
                tools,
                model_id,
                provider=provider,
                exact_provider_model=exact_provider_model,
            )
            prediction_id = str(prediction["prediction_id"])
        execution = self.create_execution(
            prediction_id=prediction_id,
            integration_name=integration_name,
            integration_version=integration_version,
            provider=provider,
            exact_provider_model=exact_provider_model,
            reasoning_effort=reasoning_effort,
        )
        detail = self.get_execution_summary(str(execution["execution_id"]))
        if adaptive_prediction is not None:
            detail["adaptive_prediction"] = {
                "original_prediction": adaptive_prediction.get("original_prediction"),
                "adjusted_prediction": adaptive_prediction.get("adjusted_prediction"),
                "profile": adaptive_prediction.get("profile"),
            }
        return detail

    def record_execution_usage(
        self,
        execution_id: str,
        *,
        event_id: str,
        provider: str,
        exact_provider_model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
        occurred_at: str | None = None,
        provider_counters: dict[str, int | None] | None = None,
    ) -> dict[str, Any]:
        """Append an idempotent usage event without accepting caller provenance labels."""
        payload: dict[str, Any] = {
            "event_id": event_id,
            "provider": provider,
            "exact_provider_model": exact_provider_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "source": "mcp_tool_submission",
            "occurred_at": occurred_at or datetime.now(UTC).isoformat(),
        }
        if provider_counters is not None:
            payload["provider_counters"] = provider_counters
        return self._execution_request(
            "POST",
            f"/api/executions/{execution_id}/usage",
            payload,
        )

    def complete_execution(
        self,
        execution_id: str,
        *,
        status: str,
        expected_usage_event_count: int | None = None,
    ) -> dict[str, Any]:
        """Mark a started execution terminal and optionally declare event completeness."""
        payload: dict[str, Any] = {"status": status}
        if expected_usage_event_count is not None:
            payload["expected_usage_event_count"] = expected_usage_event_count
        return self._execution_request(
            "POST", f"/api/executions/{execution_id}/complete", payload
        )

    def get_execution_summary(self, execution_id: str) -> dict[str, Any]:
        """Read an execution summary from the configured workspace."""
        return self._execution_request("GET", f"/api/executions/{execution_id}")

    def close(self) -> None:
        """Close the underlying HTTP client session."""
        if not self._custom_client:
            self._client.close()

    def __enter__(self) -> "CitadelClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def predict_cost(
    task_text: str,
    tools: list[str] | None = None,
    num_tools: int | None = None,
    model_id: str = "claude-sonnet",
    api_key: str | None = None,
    api_url: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience function to predict token budget and cost ranges for an agent run.

    Example:
        >>> from citadel_predict import predict_cost
        >>> result = predict_cost(
        ...     task_text="Research competitor pricing and draft report",
        ...     tools=["web_search", "draft_document"]
        ... )
        >>> print(result["expected_tokens"])
    """
    client_timeout = timeout if timeout is not None else 10.0
    with CitadelClient(
        api_key=api_key,
        base_url=api_url,
        timeout=client_timeout,
    ) as client:
        return client.predict(
            task_text=task_text,
            tools=tools,
            num_tools=num_tools,
            model_id=model_id,
        )
