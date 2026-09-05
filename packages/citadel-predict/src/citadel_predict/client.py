"""
HTTP client for the Citadel Predict API.
"""

from pathlib import Path
from typing import Any, Optional

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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        config_path: Optional[Path] = None,
        http_client: Optional[httpx.Client] = None,
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
            "User-Agent": "citadel-predict-python/0.1.1",
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
                    f"{err.get('loc', [])}: {err.get('msg', '')}" if isinstance(err, dict) else str(err)
                    for err in detail
                )
            elif detail:
                msg = str(detail)
            else:
                msg = response.text or f"HTTP error {status_code}"
        except Exception:
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
            retry_after: Optional[int] = None
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
        tools: Optional[list[str]] = None,
        num_tools: Optional[int] = None,
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
            Dictionary matching the API response schema exactly:
            {
                "model_id": "claude-sonnet",
                "low_tokens": int,
                "expected_tokens": int,
                "high_tokens": int,
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
    tools: Optional[list[str]] = None,
    num_tools: Optional[int] = None,
    model_id: str = "claude-sonnet",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: Optional[float] = None,
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
