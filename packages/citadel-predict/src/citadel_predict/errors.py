"""
Exception hierarchy for citadel-predict.
"""

from typing import Any, Optional


class CitadelError(Exception):
    """Base exception for all Citadel Predict client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class CitadelAuthError(CitadelError):
    """Raised when authentication fails (HTTP 401) or API key is missing/invalid."""

    def __init__(
        self,
        message: str = (
            "Authentication failed: Missing or invalid Citadel API key. "
            "Set CITADEL_API_KEY environment variable, pass api_key='...', "
            "or configure api_key in ~/.citadel/config.toml"
        ),
        status_code: int = 401,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_data=response_data)


class CitadelRateLimitError(CitadelError):
    """Raised when API rate limits are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please back off and retry later.",
        status_code: int = 429,
        response_data: Optional[Any] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_data=response_data)
        self.retry_after = retry_after

    def __str__(self) -> str:
        base = f"[{self.status_code}] {self.message}"
        if self.retry_after is not None:
            base += f" (Retry-After: {self.retry_after}s)"
        return base


class CitadelValidationError(CitadelError):
    """Raised when the server rejects invalid request data (HTTP 422)."""

    def __init__(
        self,
        message: str = "Request validation failed. Check task_text and tools formatting.",
        status_code: int = 422,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_data=response_data)


class CitadelBadRequestError(CitadelError):
    """Raised when the request is bad or model_id is unsupported (HTTP 400)."""

    def __init__(
        self,
        message: str = "Bad request. Unsupported model or invalid payload.",
        status_code: int = 400,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_data=response_data)


class CitadelServerError(CitadelError):
    """Raised when the Citadel API server returns a 5xx error."""

    def __init__(
        self,
        message: str = "Citadel Predict server error. Please try again later.",
        status_code: int = 500,
        response_data: Optional[Any] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_data=response_data)


class CitadelNetworkError(CitadelError):
    """Raised when a network connection fails or requests time out."""

    def __init__(
        self,
        message: str = "Network error: Failed to connect to Citadel Predict API.",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
