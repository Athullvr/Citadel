"""
citadel-predict: Developer client for AI agent pre-execution cost prediction.
"""

from .client import CitadelClient, predict_cost
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

__version__ = "0.2.0"
__all__ = [
    "CitadelAuthError",
    "CitadelBadRequestError",
    "CitadelClient",
    "CitadelError",
    "CitadelNetworkError",
    "CitadelRateLimitError",
    "CitadelServerError",
    "CitadelValidationError",
    "predict_cost",
    "resolve_api_key",
    "resolve_api_url",
]
