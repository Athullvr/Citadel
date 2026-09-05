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

__version__ = "0.1.1"
__all__ = [
    "predict_cost",
    "CitadelClient",
    "resolve_api_key",
    "resolve_api_url",
    "CitadelError",
    "CitadelAuthError",
    "CitadelRateLimitError",
    "CitadelValidationError",
    "CitadelBadRequestError",
    "CitadelServerError",
    "CitadelNetworkError",
]
