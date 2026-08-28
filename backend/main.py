"""
FastAPI backend for Citadel Predict — AI Agent Cost Predictor.

Features:
- Bearer API key authentication via CITADEL_API_KEY / API_KEY env var
- Rate limiting via slowapi
- Strict Pydantic input validation (character/array bounds)
- Multi-model-ready prediction registry via model_id (default: 'claude-sonnet')
- Out-of-Distribution (OOD) & confidence estimation
- Structured JSON logging
- Explicit CORS origin enforcement
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# Configure JSON logger
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "structured_data"):
            log_obj.update(record.structured_data)
        return json.dumps(log_obj)


logger = logging.getLogger("citadel.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# Set up robust module resolution for data_collection
def resolve_data_collection_dir() -> Path:
    env_dir = os.environ.get("DATA_COLLECTION_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.exists():
            return p

    candidates = [
        Path(__file__).resolve().parent.parent / "data_collection",
        Path(__file__).resolve().parent / "data_collection",
        Path.cwd() / "data_collection",
        Path.cwd().parent / "data_collection",
    ]
    for c in candidates:
        if c.exists() and (c / "predict.py").exists():
            return c.resolve()

    return Path(__file__).resolve().parent.parent / "data_collection"


DATA_COLLECTION_DIR = resolve_data_collection_dir()
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from predict import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    load_bundle,
)
from predict import (
    predict as run_predict,
)
from tools import TOOL_SCHEMAS

# Rate Limiter setup
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="Citadel Predict API",
    version="1.0.0",
    description="Pre-execution LLM token budget and cost predictor for AI agent runs",
)
app.state.limiter = limiter


# Standardized Error Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_server_error",
    }
    error_type = code_map.get(exc.status_code, "http_error")
    detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    body: dict[str, Any] = {
        "error": error_type,
        "message": detail_str,
        "detail": jsonable_encoder(exc.detail),
    }
    headers = dict(exc.headers or {})
    return JSONResponse(status_code=exc.status_code, content=body, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    msg = "; ".join(
        f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}"
        for err in errors
    )
    body: dict[str, Any] = {
        "error": "validation_error",
        "message": msg or "Invalid request parameters",
        "detail": errors,
    }
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    retry_after = 60
    body: dict[str, Any] = {
        "error": "rate_limited",
        "message": f"Rate limit exceeded: {exc.detail}",
        "detail": f"Rate limit exceeded: {exc.detail}",
        "retry_after": retry_after,
    }
    headers = {
        "Retry-After": str(retry_after),
        "X-RateLimit-Limit": "30",
        "X-RateLimit-Remaining": "0",
    }
    return JSONResponse(status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=body, headers=headers)


# CORS configuration
raw_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in raw_allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# API Key Authentication Setup
API_KEY_ENV = os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY")
security_scheme = HTTPBearer(auto_error=False)


def is_auth_enabled() -> bool:
    """Check if authentication is active (via CITADEL_REQUIRE_AUTH or configured API key)."""
    require_auth = os.environ.get("CITADEL_REQUIRE_AUTH", "").strip().lower() in ("true", "1", "yes")
    has_key = bool(os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY"))
    return require_auth or has_key


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security_scheme),  # noqa: B008
) -> str | None:
    """Verify Bearer token against configured CITADEL_API_KEY / API_KEY."""
    expected_key = os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY")
    require_auth = os.environ.get("CITADEL_REQUIRE_AUTH", "").strip().lower() in ("true", "1", "yes")

    if not expected_key and not require_auth:
        # Auth not configured -> open access (dev/demo mode)
        return None

    if not credentials or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <API_KEY>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server has CITADEL_REQUIRE_AUTH=true but no CITADEL_API_KEY is configured in the environment.",
        )

    if credentials.credentials.strip() != expected_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials.strip()


# Middleware for Structured JSON Request Logging & Rate-Limit Visibility
@app.middleware("http")
async def structured_log_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Attach rate limit visibility header if not already present
    if "X-RateLimit-Limit" not in response.headers:
        if request.url.path == "/api/predict":
            response.headers["X-RateLimit-Limit"] = "30"
        else:
            response.headers["X-RateLimit-Limit"] = "60"

    client_host = request.client.host if request.client else "unknown"
    record = logging.LogRecord(
        name="citadel.request",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=f"{request.method} {request.url.path} {response.status_code}",
        args=(),
        exc_info=None,
    )
    record.structured_data = {  # type: ignore[attr-defined]
        "client_ip": client_host,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": duration_ms,
    }
    logger.handle(record)
    return response


# Preload default model bundle at startup
try:
    load_bundle(DEFAULT_MODEL)
    logger.info(f"Loaded default calibration bundle for model '{DEFAULT_MODEL}'")
except (FileNotFoundError, OSError, ValueError) as e:
    logger.warning(f"Could not preload default bundle for '{DEFAULT_MODEL}': {e}")


# Schemas with bounds
class PredictRequest(BaseModel):
    task_text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Raw task description for the agent run (max 4000 characters).",
        examples=["Research competitor pricing across 3 sources and draft a summary."],
    )
    tools: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="List of tool names available to the agent (max 20 tools).",
        examples=[["web_search", "fetch_url", "draft_document"]],
    )
    model_id: str = Field(
        default=DEFAULT_MODEL,
        max_length=64,
        description="Model calibration identifier (e.g. 'claude-sonnet').",
    )

    @field_validator("tools")
    @classmethod
    def validate_tool_names(cls, v: list[str]) -> list[str]:
        for tool in v:
            if len(tool) > 64:
                raise ValueError(f"Tool name '{tool}' exceeds maximum length of 64 characters")
        return v


class PredictResponse(BaseModel):
    model_id: str
    low_tokens: int
    expected_tokens: int
    high_tokens: int
    driving_factors: list[str]
    features: dict[str, Any]
    confidence: str
    out_of_distribution: bool
    ood_reasons: list[str]


@app.get("/api/health")
@app.get("/api/version")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_id": DEFAULT_MODEL,
        "supported_models": list(SUPPORTED_MODELS),
        "auth_enabled": is_auth_enabled(),
    }


@app.get("/api/tools")
@limiter.limit("60/minute")
def list_tools(request: Request):
    """The known tool names + descriptions for the UI's tool picker."""
    return [
        {"name": name, "description": schema["description"]}
        for name, schema in TOOL_SCHEMAS.items()
    ]


@app.post("/api/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict(
    request: Request,
    req: PredictRequest,
    _auth: str | None = Depends(verify_api_key),
):
    """
    Predict token cost range [low, expected, high] for a given task and tool list.
    Requires Bearer API key auth if CITADEL_API_KEY / API_KEY is set.
    """
    if req.model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported model_id '{req.model_id}'. "
                f"Supported models in v1: {list(SUPPORTED_MODELS)}"
            ),
        )

    try:
        result = run_predict(req.task_text, req.tools, model_id=req.model_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/api/validation-examples")
@limiter.limit("60/minute")
def validation_examples(request: Request):
    """4 real leave-one-task-out examples showing model performance on unseen tasks."""
    path = DATA_COLLECTION_DIR / "data" / "validation_examples.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Validation examples not found",
        )
    return json.loads(path.read_text(encoding="utf-8"))
