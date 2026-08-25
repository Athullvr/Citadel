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
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
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
            log_obj.update(getattr(record, "structured_data"))
        return json.dumps(log_obj)


logger = logging.getLogger("citadel.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# Set up module resolution for data_collection
DATA_COLLECTION_DIR = Path(
    os.environ.get(
        "DATA_COLLECTION_DIR",
        str(Path(__file__).resolve().parent.parent / "data_collection"),
    )
)
sys.path.insert(0, str(DATA_COLLECTION_DIR))

from predict import (  # noqa: E402
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    load_bundle,
    predict as run_predict,
)
from tools import TOOL_SCHEMAS  # noqa: E402

# Rate Limiter setup
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="Citadel Predict API",
    version="1.0.0",
    description="Pre-execution LLM token budget and cost predictor for AI agent runs",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
) -> Optional[str]:
    """Verify Bearer token against configured CITADEL_API_KEY / API_KEY."""
    expected_key = os.environ.get("CITADEL_API_KEY") or os.environ.get("API_KEY")
    if not expected_key:
        # Auth not configured -> open access (dev/demo mode)
        return None

    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <API_KEY>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


# Middleware for Structured JSON Request Logging
@app.middleware("http")
async def structured_log_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

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
except Exception as e:
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
def health():
    return {
        "status": "ok",
        "supported_models": list(SUPPORTED_MODELS),
        "auth_enabled": bool(API_KEY_ENV),
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
    _auth: Optional[str] = Depends(verify_api_key),
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
