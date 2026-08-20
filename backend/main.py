"""
FastAPI backend for the cost predictor demo UI.

Wraps the exact prediction logic validated in Phase 2 (data_collection/predict.py,
features.py, model.joblib) -- deliberately NOT reimplemented here, so the UI can
never silently drift from what was actually trained/validated.
"""

import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# In local dev, data_collection/ sits next to backend/. In the Docker image,
# only the inference-relevant files are copied to /app/data_collection (see
# backend/Dockerfile) -- DATA_COLLECTION_DIR lets both layouts work unchanged.
DATA_COLLECTION_DIR = Path(
    os.environ.get("DATA_COLLECTION_DIR", str(Path(__file__).resolve().parent.parent / "data_collection"))
)
sys.path.insert(0, str(DATA_COLLECTION_DIR))

from predict import load_bundle, predict as run_predict  # noqa: E402
from tools import TOOL_SCHEMAS  # noqa: E402

app = FastAPI(title="Agent Cost Predictor API")

# Comma-separated list, e.g. "https://cost-predictor.vercel.app,http://localhost:3000"
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model bundle once at startup, not per-request.
_bundle = load_bundle(str(DATA_COLLECTION_DIR / "model.joblib"))


class PredictRequest(BaseModel):
    task_text: str
    tools: list[str] = []


class PredictResponse(BaseModel):
    low_tokens: int
    expected_tokens: int
    high_tokens: int
    driving_factors: list[str]
    features: dict


@app.get("/api/tools")
def list_tools():
    """The known tool names + descriptions, for the UI's tool picker."""
    return [
        {"name": name, "description": schema["description"]}
        for name, schema in TOOL_SCHEMAS.items()
    ]


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = run_predict(req.task_text, req.tools, bundle=_bundle)
    return result


@app.get("/api/validation-examples")
def validation_examples():
    """4 real leave-one-task-out examples (data_collection/generate_examples.py)
    showing where the model's predicted range matched -- and where it
    missed -- an actual observed run. Static/precomputed, not live."""
    path = DATA_COLLECTION_DIR / "data" / "validation_examples.json"
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok"}
