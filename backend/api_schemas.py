"""Pydantic request and response schemas shared by the HTTP API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    task_text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Raw task description for the agent run (max 50000 characters).",
        examples=["Research competitor pricing across 3 sources and draft a summary."],
    )
    tools: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="List of tool names available to the agent (max 20 tools).",
        examples=[["web_search", "fetch_url", "draft_document"]],
    )
    model_id: str = Field(
        default="claude-sonnet",
        max_length=64,
        description="Model calibration identifier (e.g. 'claude-sonnet').",
    )

    @field_validator("tools")
    @classmethod
    def validate_tool_names(cls, value: list[str]) -> list[str]:
        for tool in value:
            if len(tool) > 64:
                raise ValueError(f"Tool name '{tool}' exceeds maximum length of 64 characters")
        return value


class PredictResponse(BaseModel):
    model_id: str
    low_tokens: int
    expected_tokens: int
    high_tokens: int
    expected_cost_usd: float | None = Field(
        default=None,
        description="Estimated expected dollar cost in USD (75% input / 25% output approximation).",
    )
    low_cost_usd: float | None = Field(
        default=None, description="Estimated lower bound dollar cost in USD."
    )
    high_cost_usd: float | None = Field(
        default=None, description="Estimated upper bound dollar cost in USD."
    )
    cost_note: str | None = Field(
        default=None, description="Information or assumptions regarding dollar cost estimation."
    )
    driving_factors: list[str]
    features: dict[str, Any]
    confidence: str
    out_of_distribution: bool
    ood_reasons: list[str]
