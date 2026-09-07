"""
Pricing table and dollar-cost calculation engine for Citadel Predict.

Maps model_id to published provider pricing (input/output cost per 1K tokens)
and computes estimated dollar bounds using empirical multi-turn agent input/output token ratios.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPricing:
    """
    Pricing metadata for a specific model.

    Attributes:
        model_id: Unique model identifier (e.g. 'claude-sonnet').
        provider: LLM provider name (e.g. 'anthropic', 'openai', 'google').
        input_cost_per_1k: Cost in USD per 1,000 input (prompt) tokens.
        output_cost_per_1k: Cost in USD per 1,000 output (completion) tokens.
        last_updated: ISO-8601 date string when pricing was verified.
        currency: Currency code (default: 'USD').
        notes: Provider citation or pricing tier notes.
    """

    model_id: str
    provider: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    last_updated: str
    currency: str = "USD"
    notes: str = ""


# Published provider pricing registry.
# Extensible data structure: new models can be registered as simple data entries.
PRICING_REGISTRY: dict[str, ModelPricing] = {
    "claude-sonnet": ModelPricing(
        model_id="claude-sonnet",
        provider="anthropic",
        input_cost_per_1k=0.003,  # $3.00 per 1M input tokens
        output_cost_per_1k=0.015,  # $15.00 per 1M output tokens
        last_updated="2025-01-01",
        currency="USD",
        notes="Anthropic published Claude 3.5 Sonnet pricing: $3.00/1M input, $15.00/1M output.",
    ),
}

# Empirical multi-turn ReAct agent trajectory composition assumption:
# In iterative agent loops with tool usage, conversation context and tool outputs accumulate
# monotonically over successive turns. Benchmark agent runs exhibit an approximate split of
# 75% input tokens and 25% output tokens.
# NOTE: This is an empirical approximation for pre-execution cost estimation, not a guaranteed exact split.
DEFAULT_INPUT_RATIO: float = 0.75


def get_pricing(model_id: str) -> Optional[ModelPricing]:
    """Retrieve pricing metadata for a given model_id, or None if unpriced."""
    return PRICING_REGISTRY.get(model_id)


def calculate_token_cost_usd(
    tokens: int,
    model_id: str,
    input_ratio: float = DEFAULT_INPUT_RATIO,
) -> Optional[float]:
    """
    Calculate estimated dollar cost (USD) for a total token count and model.

    Formula:
        input_tokens = tokens * input_ratio
        output_tokens = tokens * (1 - input_ratio)
        cost = (input_tokens * input_rate / 1000) + (output_tokens * output_rate / 1000)

    Returns:
        Estimated USD cost rounded to 4 decimal places, or None if model has no pricing entry.
    """
    pricing = get_pricing(model_id)
    if pricing is None:
        return None

    clamped_ratio = max(0.0, min(1.0, input_ratio))
    input_tokens = tokens * clamped_ratio
    output_tokens = tokens * (1.0 - clamped_ratio)

    cost = (input_tokens * pricing.input_cost_per_1k / 1000.0) + (
        output_tokens * pricing.output_cost_per_1k / 1000.0
    )
    return round(cost, 4)


def calculate_token_cost_range(
    low_tokens: int,
    expected_tokens: int,
    high_tokens: int,
    model_id: str,
    input_ratio: float = DEFAULT_INPUT_RATIO,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Compute low, expected, and high dollar cost estimates for predicted token bounds.

    Returns:
        (low_cost_usd, expected_cost_usd, high_cost_usd, cost_note)
    """
    pricing = get_pricing(model_id)
    if pricing is None:
        note = (
            f"No pricing data registered for model '{model_id}'. "
            "Dollar cost estimate unavailable."
        )
        return None, None, None, note

    low_cost = calculate_token_cost_usd(low_tokens, model_id, input_ratio=input_ratio)
    expected_cost = calculate_token_cost_usd(
        expected_tokens, model_id, input_ratio=input_ratio
    )
    high_cost = calculate_token_cost_usd(high_tokens, model_id, input_ratio=input_ratio)

    in_pct = round(input_ratio * 100)
    out_pct = round((1.0 - input_ratio) * 100)
    note = (
        f"Cost estimated using {pricing.provider.capitalize()} published rates "
        f"(${pricing.input_cost_per_1k * 1000:.2f}/M in, ${pricing.output_cost_per_1k * 1000:.2f}/M out) "
        f"assuming {in_pct}% in / {out_pct}% out agent trajectory composition."
    )
    return low_cost, expected_cost, high_cost, note
