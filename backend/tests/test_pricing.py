"""
Unit tests for the Citadel Predict pricing registry and dollar-cost calculation engine.
"""

from pricing import (
    DEFAULT_INPUT_RATIO,
    PRICING_REGISTRY,
    ModelPricing,
    calculate_token_cost_range,
    calculate_token_cost_usd,
    get_pricing,
)
from predict import predict


def test_pricing_registry_claude_sonnet():
    pricing = get_pricing("claude-sonnet")
    assert pricing is not None
    assert isinstance(pricing, ModelPricing)
    assert pricing.model_id == "claude-sonnet"
    assert pricing.provider == "anthropic"
    assert pricing.input_cost_per_1k == 0.003
    assert pricing.output_cost_per_1k == 0.015
    assert pricing.currency == "USD"
    assert len(pricing.last_updated) == 10  # YYYY-MM-DD
    assert "Anthropic" in pricing.notes


def test_pricing_lookup_unknown_model():
    pricing = get_pricing("nonexistent-future-model-999")
    assert pricing is None


def test_calculate_token_cost_usd_math():
    # 10,000 tokens on Claude Sonnet with standard 75% in / 25% out split:
    # input_tokens = 7,500 * 0.003 / 1000 = $0.0225
    # output_tokens = 2,500 * 0.015 / 1000 = $0.0375
    # total = $0.0600
    cost = calculate_token_cost_usd(10000, "claude-sonnet")
    assert cost == 0.06

    # 100% input tokens:
    cost_pure_input = calculate_token_cost_usd(10000, "claude-sonnet", input_ratio=1.0)
    assert cost_pure_input == 0.03

    # 100% output tokens:
    cost_pure_output = calculate_token_cost_usd(10000, "claude-sonnet", input_ratio=0.0)
    assert cost_pure_output == 0.15

    # Zero tokens:
    assert calculate_token_cost_usd(0, "claude-sonnet") == 0.0

    # Unknown model returns None:
    assert calculate_token_cost_usd(10000, "unknown-model") is None


def test_calculate_token_cost_range_known_model():
    low, expected, high, note = calculate_token_cost_range(
        low_tokens=2000,
        expected_tokens=5000,
        high_tokens=10000,
        model_id="claude-sonnet",
    )
    assert low is not None
    assert expected is not None
    assert high is not None
    assert low < expected <= high
    assert note is not None
    assert "Anthropic" in note
    assert "75% in / 25% out" in note


def test_calculate_token_cost_range_unknown_model():
    low, expected, high, note = calculate_token_cost_range(
        low_tokens=2000,
        expected_tokens=5000,
        high_tokens=10000,
        model_id="unknown-model-xyz",
    )
    assert low is None
    assert expected is None
    assert high is None
    assert note is not None
    assert "No pricing data registered" in note


def test_predict_integration_returns_cost():
    result = predict("Analyze codebase vulnerabilities", ["fetch_url"], model_id="claude-sonnet")
    assert "expected_cost_usd" in result
    assert "low_cost_usd" in result
    assert "high_cost_usd" in result
    assert "cost_note" in result

    assert isinstance(result["expected_cost_usd"], float)
    assert result["expected_cost_usd"] > 0
    assert result["low_cost_usd"] <= result["expected_cost_usd"] <= result["high_cost_usd"]
    assert "Anthropic" in result["cost_note"]
