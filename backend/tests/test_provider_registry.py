from dataclasses import replace

import pytest
from prediction_registry import (
    LEGACY_CLAUDE_ESTIMATOR,
    resolve_prediction_registration,
    validate_prediction_request,
)
from provider_registry import (
    PROVIDER_MODELS,
    is_pricing_version_compatible,
    resolve_provider_model,
    validate_reasoning_effort,
)


def test_exact_provider_model_and_friendly_alias_resolution_are_provider_scoped():
    exact = resolve_provider_model("openai", "gpt-6-luna")
    alias = resolve_provider_model("OPENAI", "gPt-6 lUna")

    assert exact is not None
    assert alias == exact
    assert exact.exact_model_id == "gpt-6-luna"
    assert resolve_provider_model("google", "gpt-6-luna") is None
    assert resolve_provider_model("openai", "gpt-6-lun") is None
    assert resolve_provider_model("openai", "GPT-6-LUNA") is None


@pytest.mark.parametrize(
    ("provider", "model", "effort"),
    [
        ("openai", "gpt-6-luna", "none"),
        ("openai", "gpt-6-astra", "max"),
        ("openai", "gpt-6.1-sol", "xhigh"),
        ("google", "gemini-3.8-flash", "high"),
    ],
)
def test_documented_reasoning_efforts_are_model_specific(provider, model, effort):
    assert validate_reasoning_effort(provider, model, effort) == effort


def test_unsupported_effort_and_unknown_model_fail_closed():
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        validate_reasoning_effort("openai", "gpt-6-astra", "none")
    with pytest.raises(ValueError, match="unknown exact model"):
        validate_reasoning_effort("openai", "unlisted-model", "high")


def test_pricing_version_must_match_registered_schedule_exactly():
    candidate = resolve_provider_model("openai", "gpt-6-luna")
    assert candidate is not None
    assert candidate.pricing_version is None
    assert not is_pricing_version_compatible(candidate, "openai-list-2026-11")

    versioned = replace(candidate, pricing_version="openai-list-2026-11")
    assert is_pricing_version_compatible(versioned, "openai-list-2026-11")
    assert not is_pricing_version_compatible(versioned, "openai-list-2026-10")


def test_claude_estimator_remains_a_legacy_alias_without_exact_model_claim():
    registration = resolve_prediction_registration("claude-sonnet")

    assert registration == LEGACY_CLAUDE_ESTIMATOR
    assert registration.estimator_status == "calibrated"
    assert registration.exact_provider_model is None
    assert validate_prediction_request(
        "claude-sonnet", provider="anthropic", exact_provider_model="claude-sonnet-5"
    ) == registration


@pytest.mark.parametrize("model_id", [model.exact_model_id for model in PROVIDER_MODELS])
def test_registered_candidate_models_never_resolve_to_claude_estimator(model_id):
    registration = resolve_prediction_registration(model_id)
    assert registration is not None
    assert registration.estimator_status == "uncalibrated"
    with pytest.raises(ValueError, match="no compatible calibrated estimator"):
        validate_prediction_request(model_id)


def test_prediction_provider_mismatch_is_rejected():
    with pytest.raises(ValueError, match="not compatible"):
        validate_prediction_request(
            "claude-sonnet", provider="openai", exact_provider_model="gpt-6-luna"
        )

    with pytest.raises(ValueError, match="registered to provider 'google'"):
        validate_prediction_request(
            "claude-sonnet", provider="anthropic", exact_provider_model="gemini-3.8-flash"
        )
