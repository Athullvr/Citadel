import pytest
from usage_normalization import normalize_usage


def test_anthropic_cache_fields_are_separate_and_not_double_counted():
    normalized = normalize_usage(
        provider="anthropic",
        exact_provider_model="claude-sonnet-5",
        provider_counters={
            "input_tokens": 40,
            "output_tokens": 8,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 20,
            "thinking_tokens": 3,
        },
    )

    assert normalized.input_tokens == 40
    assert normalized.output_tokens == 8
    assert normalized.cache_creation_input_tokens == 10
    assert normalized.cache_read_input_tokens == 20
    assert normalized.cached_input_tokens == 30
    assert normalized.reasoning_tokens == 3
    assert normalized.total_tokens is None
    assert normalized.usage_completeness == "complete"
    assert normalized.provider_counters["cache_read_input_tokens"] == 20


def test_anthropic_missing_cache_counter_remains_null_and_incomplete():
    normalized = normalize_usage(
        provider="anthropic",
        exact_provider_model="claude-sonnet-5",
        provider_counters={"input_tokens": 12, "output_tokens": 4},
    )

    assert normalized.cache_creation_input_tokens is None
    assert normalized.cache_read_input_tokens is None
    assert normalized.cached_input_tokens is None
    assert normalized.usage_completeness == "incomplete"
    assert "cache_read_input_tokens" not in normalized.provider_counters


def test_openai_cached_and_reasoning_counters_are_breakdowns_not_additions():
    normalized = normalize_usage(
        provider="openai",
        exact_provider_model="gpt-6-luna",
        provider_counters={
            "input_tokens": 20,
            "input_tokens_details.cached_tokens": 8,
            "output_tokens": 11,
            "output_tokens_details.reasoning_tokens": 6,
            "total_tokens": 31,
        },
    )

    assert normalized.input_tokens == 20
    assert normalized.cached_input_tokens == 8
    assert normalized.output_tokens == 11
    assert normalized.reasoning_tokens == 6
    assert normalized.total_tokens == 31
    assert normalized.cache_creation_input_tokens is None
    assert normalized.cache_read_input_tokens is None
    assert normalized.usage_completeness == "complete"


def test_gemini_cached_prompt_and_reasoning_fields_keep_provider_total_semantics():
    normalized = normalize_usage(
        provider="google",
        exact_provider_model="gemini-3.8-flash",
        provider_counters={
            "promptTokenCount": 20,
            "cachedContentTokenCount": 8,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 6,
            "toolUsePromptTokenCount": 3,
            "totalTokenCount": 34,
        },
    )

    assert normalized.input_tokens == 20
    assert normalized.cached_input_tokens == 8
    assert normalized.output_tokens == 5
    assert normalized.reasoning_tokens == 6
    assert normalized.tool_use_prompt_tokens == 3
    assert normalized.total_tokens == 34
    assert normalized.usage_completeness == "complete"


def test_unknown_provider_counters_are_retained_but_marked_unclassified():
    normalized = normalize_usage(
        provider="some-provider",
        exact_provider_model="model-v1",
        provider_counters={"input_tokens": None, "vendor_total": 3},
    )

    assert normalized.input_tokens is None
    assert normalized.provider_counters == {"input_tokens": None, "vendor_total": 3}
    assert normalized.token_semantics == "some-provider.unclassified.v1"
    assert normalized.usage_completeness == "unknown"


@pytest.mark.parametrize(
    "counters",
    [
        {"input_tokens": -1},
        {"input_tokens": True},
        {"input_tokens": "4"},
        {"": 2},
    ],
)
def test_invalid_provider_counter_values_are_rejected(counters):
    with pytest.raises(ValueError):
        normalize_usage(
            provider="openai",
            exact_provider_model="gpt-6-luna",
            provider_counters=counters,
        )


def test_legacy_non_anthropic_shape_is_preserved_as_unclassified():
    normalized = normalize_usage(
        provider="openai",
        exact_provider_model="gpt-6-luna",
        legacy_counters={"input_tokens": 9, "output_tokens": 2},
    )

    assert normalized.input_tokens == 9
    assert normalized.output_tokens == 2
    assert normalized.token_semantics == "openai.unclassified.v1"
    assert normalized.usage_completeness == "unknown"
