"""Provider-specific usage normalization into a lossless internal record.

Normalized counters retain their provider semantics. In particular, shared
field names do not make OpenAI, Gemini, and Anthropic token totals comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from provider_registry import normalize_provider_id

UsageCompleteness = Literal["complete", "incomplete", "unknown"]
MAX_COUNTER_VALUE = 2**63 - 1


@dataclass(frozen=True, slots=True)
class NormalizedUsage:
    provider: str
    exact_provider_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    tool_use_prompt_tokens: int | None
    total_tokens: int | None
    token_semantics: str
    usage_completeness: UsageCompleteness
    provider_counters: dict[str, int | None]


class UsageAdapter(Protocol):
    provider_id: str

    def normalize(
        self,
        exact_provider_model: str | None,
        counters: Mapping[str, int | None],
    ) -> NormalizedUsage: ...


def validate_counter_map(counters: Mapping[str, int | None]) -> dict[str, int | None]:
    """Validate a flat provider counter map, rejecting nested/raw content."""
    if len(counters) > 32:
        raise ValueError("provider_counters may contain at most 32 counter fields.")
    checked: dict[str, int | None] = {}
    for name, value in counters.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 96:
            raise ValueError("Provider counter names must be non-empty and at most 96 characters.")
        if value is not None and (type(value) is not int or not 0 <= value <= MAX_COUNTER_VALUE):
            raise ValueError(f"Provider counter {name!r} must be a nullable non-negative integer.")
        checked[name] = value
    return checked


def _value(counters: Mapping[str, int | None], name: str) -> int | None:
    return counters.get(name)


def _completeness(
    counters: Mapping[str, int | None], required: tuple[str, ...]
) -> UsageCompleteness:
    if all(name in counters and counters[name] is not None for name in required):
        return "complete"
    if any(name in counters for name in required):
        return "incomplete"
    return "unknown"


class AnthropicMessagesAdapter:
    """Normalize Anthropic Messages counters without adding cache twice."""

    provider_id = "anthropic"

    def normalize(
        self, exact_provider_model: str | None, counters: Mapping[str, int | None]
    ) -> NormalizedUsage:
        input_tokens = _value(counters, "input_tokens")
        output_tokens = _value(counters, "output_tokens")
        cache_creation = _value(counters, "cache_creation_input_tokens")
        cache_read = _value(counters, "cache_read_input_tokens")
        cached_input = (
            cache_creation + cache_read
            if cache_creation is not None and cache_read is not None
            else None
        )
        return NormalizedUsage(
            provider=self.provider_id,
            exact_provider_model=exact_provider_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            cached_input_tokens=cached_input,
            reasoning_tokens=_value(counters, "thinking_tokens"),
            tool_use_prompt_tokens=None,
            total_tokens=_value(counters, "total_tokens"),
            token_semantics="anthropic.messages.cache_exclusive_input.v1",
            usage_completeness=_completeness(
                counters,
                (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ),
            ),
            provider_counters=dict(counters),
        )


class OpenAIResponsesAdapter:
    """Normalize OpenAI Responses usage fields as documented by OpenAI."""

    provider_id = "openai"

    def normalize(
        self, exact_provider_model: str | None, counters: Mapping[str, int | None]
    ) -> NormalizedUsage:
        return NormalizedUsage(
            provider=self.provider_id,
            exact_provider_model=exact_provider_model,
            input_tokens=_value(counters, "input_tokens"),
            output_tokens=_value(counters, "output_tokens"),
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cached_input_tokens=_value(counters, "input_tokens_details.cached_tokens"),
            reasoning_tokens=_value(counters, "output_tokens_details.reasoning_tokens"),
            tool_use_prompt_tokens=None,
            total_tokens=_value(counters, "total_tokens"),
            token_semantics="openai.responses.input_includes_cached.v1",
            usage_completeness=_completeness(
                counters,
                (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "input_tokens_details.cached_tokens",
                    "output_tokens_details.reasoning_tokens",
                ),
            ),
            provider_counters=dict(counters),
        )


class GeminiGenerateContentAdapter:
    """Normalize Gemini GenerateContent usageMetadata counter names."""

    provider_id = "google"

    def normalize(
        self, exact_provider_model: str | None, counters: Mapping[str, int | None]
    ) -> NormalizedUsage:
        return NormalizedUsage(
            provider=self.provider_id,
            exact_provider_model=exact_provider_model,
            input_tokens=_value(counters, "promptTokenCount"),
            output_tokens=_value(counters, "candidatesTokenCount"),
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            cached_input_tokens=_value(counters, "cachedContentTokenCount"),
            reasoning_tokens=_value(counters, "thoughtsTokenCount"),
            tool_use_prompt_tokens=_value(counters, "toolUsePromptTokenCount"),
            total_tokens=_value(counters, "totalTokenCount"),
            token_semantics="google.gemini.generate_content.v1",
            usage_completeness=_completeness(
                counters,
                (
                    "promptTokenCount",
                    "candidatesTokenCount",
                    "totalTokenCount",
                    "cachedContentTokenCount",
                    "thoughtsTokenCount",
                    "toolUsePromptTokenCount",
                ),
            ),
            provider_counters=dict(counters),
        )


_ADAPTERS: dict[str, UsageAdapter] = {
    "anthropic": AnthropicMessagesAdapter(),
    "openai": OpenAIResponsesAdapter(),
    "google": GeminiGenerateContentAdapter(),
}


def normalize_usage(
    *,
    provider: str,
    exact_provider_model: str | None,
    provider_counters: Mapping[str, int | None] | None = None,
    legacy_counters: Mapping[str, int | None] | None = None,
    legacy_fields_present: set[str] | None = None,
) -> NormalizedUsage:
    """Normalize provider counters or preserve the legacy Anthropic-shaped API.

    Legacy direct fields remain accepted. They retain Anthropic semantics only
    when the request provider is Anthropic. For other or unknown providers the
    values are preserved with an explicitly unclassified semantics identifier.
    """
    provider_id = normalize_provider_id(provider)
    if provider_counters is not None and legacy_counters:
        raise ValueError("Use provider_counters or legacy token fields, not both.")

    if provider_counters is not None:
        counters = validate_counter_map(provider_counters)
        adapter = _ADAPTERS.get(provider_id)
        if adapter is None:
            return NormalizedUsage(
                provider=provider_id,
                exact_provider_model=exact_provider_model,
                input_tokens=_value(counters, "input_tokens"),
                output_tokens=_value(counters, "output_tokens"),
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                cached_input_tokens=None,
                reasoning_tokens=None,
                tool_use_prompt_tokens=None,
                total_tokens=_value(counters, "total_tokens"),
                token_semantics=f"{provider_id or 'unknown'}.unclassified.v1",
                usage_completeness="unknown",
                provider_counters=counters,
            )
        return adapter.normalize(exact_provider_model, counters)

    counters = validate_counter_map(legacy_counters or {})
    if provider_id == "anthropic":
        present = legacy_fields_present or set(counters)
        raw = {key: value for key, value in counters.items() if key in present}
        return _ADAPTERS["anthropic"].normalize(exact_provider_model, raw)

    return NormalizedUsage(
        provider=provider_id,
        exact_provider_model=exact_provider_model,
        input_tokens=_value(counters, "input_tokens"),
        output_tokens=_value(counters, "output_tokens"),
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        cached_input_tokens=None,
        reasoning_tokens=None,
        tool_use_prompt_tokens=None,
        total_tokens=None,
        token_semantics=f"{provider_id or 'unknown'}.unclassified.v1",
        usage_completeness="unknown",
        provider_counters=counters,
    )
