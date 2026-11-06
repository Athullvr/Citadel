"""Provider and exact-model metadata used by prediction and telemetry boundaries.

Registry entries describe documented provider models; they do not imply that
Citadel has an estimator, authenticated provider integration, or verified usage
writer for that model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RegistryStatus = Literal["supported", "scaffolded", "unavailable"]
EstimatorStatus = Literal["calibrated", "uncalibrated", "unavailable"]

REGISTRY_VERIFIED_ON = "2026-11-06"


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    provider_id: str
    display_name: str
    status: RegistryStatus
    usage_normalizer: bool
    authenticated_telemetry: bool


@dataclass(frozen=True, slots=True)
class ProviderModel:
    provider: str
    exact_model_id: str
    friendly_alias: str
    model_family: str
    capabilities: frozenset[str]
    context_window_tokens: int | None
    max_output_tokens: int | None
    input_token_semantics: str
    output_token_semantics: str
    cache_accounting_semantics: str
    reasoning_efforts: frozenset[str]
    pricing_version: str | None
    estimator_status: EstimatorStatus
    telemetry_status: RegistryStatus
    documentation_url: str
    verified_on: str = REGISTRY_VERIFIED_ON

    def public_dict(self) -> dict[str, object]:
        """Return public model metadata without internal workspace information."""
        result = asdict(self)
        result["capabilities"] = sorted(self.capabilities)
        result["reasoning_efforts"] = sorted(self.reasoning_efforts)
        return result


PROVIDERS: dict[str, ProviderDefinition] = {
    "anthropic": ProviderDefinition(
        "anthropic", "Anthropic", "scaffolded", usage_normalizer=True,
        authenticated_telemetry=False,
    ),
    "openai": ProviderDefinition(
        "openai", "OpenAI", "scaffolded", usage_normalizer=True,
        authenticated_telemetry=False,
    ),
    "google": ProviderDefinition(
        "google", "Google", "scaffolded", usage_normalizer=True,
        authenticated_telemetry=False,
    ),
}


# Candidate model metadata is copied from current official provider docs. Prices
# are intentionally absent: price schedules are separate, versioned data and
# are not used to create predictions for uncalibrated models.
PROVIDER_MODELS: tuple[ProviderModel, ...] = (
    ProviderModel(
        provider="openai",
        exact_model_id="gpt-6-luna",
        friendly_alias="GPT-6 Luna",
        model_family="GPT-6",
        capabilities=frozenset({"text_input", "image_input", "text_output", "tools"}),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_token_semantics="provider_input_includes_cached_tokens",
        output_token_semantics="provider_output_includes_reasoning_tokens",
        cache_accounting_semantics="cached_tokens_are_a_breakdown_of_input_tokens",
        reasoning_efforts=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
        pricing_version=None,
        estimator_status="uncalibrated",
        telemetry_status="scaffolded",
        documentation_url="https://developers.openai.com/api/docs/models/gpt-6-luna",
    ),
    ProviderModel(
        provider="openai",
        exact_model_id="gpt-6.1-terra",
        friendly_alias="GPT-6.1 Terra",
        model_family="GPT-6",
        capabilities=frozenset({"text_input", "image_input", "text_output", "tools"}),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_token_semantics="provider_input_includes_cached_tokens",
        output_token_semantics="provider_output_includes_reasoning_tokens",
        cache_accounting_semantics="cached_tokens_are_a_breakdown_of_input_tokens",
        reasoning_efforts=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
        pricing_version=None,
        estimator_status="uncalibrated",
        telemetry_status="scaffolded",
        documentation_url="https://developers.openai.com/api/docs/models/gpt-6.1-terra",
    ),
    ProviderModel(
        provider="openai",
        exact_model_id="gpt-6.1-sol",
        friendly_alias="GPT-6.1 Sol",
        model_family="GPT-6",
        capabilities=frozenset({"text_input", "image_input", "text_output", "tools"}),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_token_semantics="provider_input_includes_cached_tokens",
        output_token_semantics="provider_output_includes_reasoning_tokens",
        cache_accounting_semantics="cached_tokens_are_a_breakdown_of_input_tokens",
        reasoning_efforts=frozenset({"low", "medium", "high", "xhigh", "max"}),
        pricing_version=None,
        estimator_status="uncalibrated",
        telemetry_status="scaffolded",
        documentation_url="https://developers.openai.com/api/docs/models/gpt-6.1-sol",
    ),
    ProviderModel(
        provider="openai",
        exact_model_id="gpt-6-astra",
        friendly_alias="GPT-6 Astra",
        model_family="GPT-6",
        capabilities=frozenset({"text_input", "image_input", "text_output", "tools"}),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_token_semantics="provider_input_includes_cached_tokens",
        output_token_semantics="provider_output_includes_reasoning_tokens",
        cache_accounting_semantics="cached_tokens_are_a_breakdown_of_input_tokens",
        reasoning_efforts=frozenset({"low", "medium", "high", "xhigh", "max"}),
        pricing_version=None,
        estimator_status="uncalibrated",
        telemetry_status="scaffolded",
        documentation_url="https://developers.openai.com/api/docs/models/gpt-6-astra",
    ),
    ProviderModel(
        provider="google",
        exact_model_id="gemini-3.8-flash",
        friendly_alias="Gemini 3.8 Flash",
        model_family="Gemini Flash",
        capabilities=frozenset({"text_input", "image_input", "audio_input", "video_input",
                                "text_output", "tools", "context_caching", "reasoning"}),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_token_semantics="prompt_tokens_include_cached_content",
        output_token_semantics="candidate_tokens_exclude_separately_reported_thoughts",
        cache_accounting_semantics="cached_content_tokens_are_a_breakdown_of_prompt_tokens",
        reasoning_efforts=frozenset({"low", "medium", "high"}),
        pricing_version=None,
        estimator_status="uncalibrated",
        telemetry_status="scaffolded",
        documentation_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    ),
)

_MODELS_BY_PROVIDER: dict[str, tuple[ProviderModel, ...]] = {
    provider: tuple(model for model in PROVIDER_MODELS if model.provider == provider)
    for provider in PROVIDERS
}


def normalize_provider_id(provider: str) -> str:
    """Normalize only provider identity casing/whitespace, never model aliases."""
    return provider.strip().lower()


def resolve_provider_model(provider: str, identifier: str) -> ProviderModel | None:
    """Resolve an exact provider ID or a registered friendly alias.

    Exact model IDs are case-sensitive. Friendly aliases are matched case-
    insensitively, and no family, prefix, or cross-provider inference is used.
    """
    provider_id = normalize_provider_id(provider)
    for model in _MODELS_BY_PROVIDER.get(provider_id, ()):
        if identifier == model.exact_model_id:
            return model
        if identifier.casefold() == model.friendly_alias.casefold():
            return model
    return None


def validate_reasoning_effort(
    provider: str, exact_model_id: str, reasoning_effort: str | None
) -> str | None:
    """Validate a requested effort against a documented exact model.

    An effort cannot be validated for an unknown model, so it fails closed.
    Omitting the optional setting preserves compatibility with legacy clients.
    """
    if reasoning_effort is None:
        return None
    model = resolve_provider_model(provider, exact_model_id)
    if model is None:
        raise ValueError("Reasoning effort cannot be validated for an unknown exact model.")
    if reasoning_effort not in model.reasoning_efforts:
        supported = ", ".join(sorted(model.reasoning_efforts)) or "none"
        raise ValueError(
            f"Unsupported reasoning effort for {exact_model_id}; supported: {supported}."
        )
    return reasoning_effort


def is_pricing_version_compatible(model: ProviderModel, pricing_version: str | None) -> bool:
    """Allow a price only when the model and estimate name the same known version."""
    return bool(model.pricing_version and pricing_version == model.pricing_version)


def public_model_registry() -> list[dict[str, object]]:
    """Return model metadata for a public, non-tenant-specific discovery route."""
    return [model.public_dict() for model in PROVIDER_MODELS]
