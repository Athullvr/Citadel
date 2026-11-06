"""Prediction-to-estimator routing with explicit calibrated-model boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from provider_registry import PROVIDER_MODELS, ProviderModel, validate_provider_model_identity

PredictionStatus = Literal["calibrated", "uncalibrated", "unavailable"]


@dataclass(frozen=True, slots=True)
class PredictionRegistration:
    model_id: str
    provider: str
    exact_provider_model: str | None
    friendly_alias: str
    estimator_id: str | None
    estimator_status: PredictionStatus
    artifact_version: str | None
    pricing_version: str | None
    description: str

    def public_dict(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "exact_provider_model": self.exact_provider_model,
            "friendly_alias": self.friendly_alias,
            "estimator_id": self.estimator_id,
            "estimator_status": self.estimator_status,
            "artifact_version": self.artifact_version,
            "pricing_version": self.pricing_version,
            "description": self.description,
        }


LEGACY_CLAUDE_ESTIMATOR = PredictionRegistration(
    model_id="claude-sonnet",
    provider="anthropic",
    exact_provider_model=None,
    friendly_alias="Claude Sonnet (legacy calibrated estimator)",
    estimator_id="frozen-claude-sonnet-bundle",
    estimator_status="calibrated",
    artifact_version=None,
    pricing_version=None,
    description=(
        "Existing frozen Claude Sonnet estimator alias. The repository does not map this "
        "alias to an exact provider model identifier."
    ),
)


def _registration_for_provider_model(model: ProviderModel) -> PredictionRegistration:
    return PredictionRegistration(
        model_id=model.exact_model_id,
        provider=model.provider,
        exact_provider_model=model.exact_model_id,
        friendly_alias=model.friendly_alias,
        estimator_id=None,
        estimator_status="uncalibrated",
        artifact_version=None,
        pricing_version=model.pricing_version,
        description=(
            "Provider model is registered, but Citadel has no compatible calibrated estimator."
        ),
    )


PROVIDER_PREDICTIONS = tuple(_registration_for_provider_model(model) for model in PROVIDER_MODELS)


def resolve_prediction_registration(model_id: str) -> PredictionRegistration | None:
    """Resolve only explicit aliases and exact model IDs; never choose a fallback."""
    if model_id == LEGACY_CLAUDE_ESTIMATOR.model_id:
        return LEGACY_CLAUDE_ESTIMATOR
    return next(
        (item for item in PROVIDER_PREDICTIONS if model_id == item.exact_provider_model),
        None,
    )


def validate_prediction_request(
    model_id: str,
    *,
    provider: str | None = None,
    exact_provider_model: str | None = None,
) -> PredictionRegistration:
    """Return a compatible calibrated estimator or fail without fallback.

    The legacy Claude alias has no exact model binding. A caller may submit the
    observed runtime model for telemetry compatibility, but it is never adopted
    as the estimator identity. New registrations with an exact model must match
    that model exactly.
    """
    registration = resolve_prediction_registration(model_id)
    if registration is None:
        raise ValueError(f"Unsupported model_id '{model_id}'.")
    if registration.estimator_status != "calibrated":
        raise ValueError(
            f"Model '{model_id}' is registered but has no compatible calibrated estimator."
        )
    if provider is not None and registration.provider != provider.strip().lower():
        raise ValueError(
            f"Estimator '{model_id}' is not compatible with provider '{provider}'."
        )
    if exact_provider_model is not None and provider is None:
        raise ValueError("provider is required when exact_provider_model is supplied.")
    if registration.exact_provider_model is not None:
        if exact_provider_model is not None and (
            exact_provider_model != registration.exact_provider_model
        ):
            raise ValueError(
                f"Estimator '{model_id}' is bound to exact model "
                f"'{registration.exact_provider_model}'."
            )
    elif exact_provider_model is not None and provider is not None:
        # Keep the historical telemetry request shape. This is an observation,
        # not a claim that the frozen estimator is calibrated for that model.
        validate_provider_model_identity(provider, exact_provider_model)
    return registration


def prediction_catalog() -> list[dict[str, str | None]]:
    return [
        LEGACY_CLAUDE_ESTIMATOR.public_dict(),
        *(item.public_dict() for item in PROVIDER_PREDICTIONS),
    ]
