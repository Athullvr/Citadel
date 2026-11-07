"""Canonical immutable identity for estimator predictions and their telemetry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

IDENTITY_SCHEMA_VERSION = "citadel.prediction-identity.v1"
IDENTITY_STATUS_RESOLVED: Final = "resolved"
IDENTITY_STATUS_LEGACY_UNRESOLVED: Final = "legacy_unresolved"
IDENTITY_STATUS_UNLINKED: Final = "unlinked"
PredictionIdentityStatus = Literal["resolved", "legacy_unresolved"]

# This is the target used by the frozen training rows: Anthropic input_tokens
# (which excludes cache counters) plus output_tokens.
CLAUDE_TOKEN_TARGET_SEMANTICS = "anthropic.cache_exclusive_input_plus_output.v1"

_IDENTITY_FIELDS = (
    "identity_schema_version",
    "identity_status",
    "provider",
    "exact_provider_model",
    "model_alias",
    "token_target_semantics",
    "estimator_artifact_version",
    "estimator_artifact_sha256",
    "pricing_schedule_version",
)


@dataclass(frozen=True, slots=True)
class PredictionIdentity:
    identity_schema_version: str
    identity_status: PredictionIdentityStatus
    provider: str
    exact_provider_model: str | None
    model_alias: str
    token_target_semantics: str
    estimator_artifact_version: str
    estimator_artifact_sha256: str
    pricing_schedule_version: str | None
    prediction_identity_hash: str

    def hash_fields(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("prediction_identity_hash")
        return values

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def identity_hash(fields: dict[str, Any]) -> str:
    """Hash the complete, schema-versioned identity using canonical JSON."""
    canonical_fields = {name: fields.get(name) for name in _IDENTITY_FIELDS}
    encoded = json.dumps(
        canonical_fields,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_prediction_identity(
    *,
    identity_status: PredictionIdentityStatus,
    provider: str,
    exact_provider_model: str | None,
    model_alias: str,
    token_target_semantics: str,
    estimator_artifact_version: str,
    estimator_artifact_sha256: str,
    pricing_schedule_version: str | None,
) -> PredictionIdentity:
    if not provider or provider != provider.strip().lower():
        raise ValueError("Prediction identity provider must be a normalized provider ID.")
    if identity_status == IDENTITY_STATUS_LEGACY_UNRESOLVED and exact_provider_model is not None:
        raise ValueError("A legacy unresolved identity cannot claim an exact provider model.")
    if identity_status == IDENTITY_STATUS_RESOLVED and not exact_provider_model:
        raise ValueError("A resolved prediction identity requires an exact provider model.")
    if exact_provider_model is not None and exact_provider_model != exact_provider_model.strip():
        raise ValueError("Exact provider model identity must be trimmed.")
    if (
        len(estimator_artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in estimator_artifact_sha256)
    ):
        raise ValueError("The estimator artifact checksum must be a lowercase SHA-256 hex digest.")
    if not estimator_artifact_version or not model_alias or not token_target_semantics:
        raise ValueError(
            "Prediction identity requires artifact, model alias, and target semantics."
        )
    fields: dict[str, Any] = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_status": identity_status,
        "provider": provider,
        "exact_provider_model": exact_provider_model,
        "model_alias": model_alias,
        "token_target_semantics": token_target_semantics,
        "estimator_artifact_version": estimator_artifact_version,
        "estimator_artifact_sha256": estimator_artifact_sha256,
        "pricing_schedule_version": pricing_schedule_version,
    }
    return PredictionIdentity(
        **fields,
        prediction_identity_hash=identity_hash(fields),
    )


def prediction_identity_hash_from_record(record: Any) -> str:
    """Recompute a prediction row's identity digest for integrity checks."""
    return identity_hash(prediction_identity_fields_from_record(record))


def prediction_identity_fields_from_record(record: Any) -> dict[str, Any]:
    """Return the canonical identity fields as stored on a prediction row."""
    return {
        "identity_schema_version": record.identity_schema_version,
        "identity_status": record.identity_status,
        "provider": record.provider,
        "exact_provider_model": record.exact_provider_model,
        "model_alias": record.model_alias,
        "token_target_semantics": record.token_target_semantics,
        "estimator_artifact_version": record.estimator_artifact_version,
        "estimator_artifact_sha256": record.estimator_artifact_sha256,
        "pricing_schedule_version": record.pricing_version,
    }


def prediction_identity_matches_snapshot(record: Any, snapshot: Any) -> bool:
    """Compare every immutable identity field, not only its digest, to the snapshot."""
    if not isinstance(snapshot, dict):
        return False
    expected = prediction_identity_fields_from_record(record)
    expected["prediction_identity_hash"] = record.prediction_identity_hash
    return all(snapshot.get(name) == value for name, value in expected.items())
