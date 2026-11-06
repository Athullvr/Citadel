# Multi-provider execution intelligence foundation

This release adds model metadata and provider-specific usage normalization. It
does not add live provider API clients, authenticated telemetry, or calibrated
estimators for the newly registered provider models.

## Availability

| Classification | Current scope |
| --- | --- |
| **Registered** | Provider and exact-model metadata is exposed by `GET /api/models`. Registration describes a model; it does not make it selectable for prediction. |
| **Scaffolded** | Anthropic Messages, OpenAI Responses, and Google Gemini GenerateContent usage normalizers accept caller-submitted counters. Submissions remain unverified. |
| **Prediction-capable** | The existing frozen `claude-sonnet` alias only. Its prediction identity is `legacy_unresolved`; the repository does not identify the exact provider model represented by the artifact. |
| **Uncalibrated / unavailable for prediction** | `gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol`, `gpt-6-astra`, and `gemini-3.8-flash`. Citadel has no compatible estimator for them. |
| **Telemetry-capable** | Public REST, SDK, and MCP endpoints accept provider-specific usage counters as `unverified_submitted`; they are not authenticated provider telemetry. |
| **Calibrated / validated exact model** | None. The legacy estimator has historical point-estimate LOTO results, but cannot be bound to an exact provider model. Its interval coverage is descriptive, not an independent validation result. |
| **Unavailable** | Provider-authenticated telemetry writers and exact-model estimators for the registered OpenAI and Google candidates. |

The model identifiers and capabilities above were checked against the
[OpenAI model catalog](https://developers.openai.com/api/docs/models), the
[Gemini model catalog](https://ai.google.dev/gemini-api/docs/models), and the
[Gemini 3.8 Flash reference](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
on 2026-11-06. Registry metadata does not mean Citadel can access or invoke a
provider model. Reasoning-effort values are validated against the selected
exact model when an execution is registered.

`GET /api/models` separates provider model metadata from prediction estimator
availability. `/api/predict` preserves its existing `claude-sonnet` request and
response shape. Persisted predictions carry an immutable identity snapshot; the
legacy alias is explicitly unresolved and cannot activate an exact-model
adaptive profile. Requesting a registered but uncalibrated provider model
returns an explicit error. Citadel never silently falls back to the Claude
estimator for another provider.

## Usage semantics

The execution usage endpoint accepts existing flat token fields for backwards
compatibility and a provider-specific `provider_counters` map for provider
usage responses. The map allows only named integer counters or `null`; it is
not a raw provider response and must not contain prompt text, API payloads, or
secrets. The normalized counters and original named counters are stored on the
append-only usage event. Missing counters stay `null` and are accompanied by
incomplete or unknown completeness metadata.

- **Anthropic Messages:** `input_tokens` excludes cache reads and cache
  creation; `cache_creation_input_tokens` and `cache_read_input_tokens` are
  retained as separate values. The existing evaluation target remains
  cache-exclusive `input_tokens + output_tokens`. Thinking tokens, when
  reported, are a breakdown of output tokens.
- **OpenAI Responses:** `input_tokens` includes cached input; the
  `input_tokens_details.cached_tokens` counter is a breakdown, never added to
  input again. `output_tokens` includes reasoning tokens, with the reported
  reasoning counter retained separately.
- **Gemini GenerateContent:** `promptTokenCount` includes cached content;
  `cachedContentTokenCount` is a breakdown. Candidate, thoughts, tool-use
  prompt, and provider-reported total counters remain separate.

Provider semantics are recorded on each event. Evaluation remains limited to
the existing Anthropic token target, so OpenAI, Gemini, unknown providers, and
unclassified legacy counters are excluded from that accuracy claim. The
normalizer does not convert one provider's tokens into another provider's
definition.

Example SDK workflow for a registered but uncalibrated model uses an existing
provider-compatible prediction only when one is available; otherwise create an
execution without a prediction through the execution endpoint. Submitted usage
remains unverified:

```python
from citadel_predict import CitadelClient

with CitadelClient() as client:
    execution = client.create_execution(
        prediction_id=None,
        integration_name="my-agent",
        integration_version="1.2.0",
        provider="openai",
        exact_provider_model="gpt-6-luna",
        reasoning_effort="high",
    )
    client.record_execution_usage(
        execution["execution_id"],
        event_id="d2ecb362-3937-4fc2-b3e0-8d6a3fd3fd36",
        provider="openai",
        exact_provider_model="gpt-6-luna",
        provider_counters={
            "input_tokens": 120,
            "input_tokens_details.cached_tokens": 40,
            "output_tokens": 35,
            "output_tokens_details.reasoning_tokens": 18,
            "total_tokens": 155,
        },
    )
    client.complete_execution(
        execution["execution_id"],
        status="completed",
        expected_usage_event_count=1,
    )
```

The caller-supplied source label does not change provenance. Public REST, SDK,
and MCP submissions are `unverified_submitted`; `is_synthetic` remains a
separate caller declaration and never establishes verified provenance. A
provider-authenticated integration must be added before telemetry can be
classified as verified.

## Pricing and prediction boundaries

Provider price is not inferred from total tokens because input/output
allocation, caching, reasoning, and provider billing rules differ. Existing
Claude pricing remains in its current pricing module and snapshot metadata.
Candidate models have no Citadel pricing version, and the prediction path does
not apply provider-list prices to them. A future pricing schedule must carry
its own version and source and match that version when calculating costs.

The model registry records official context and output limits and supported
reasoning-effort options for currently documented candidates. It does not
install provider SDKs or make live calls. No API keys, live service access, or
verified telemetry were used in this implementation.
