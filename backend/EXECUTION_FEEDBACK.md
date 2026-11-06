# Execution feedback API (0.3.0)

This opt-in API adds a durable execution ledger while leaving `POST /api/predict`
and the existing `estimate_agent_cost` MCP tool unchanged. It records prediction
snapshots, execution metadata, and provider-reported usage. It does not enforce
budgets, perform online learning, or intercept arbitrary agent traffic.

## Storage and configuration

Set these environment variables on every API worker:

| Variable | Required | Purpose |
| --- | --- | --- |
| `CITADEL_DATABASE_URL` | Yes for feedback routes | Hosted PostgreSQL URL (`postgresql://...` or `postgres://...`, normalized to psycopg 3). Local development may use a file-backed URL such as `sqlite:////absolute/path/citadel.db`. In-memory SQLite is rejected because it is not durable across workers. |
| `CITADEL_EXECUTION_WORKSPACE_ID` | Yes for feedback routes | Server-owned identity used for every record. It is not accepted from request bodies. This release uses one configured workspace; it does not implement multi-tenant authentication. |
| `CITADEL_API_KEY` | Yes for feedback/evaluation routes | Bearer key for `/api/executions` and `/api/evaluations/summary`. `API_KEY` is accepted as the existing deployment fallback. These routes return 503 when neither is configured and 401 for missing or invalid credentials when a key is configured. |
| `CITADEL_TASK_HASH_KEY` | Optional | Secret HMAC key for a keyed task hash. Without it, no task hash is stored. Raw task text is never stored in the prediction record. |

PostgreSQL is the hosted source of truth. All workers must use the same database
and workspace configuration. Use PostgreSQL for multiple API workers, a
restricted database role, and TLS for remote connections. File-backed SQLite is
intended for local, single-process development only.

Install backend requirements, set `CITADEL_DATABASE_URL`, then apply the schema
explicitly from the backend directory:

```bash
pip install -r requirements.txt
alembic upgrade head
```

Migrations are not run by application startup. The initial migration creates
the prediction, execution, and usage tables plus append-only database triggers
for prediction snapshots and usage events. Back up PostgreSQL before applying
schema changes in a deployed environment.

Migration `0003_cancelled_execution_status` extends the execution status check
to allow `cancelled`. Its downgrade refuses to run while cancelled rows exist so
it cannot silently rewrite those outcomes as failures.

## Lifecycle

1. `POST /api/executions/predictions` calls the existing predictor and stores
its output, feature values/schema, model artifact SHA-256 when available,
   pricing rates/date, and the 75/25 input/output cost assumption.
2. `POST /api/executions` creates an execution with an optional prediction ID.
   The response's `execution_id` is the correlation key for later calls.
3. `POST /api/executions/{execution_id}/usage` appends one provider response's
   usage. Supply a stable UUID `event_id` and reuse it unchanged on retries.
4. `POST /api/executions/{execution_id}/complete` moves a started execution to
   `completed`, `failed`, or `cancelled` and returns its current usage aggregate. Callers may
   include `expected_usage_event_count`, the number of unique usage reports they
   expect for this execution. Omission leaves completeness `unknown` for legacy
   compatibility. A count mismatch is `incomplete`; equality is `complete` by
   the caller's declaration and does not prove provider telemetry is truthful.
   Completion and detail responses include `usage_completeness` and the declared
   count in execution metadata.
5. `GET /api/executions/{execution_id}` returns metadata, linked prediction
   summary, and usage aggregate.

Terminal states are final: a second completion cannot overwrite the first, and
new usage events cannot be added afterward. Cancelled executions remain distinct
from failed executions and are excluded from completed-execution accuracy reports.

The caller must carry `execution_id` from the start response through usage and
completion requests. A prediction can be associated with multiple execution
records. Parent IDs are metadata only; parent and child usage are not rolled up.

All requests use `Authorization: Bearer <CITADEL_API_KEY>`. Workspace identity
is derived on the server. Request bodies reject additional fields, including
`workspace_id`, and execution POST bodies are limited to 256 KiB.

Example usage event:

```json
{
  "event_id": "55f8f99d-cc54-5db1-9d9d-2ece3f82295f",
  "provider": "anthropic",
  "exact_provider_model": null,
  "input_tokens": 1200,
  "output_tokens": 240,
  "cache_creation_input_tokens": 300,
  "cache_read_input_tokens": 40,
  "source": "anthropic_messages_api",
  "occurred_at": "2026-11-02T10:00:00Z"
}
```

`input_tokens` and `output_tokens` are required non-negative integers. Cache
counters are nullable: `null` means the provider response omitted the field;
zero means it reported a zero. A retry with the same event ID, provider/model,
source, and counters returns `recorded: false`; the first observation timestamp
is retained if a client generates a new one during a retry. Reusing an event ID
with different usage data returns 409. New usage cannot be
added after terminal completion, but an identical retry of an already stored
event remains idempotent.

For Anthropic, this release records the Messages API usage fields separately.
Anthropic documents total input as `input_tokens` plus cache creation and cache
read counters, so Citadel's aggregate `total_input_tokens` adds those counters
only when every event has both cache values. Other providers' totals remain
null because their accounting semantics have not been established here. The
current SDK Message schema exposes `id`, `model`, and `usage`, but no response
creation timestamp; the runner therefore sends its UTC observation time as
`occurred_at`. See Anthropic's
[prompt caching usage fields](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
and [Python SDK message schema](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message.py).

The runner's configured model string is a request value, not a verified
provider-resolved model identity. Historical run rows store that configured
string and the frozen estimator bundle contains no exact model metadata, so
new predictions under `claude-sonnet` have `identity_status` set to
`legacy_unresolved` and no exact model. The runner does not promote its
caller-submitted response model into execution identity. This keeps synthetic,
unverified usage events attached to the unresolved prediction without making
them exact-model calibration evidence.

Prediction identity snapshots contain the provider, exact model when resolved,
model alias, token-target semantics, estimator artifact version and SHA-256,
pricing schedule version when present, identity schema version, and a canonical
identity hash. Executions and usage events copy the schema/status, token target,
and prediction identity hash; provider and exact model must match at each API
boundary. An identity mismatch returns 409. Linked predictions with unresolved
identity remain readable but are excluded from evaluation and adaptive profiles.

Migration `0005_prediction_identity` adds nullable identity fields and marks old
predictions, executions, and usage events as unresolved without backfilling a
model guess. Downgrade refuses to remove new identity-bearing or null-model
execution/usage rows. Historical records that predate this migration remain
readable and are explicitly unresolved after upgrade.

## Anthropic reference runner

Telemetry is disabled by default. To opt in, set `CITADEL_API_URL` and
`CITADEL_API_KEY`, install `data_collection/requirements.txt`, and run:

```bash
python data_collection/run_collection.py --task-ids task-01 --runs-per-task 1 --citadel-telemetry
```

The runner reports each Anthropic Messages response as a separate event. It
uses the response's `id` for a deterministic UUIDv5 retry key, the response's
`model` for the exact provider model, and the provider-reported input, output,
cache-creation, and cache-read fields. Missing optional cache fields stay null.
The adapter retries transient HTTP failures with the same event ID and does
not log task or usage payloads. Telemetry is best-effort; a Citadel outage does
not stop the agent run.

Runner task inputs and tools are benchmark data and tools are deterministic
mocks. The execution is flagged `is_synthetic: true` and the usage source says
`anthropic_messages_api_synthetic_tools`; do not treat these records as
production-agent telemetry. Local JSON telemetry output omits raw task text and
uses `data/telemetry_runs.jsonl`, leaving the frozen `data/runs.jsonl` baseline
untouched. The runner's historical `total_input_tokens` and `total_tokens`
fields retain their previous semantics; `total_processed_input_tokens` is an
additional cache-inclusive value when all cache counters are present.

## Prediction evaluation (0.3.1)

`GET /api/evaluations/summary` is authenticated with the same server API key and
uses the server-selected `CITADEL_EXECUTION_WORKSPACE_ID`; it accepts no
workspace filter. Optional `from` and `to` timestamps are timezone-aware and
filter on `execution.started_at` with an inclusive start and exclusive end.
Optional filters select the prediction alias/artifact version, provider, exact
provider model, and integration name/version. The report is computed from the
immutable prediction, execution, and usage-event records and is not persisted.
Prompts, feature snapshots, and task hashes are not returned.

The primary target is Anthropic `input_tokens + output_tokens`. The recorded
Anthropic `input_tokens` is already cache-exclusive, so cache creation and cache
read counters are not subtracted from it. Those nullable counters are reported
separately; cache-inclusive totals are unavailable when any cache field is
missing. The execution detail endpoint's legacy `total_tokens` remains its
cache-inclusive value and is not the evaluation target. Other providers are
excluded until their token-counter semantics are explicitly supported.

Usage events are grouped in SQL by `(workspace_id, execution_id)` before they
are joined to predictions. Identical event retries do not add rows or inflate
the event count. A prediction linked to more than one execution is excluded
from accuracy metrics rather than choosing or combining a retry. Started
executions and failed executions are kept out of completed-execution accuracy;
failed usage is shown in a separate consumption group. Missing usage is counted
separately and is never treated as zero.

Every metric value includes a sample count, excluded count, and exclusion
reasons. Point metrics use the expected prediction and signed error
`actual - predicted`: positive bias means underprediction. MAE is mean absolute
error; WAPE is `sum(abs(error)) / sum(actual) * 100` and is unavailable when
the actual-token denominator is zero; median error uses the conventional
middle-value median; P90 uses nearest-rank `ceil(0.9 * n)`; under/overprediction
rates count strict positive/negative errors over all valid point observations;
interval coverage is the percentage of actuals within valid inclusive bounds;
interval width is the mean `high - low`. Invalid intervals are excluded only
from interval metrics and include an exclusion reason. Empty samples return
unavailable values. Groups with fewer than 30 distinct available task hashes,
or with any missing task hash, are marked insufficient or independence-unknown.

The report separates `verified_provider`, `unverified_submitted`,
`synthetic_benchmark`, and legacy `unknown` provenance. The shared Bearer key
does not authenticate provider identity: new non-synthetic submissions are
unverified regardless of the caller-supplied `source`, while records explicitly
marked synthetic are isolated as benchmark telemetry. No current API path can
establish `verified_provider`, so descriptive metrics must not be presented as
verified production accuracy. The report also segments available samples by
prediction artifact, provider/exact model, and integration. Task category is
unavailable because production execution records do not store it.

Example summary shape (metric groups contain all point and interval metrics):

```json
{
  "evaluation_status": "available",
  "methodology_version": "citadel-token-evaluation-v1",
  "primary_token_target": {
    "id": "anthropic.cache_exclusive_input_plus_output.v1"
  },
  "eligible_completed_execution_count": 1,
  "excluded_execution_count": 0,
  "missing_usage_event_execution_count": 0,
  "completeness_distribution": {"complete": 1},
  "provenance_groups": [
    {
      "provenance": "unverified_submitted",
      "metric_class": "descriptive_unverified",
      "sample_count": 1,
      "reliability": "task_independence_unknown",
      "production_accuracy_claim_eligible": false,
      "metrics": {
        "mean_absolute_error_tokens": {
          "value": 58.0,
          "sample_count": 1,
          "excluded_count": 0,
          "exclusion_counts": {},
          "unavailable_reason": null
        }
      }
    }
  ]
}
```

## Responses and error behavior

Successful prediction and execution creation return 201. Usage, completion,
and detail requests return 200. Feedback routes document:

- 400: unsupported estimator model.
- 401: missing or invalid Bearer key.
- 404: record absent from the configured workspace.
- 409: event ID conflict, provider mismatch, or invalid state transition.
- 413: request body exceeds 256 KiB.
- 422: schema or timestamp validation failure.
- 503: database or required auth/workspace configuration unavailable.

Normal FastAPI validation errors keep the backend's existing standardized JSON
error envelope. Legacy prediction behavior and optional legacy authentication
are unchanged.

## Adaptive execution profiles (0.5.0)

`POST /api/predictions/adaptive` creates a persisted prediction using the same
estimator, then returns `original_prediction`, `adjusted_prediction`, and a
safe profile summary. It requires the existing execution Bearer key and uses
the server-selected workspace. The existing `POST /api/predict` contract is
unchanged. The stored prediction keeps the model's original token values in its
immutable columns and response snapshot; the profile result and shifted token
interval are stored as separate metadata in that same immutable snapshot.

The endpoint accepts `task_text`, `tools`, `model_id`, `provider`, and the exact
`exact_provider_model`. It never returns the feature snapshot, prompt, task
hash, or workspace identifier. The total-token adjustment shifts low, expected,
and high estimates by the same integer amount, preserving interval width and
order. Dollar cost fields stay at the original estimate because a total-token
residual cannot be split reliably between input and output tokens.

Profiles use only evaluator-eligible completed executions in the configured
workspace with `verified_provider` provenance, matching model artifact, the
Anthropic cache-exclusive input-plus-output target, and an HMAC task hash. Each
task hash contributes one median residual, so duplicate executions and retries
cannot multiply the same task's weight. Profiles need at least 30 distinct
hashed tasks, and the exact 95% distribution-free order-statistic confidence
interval for the median residual must exclude zero. The median correction is
bounded by 50% of the current expected estimate, the existing interval width,
and the non-negative lower-bound constraint. Fractional median residuals round
to the nearest whole token with half values rounded away from zero. Profiles
older than 180 days are stale and do not adjust estimates.

All candidate cohorts are scoped to the requested exact provider model and
matching prediction artifact. The workspace cohort reports the same
workspace-local compatible evidence; the provider/model profile retains
selection priority. Residuals from different exact provider model identifiers
are never pooled without an explicit compatibility policy. Project identity and
task category are not stored, so those levels are reported unavailable. The
global baseline is the unchanged model output; Citadel does not pool residuals
across workspaces. Status, sample counts, provenance counts, exclusion reasons,
confidence interval, and fallback reason are included in the response.

Configure a stable `CITADEL_TASK_HASH_KEY` to allow exact repeated-task
deduplication. Without it, task independence is unknown and no profile can
activate. Current API submissions remain `unverified_submitted`; the shared
Bearer key and caller-provided source do not prove provider authenticity. This
means profiles safely fall back to the baseline until an execution is recorded
through a trusted path that establishes `verified_provider`. The median
confidence interval assumes distinct task hashes represent independent tasks;
semantically near-identical prompts are not detected.
