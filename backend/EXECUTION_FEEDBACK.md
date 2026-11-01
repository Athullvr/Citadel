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
| `CITADEL_API_KEY` | Yes for feedback routes | Bearer key for all `/api/executions` routes. `API_KEY` is accepted as the existing deployment fallback. These routes return 503 when neither is configured and 401 for missing or invalid credentials when a key is configured. |
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

## Lifecycle

1. `POST /api/executions/predictions` calls the existing predictor and stores
its output, feature values/schema, model artifact SHA-256 when available,
   pricing rates/date, and the 75/25 input/output cost assumption.
2. `POST /api/executions` creates an execution with an optional prediction ID.
   The response's `execution_id` is the correlation key for later calls.
3. `POST /api/executions/{execution_id}/usage` appends one provider response's
   usage. Supply a stable UUID `event_id` and reuse it unchanged on retries.
4. `POST /api/executions/{execution_id}/complete` moves a started execution to
   `completed` or `failed` and returns its current usage aggregate.
5. `GET /api/executions/{execution_id}` returns metadata, linked prediction
   summary, and usage aggregate.

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
  "exact_provider_model": "claude-sonnet-5",
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
zero means it reported a zero. Identical retries return `recorded: false`.
Reusing an event ID with different values returns 409. New usage cannot be
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

Prediction costs remain estimates based on the existing `claude-sonnet` pricing
alias and 75/25 split, not actual billed cost. The existing runner requests the
exact model ID `claude-sonnet-5`; telemetry stores that identifier alongside
the estimator alias and does not claim those identifiers or pricing schedules
are equivalent. Current pricing registry metadata is preserved as-is, including
its `last_updated` value; it is not refreshed or independently verified here.

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
