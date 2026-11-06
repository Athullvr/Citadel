from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from database import create_database_engine
from fastapi import HTTPException
from execution_feedback import (
    CreateExecutionRequest,
    CreateUsageEventRequest,
    create_execution,
    get_execution,
    record_usage_event,
)
from execution_models import ExecutionRecord, UsageEvent
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ID = "provider-test-workspace"


@pytest.fixture
def provider_db(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'provider-test.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    monkeypatch.setenv("CITADEL_EXECUTION_WORKSPACE_ID", WORKSPACE_ID)
    command.upgrade(Config(str(BACKEND_DIR / "alembic.ini")), "head")
    engine = create_database_engine(database_url)
    with Session(engine, expire_on_commit=False) as db:
        yield db
    engine.dispose()


def _new_execution(db: Session, *, provider: str = "openai", exact_model: str = "gpt-6-luna"):
    return create_execution(
        CreateExecutionRequest(
            integration_name="provider-test",
            provider=provider,
            exact_provider_model=exact_model,
            reasoning_effort="high" if provider == "openai" else None,
        ),
        _auth=None,
        db=db,
    )


def _usage_request(
    *, execution_provider: str = "openai", exact_model: str = "gpt-6-luna", counters=None
):
    return CreateUsageEventRequest(
        event_id=uuid4(),
        provider=execution_provider,
        exact_provider_model=exact_model,
        provider_counters=(
            counters
            if counters is not None
            else {
                "input_tokens": 20,
                "input_tokens_details.cached_tokens": 8,
                "output_tokens": 11,
                "output_tokens_details.reasoning_tokens": 6,
                "total_tokens": 31,
            }
        ),
        source="arbitrary-caller-label",
        occurred_at=datetime.now(UTC),
    )


def test_openai_usage_is_persisted_idempotently_and_summarized_by_its_own_semantics(
    provider_db,
):
    execution = _new_execution(provider_db)
    payload = _usage_request()

    first = record_usage_event(execution.execution_id, payload, _auth=None, db=provider_db)
    duplicate = record_usage_event(execution.execution_id, payload, _auth=None, db=provider_db)
    summary = get_execution(execution.execution_id, _auth=None, db=provider_db)

    assert first.recorded is True
    assert first.token_semantics == "openai.responses.input_includes_cached.v1"
    assert first.usage_completeness == "complete"
    assert duplicate.recorded is False
    assert summary.usage.input_tokens == 20
    assert summary.usage.cached_input_tokens == 8
    assert summary.usage.output_tokens == 11
    assert summary.usage.reasoning_tokens == 6
    assert summary.usage.provider_reported_total_tokens == 31
    assert summary.usage.cache_read_input_tokens is None
    assert summary.execution.reasoning_effort == "high"
    assert summary.execution.telemetry_provenance == "unverified_submitted"
    assert provider_db.scalar(select(func.count(UsageEvent.event_id))) == 1
    event = provider_db.scalar(select(UsageEvent))
    assert event is not None
    assert event.provider_counters == payload.provider_counters


def test_missing_provider_counter_stays_null_and_exact_model_mismatch_is_rejected(provider_db):
    execution = _new_execution(provider_db)
    partial = _usage_request(counters={"output_tokens": 4})
    record_usage_event(execution.execution_id, partial, _auth=None, db=provider_db)
    summary = get_execution(execution.execution_id, _auth=None, db=provider_db)

    assert summary.usage.input_tokens is None
    assert summary.usage.output_tokens == 4
    assert summary.usage.input_tokens_missing_event_count == 1
    assert summary.usage.provider_usage_completeness == {"incomplete": 1}
    provider_db.commit()

    mismatched_model = _usage_request(exact_model="gpt-6.1-sol")
    with pytest.raises(HTTPException) as mismatch:
        record_usage_event(
            execution.execution_id, mismatched_model, _auth=None, db=provider_db
        )
    assert mismatch.value.status_code == 409
    incompatible_summary = get_execution(execution.execution_id, _auth=None, db=provider_db)
    assert incompatible_summary.usage.input_tokens is None
    assert incompatible_summary.usage.incompatible_usage_event_count == 0


@pytest.mark.parametrize(
    ("provider", "model", "effort"),
    [
        ("openai", "gpt-6-astra", "none"),
        ("google", "gemini-3.8-flash", "minimal"),
        ("openai", "unknown-model", "high"),
    ],
)
def test_execution_reasoning_effort_is_validated_for_exact_model(provider, model, effort):
    with pytest.raises(ValidationError):
        CreateExecutionRequest(
            integration_name="provider-test",
            provider=provider,
            exact_provider_model=model,
            reasoning_effort=effort,
        )


def test_known_exact_model_cannot_be_associated_with_another_provider():
    with pytest.raises(ValidationError):
        CreateExecutionRequest(
            integration_name="provider-test",
            provider="openai",
            exact_provider_model="gemini-3.8-flash",
        )


def test_provider_metadata_migration_round_trips_legacy_usage(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'provider-migration-roundtrip.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    command.upgrade(config, "0003_cancelled_execution_status")
    engine = create_database_engine(database_url)
    execution_id = uuid4()
    event_id = uuid4()
    occurred_at = datetime.now(UTC).isoformat()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO execution_records "
            "(execution_id, workspace_id, integration_name, provider, "
            "exact_provider_model, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                execution_id.hex,
                WORKSPACE_ID,
                "legacy-client",
                "anthropic",
                "test-anthropic-model-v1",
                "completed",
                occurred_at,
            ),
        )
        connection.exec_driver_sql(
            "INSERT INTO usage_events "
            "(event_id, execution_id, workspace_id, provider, exact_provider_model, "
            "input_tokens, output_tokens, source, occurred_at, received_at, "
            "idempotency_identifier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id.hex,
                execution_id.hex,
                WORKSPACE_ID,
                "anthropic",
                "test-anthropic-model-v1",
                12,
                4,
                "legacy-client",
                occurred_at,
                occurred_at,
                str(event_id),
            ),
        )
    command.upgrade(config, "head")
    engine.dispose()

    engine = create_database_engine(database_url)
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT input_tokens, output_tokens, usage_completeness "
            "FROM usage_events WHERE event_id = ?",
            (event_id.hex,),
        ).one()
        assert tuple(row) == (12, 4, "unknown")
    engine.dispose()

    command.downgrade(config, "0003_cancelled_execution_status")
    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT input_tokens, output_tokens, usage_completeness "
            "FROM usage_events WHERE event_id = ?",
            (event_id.hex,),
        ).one()
        assert tuple(row) == (12, 4, "unknown")
        assert connection.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'trg_usage_events_update_immutable'"
        ).first()
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE usage_events SET input_tokens = 13 WHERE event_id = ?",
                (event_id.hex,),
            )
    engine.dispose()


def test_provider_metadata_downgrade_refuses_to_drop_new_usage_fields(
    tmp_path, monkeypatch
):
    database_url = f"sqlite:///{(tmp_path / 'provider-metadata-guard.sqlite').as_posix()}"
    monkeypatch.setenv("CITADEL_DATABASE_URL", database_url)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    execution = ExecutionRecord(
        execution_id=uuid4(),
        workspace_id=WORKSPACE_ID,
        integration_name="provider-client",
        provider="openai",
        exact_provider_model="gpt-6-luna",
        reasoning_effort="high",
        status="completed",
        started_at=datetime.now(UTC),
    )
    event_id = uuid4()
    with Session(engine) as db:
        db.add(execution)
        db.flush()
        db.add(
            UsageEvent(
                event_id=event_id,
                execution_id=execution.execution_id,
                workspace_id=WORKSPACE_ID,
                provider="openai",
                exact_provider_model="gpt-6-luna",
                input_tokens=20,
                output_tokens=11,
                cached_input_tokens=8,
                reasoning_tokens=6,
                total_tokens=31,
                token_semantics="openai.responses.input_includes_cached.v1",
                usage_completeness="complete",
                provider_counters={"input_tokens": 20, "output_tokens": 11},
                source="provider-client",
                occurred_at=datetime.now(UTC),
                idempotency_identifier=str(event_id),
            )
        )
        db.commit()
    engine.dispose()

    with pytest.raises(RuntimeError, match="provider-specific data exists"):
        command.downgrade(config, "0003_cancelled_execution_status")

    command.upgrade(config, "head")
    engine = create_database_engine(database_url)
    with Session(engine) as db:
        event = db.get(UsageEvent, event_id)
        assert event is not None
        assert event.provider_counters == {"input_tokens": 20, "output_tokens": 11}
        assert event.reasoning_tokens == 6
    engine.dispose()
