"""PostgreSQL-specific migration, type, constraint, and trigger checks."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from database import create_database_engine
from execution_models import ExecutionRecord, PredictionRecord, UsageEvent
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from postgres.conftest import ALEMBIC_CONFIG, downgrade_migration, run_migration

pytestmark = pytest.mark.postgres


def _insert_ledger_rows(engine):
    prediction_id = uuid4()
    execution_id = uuid4()
    event_id = uuid4()
    now = datetime.now(UTC)
    with Session(engine) as db:
        db.add(
            PredictionRecord(
                prediction_id=prediction_id,
                workspace_id="workspace-a",
                created_at=now,
                model_alias="claude-sonnet",
                predicted_low_tokens=10,
                predicted_expected_tokens=20,
                predicted_high_tokens=30,
                prediction_payload_snapshot={"response": {"expected_tokens": 20}},
            )
        )
        db.flush()
        db.add(
            ExecutionRecord(
                execution_id=execution_id,
                prediction_id=prediction_id,
                workspace_id="workspace-a",
                integration_name="postgres-release-gate",
                provider="anthropic",
                exact_provider_model="test-anthropic-model-v1",
                reasoning_effort="high",
                status="started",
                started_at=now,
            )
        )
        db.flush()
        db.add(
            UsageEvent(
                event_id=event_id,
                execution_id=execution_id,
                workspace_id="workspace-a",
                provider="anthropic",
                exact_provider_model="test-anthropic-model-v1",
                input_tokens=120,
                output_tokens=30,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=10,
                cached_input_tokens=30,
                reasoning_tokens=2,
                token_semantics="anthropic.messages.cache_exclusive_input.v1",
                usage_completeness="complete",
                provider_counters={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 10,
                    "thinking_tokens": 2,
                },
                source="postgres_release_gate",
                occurred_at=now,
                idempotency_identifier=str(event_id),
            )
        )
        db.commit()
        assert db.get(PredictionRecord, prediction_id).prediction_id == prediction_id
        assert db.get(ExecutionRecord, execution_id).execution_id == execution_id
        assert db.get(UsageEvent, event_id).event_id == event_id
    return prediction_id, execution_id, event_id


def test_postgres_migration_has_native_types_constraints_indexes_and_triggers(
    postgres_engine,
):
    inspector = inspect(postgres_engine)
    assert {
        "alembic_version",
        "execution_predictions",
        "execution_records",
        "usage_events",
    } <= set(inspector.get_table_names())
    execution_columns = {column["name"] for column in inspector.get_columns("execution_records")}
    assert {
        "expected_usage_event_count",
        "telemetry_provenance",
        "reasoning_effort",
    } <= execution_columns
    usage_columns = {column["name"] for column in inspector.get_columns("usage_events")}
    assert {
        "cached_input_tokens",
        "reasoning_tokens",
        "tool_use_prompt_tokens",
        "total_tokens",
        "token_semantics",
        "usage_completeness",
        "provider_counters",
    } <= usage_columns

    expected_indexes = {
        "ix_execution_predictions_workspace_created",
        "ix_execution_records_workspace_started",
        "ix_usage_events_workspace_execution",
    }
    actual_indexes = {
        index["name"]
        for table in ("execution_predictions", "execution_records", "usage_events")
        for index in inspector.get_indexes(table)
    }
    assert expected_indexes <= actual_indexes

    expected_unique_constraints = {
        "uq_prediction_workspace",
        "uq_execution_workspace",
        "uq_usage_idempotency",
    }
    actual_unique_constraints = {
        constraint["name"]
        for table in ("execution_predictions", "execution_records", "usage_events")
        for constraint in inspector.get_unique_constraints(table)
    }
    assert expected_unique_constraints <= actual_unique_constraints

    expected_foreign_keys = {
        "fk_execution_prediction_workspace",
        "fk_execution_parent_workspace",
        "fk_usage_execution_workspace",
    }
    actual_foreign_keys = {
        foreign_key["name"]
        for table in ("execution_records", "usage_events")
        for foreign_key in inspector.get_foreign_keys(table)
    }
    assert expected_foreign_keys <= actual_foreign_keys

    expected_checks = {
        "ck_execution_status",
        "ck_usage_input_nonnegative",
        "ck_usage_output_nonnegative",
        "ck_usage_cache_creation_nonnegative",
        "ck_usage_cache_read_nonnegative",
        "ck_usage_cached_nonnegative",
        "ck_usage_reasoning_nonnegative",
        "ck_usage_tool_prompt_nonnegative",
        "ck_usage_total_nonnegative",
        "ck_usage_event_completeness",
        "ck_execution_expected_usage_event_count_nonnegative",
        "ck_execution_telemetry_provenance",
    }
    actual_checks = {
        check["name"]
        for table in ("execution_records", "usage_events")
        for check in inspector.get_check_constraints(table)
    }
    assert expected_checks <= actual_checks
    status_check = next(
        check["sqltext"]
        for check in inspector.get_check_constraints("execution_records")
        if check["name"] == "ck_execution_status"
    )
    assert "cancelled" in status_check

    expected_triggers = {
        ("trg_prediction_immutable", "execution_predictions"),
        ("trg_usage_append_only", "usage_events"),
    }
    with postgres_engine.connect() as connection:
        actual_triggers = set(
            connection.execute(
                text(
                    "SELECT t.tgname, table_class.relname "
                    "FROM pg_trigger AS t "
                    "JOIN pg_class AS table_class ON table_class.oid = t.tgrelid "
                    "JOIN pg_namespace AS schema_name "
                    "ON schema_name.oid = table_class.relnamespace "
                    "WHERE NOT t.tgisinternal AND schema_name.nspname = current_schema()"
                )
            ).tuples()
        )
    assert expected_triggers <= actual_triggers

    prediction_id, execution_id, event_id = _insert_ledger_rows(postgres_engine)
    with postgres_engine.connect() as connection:
        native_types = connection.execute(
            text(
                "SELECT "
                "pg_typeof(prediction_id)::text, "
                "pg_typeof(prediction_payload_snapshot)::text "
                "FROM execution_predictions WHERE prediction_id = CAST(:id AS uuid)"
            ),
            {"id": str(prediction_id)},
        ).one()
        usage_uuid_type = connection.execute(
            text(
                "SELECT pg_typeof(event_id)::text FROM usage_events "
                "WHERE event_id = CAST(:id AS uuid)"
            ),
            {"id": str(event_id)},
        ).scalar_one()
    assert native_types == ("uuid", "jsonb")
    assert usage_uuid_type == "uuid"

    for statement, parameter in (
        (
            text(
                "UPDATE execution_predictions SET model_alias = 'changed' "
                "WHERE prediction_id = CAST(:id AS uuid)"
            ),
            str(prediction_id),
        ),
        (
            text("DELETE FROM execution_predictions WHERE prediction_id = CAST(:id AS uuid)"),
            str(prediction_id),
        ),
        (
            text(
                "UPDATE usage_events SET output_tokens = 999 "
                "WHERE event_id = CAST(:id AS uuid)"
            ),
            str(event_id),
        ),
        (
            text("DELETE FROM usage_events WHERE event_id = CAST(:id AS uuid)"),
            str(event_id),
        ),
    ):
        with pytest.raises(DBAPIError):
            with postgres_engine.begin() as connection:
                connection.execute(statement, {"id": parameter})

    with Session(postgres_engine) as db:
        assert db.get(PredictionRecord, prediction_id).model_alias == "claude-sonnet"
        stored_usage = db.get(UsageEvent, event_id)
        assert stored_usage.output_tokens == 30
        assert stored_usage.provider_counters["thinking_tokens"] == 2
        assert stored_usage.token_semantics == "anthropic.messages.cache_exclusive_input.v1"
        assert db.get(ExecutionRecord, execution_id).reasoning_effort == "high"

        db.add(
            ExecutionRecord(
                execution_id=uuid4(),
                prediction_id=prediction_id,
                workspace_id="workspace-b",
                integration_name="postgres-release-gate",
                provider="anthropic",
                exact_provider_model="test-anthropic-model-v1",
                started_at=datetime.now(UTC),
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        execution = db.get(ExecutionRecord, execution_id)
        execution.expected_usage_event_count = -1
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()

        db.add(
            UsageEvent(
                event_id=uuid4(),
                execution_id=execution_id,
                workspace_id="workspace-a",
                provider="anthropic",
                exact_provider_model="test-anthropic-model-v1",
                input_tokens=-1,
                output_tokens=1,
                source="postgres_release_gate",
                occurred_at=datetime.now(UTC),
                idempotency_identifier=str(uuid4()),
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_postgres_migration_downgrade_and_repeat_upgrade_are_safe(
    postgres_migration_database,
):
    database = postgres_migration_database
    run_migration(database.database_url, "head")
    run_migration(database.database_url, "head")

    config = Config(str(ALEMBIC_CONFIG))
    script = ScriptDirectory.from_config(config)
    engine = create_database_engine(database.database_url)
    try:
        with engine.connect() as connection:
            current_heads = MigrationContext.configure(connection).get_current_heads()
            assert set(current_heads) == set(script.get_heads())
            assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

        downgrade_migration(database.database_url, "base")

        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT to_regclass('execution_predictions')")
            ).scalar_one() is None
            assert connection.execute(
                text("SELECT to_regprocedure('citadel_reject_ledger_mutation()')")
            ).scalar_one() is None

        run_migration(database.database_url, "head")
        run_migration(database.database_url, "head")
        with engine.connect() as connection:
            assert set(MigrationContext.configure(connection).get_current_heads()) == set(
                script.get_heads()
            )
    finally:
        engine.dispose()
