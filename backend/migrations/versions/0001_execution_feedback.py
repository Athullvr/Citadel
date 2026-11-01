"""Create the execution feedback ledger.

Revision ID: 0001_execution_feedback
Revises:
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_execution_feedback"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _install_immutability_triggers() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE FUNCTION citadel_reject_ledger_mutation() RETURNS trigger "
            "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'ledger rows are append-only'; END; $$"
        )
        op.execute(
            "CREATE TRIGGER trg_prediction_immutable BEFORE UPDATE OR DELETE "
            "ON execution_predictions FOR EACH ROW "
            "EXECUTE FUNCTION citadel_reject_ledger_mutation()"
        )
        op.execute(
            "CREATE TRIGGER trg_usage_append_only BEFORE UPDATE OR DELETE "
            "ON usage_events FOR EACH ROW EXECUTE FUNCTION citadel_reject_ledger_mutation()"
        )
    elif bind.dialect.name == "sqlite":
        for table, operation, suffix in (
            ("execution_predictions", "UPDATE", "update"),
            ("execution_predictions", "DELETE", "delete"),
            ("usage_events", "UPDATE", "update"),
            ("usage_events", "DELETE", "delete"),
        ):
            trigger_name = f"trg_{table}_{suffix}_immutable"
            op.execute(
                f"CREATE TRIGGER {trigger_name} BEFORE {operation} ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'ledger rows are append-only'); END"
            )


def _remove_immutability_triggers() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_prediction_immutable ON execution_predictions")
        op.execute("DROP TRIGGER IF EXISTS trg_usage_append_only ON usage_events")
        op.execute("DROP FUNCTION IF EXISTS citadel_reject_ledger_mutation()")
    elif bind.dialect.name == "sqlite":
        for table, suffix in (
            ("execution_predictions", "update"),
            ("execution_predictions", "delete"),
            ("usage_events", "update"),
            ("usage_events", "delete"),
        ):
            op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_{suffix}_immutable")


def upgrade() -> None:
    json_type = (
        postgresql.JSONB()
        if op.get_bind().dialect.name == "postgresql"
        else sa.JSON()
    )
    op.create_table(
        "execution_predictions",
        sa.Column("prediction_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_alias", sa.String(length=64), nullable=False),
        sa.Column("model_artifact_version", sa.String(length=128), nullable=True),
        sa.Column("feature_schema_version", sa.String(length=128), nullable=True),
        sa.Column("predicted_low_tokens", sa.BigInteger(), nullable=False),
        sa.Column("predicted_expected_tokens", sa.BigInteger(), nullable=False),
        sa.Column("predicted_high_tokens", sa.BigInteger(), nullable=False),
        sa.Column("estimated_low_cost", sa.Float(), nullable=True),
        sa.Column("estimated_expected_cost", sa.Float(), nullable=True),
        sa.Column("estimated_high_cost", sa.Float(), nullable=True),
        sa.Column("pricing_version", sa.String(length=128), nullable=True),
        sa.Column("prediction_payload_snapshot", json_type, nullable=False),
        sa.Column("task_hash", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("prediction_id", name="pk_execution_predictions"),
        sa.UniqueConstraint("prediction_id", "workspace_id", name="uq_prediction_workspace"),
    )
    op.create_index(
        "ix_execution_predictions_workspace_created",
        "execution_predictions",
        ["workspace_id", "created_at"],
    )

    op.create_table(
        "execution_records",
        sa.Column("execution_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("prediction_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("integration_name", sa.String(length=128), nullable=False),
        sa.Column("integration_version", sa.String(length=64), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("exact_provider_model", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="started", nullable=False),
        sa.Column("external_run_id", sa.String(length=256), nullable=True),
        sa.Column("parent_execution_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("is_synthetic", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('started', 'completed', 'failed')", name="ck_execution_status"
        ),
        sa.ForeignKeyConstraint(
            ["parent_execution_id", "workspace_id"],
            ["execution_records.execution_id", "execution_records.workspace_id"],
            name="fk_execution_parent_workspace",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id", "workspace_id"],
            ["execution_predictions.prediction_id", "execution_predictions.workspace_id"],
            name="fk_execution_prediction_workspace",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("execution_id", name="pk_execution_records"),
        sa.UniqueConstraint("execution_id", "workspace_id", name="uq_execution_workspace"),
    )
    op.create_index(
        "ix_execution_records_workspace_started",
        "execution_records",
        ["workspace_id", "started_at"],
    )

    op.create_table(
        "usage_events",
        sa.Column("event_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("execution_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("exact_provider_model", sa.String(length=128), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_creation_input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("cache_read_input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("idempotency_identifier", sa.String(length=36), nullable=False),
        sa.CheckConstraint("input_tokens >= 0", name="ck_usage_input_nonnegative"),
        sa.CheckConstraint("output_tokens >= 0", name="ck_usage_output_nonnegative"),
        sa.CheckConstraint(
            "cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0",
            name="ck_usage_cache_creation_nonnegative",
        ),
        sa.CheckConstraint(
            "cache_read_input_tokens IS NULL OR cache_read_input_tokens >= 0",
            name="ck_usage_cache_read_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["execution_id", "workspace_id"],
            ["execution_records.execution_id", "execution_records.workspace_id"],
            name="fk_usage_execution_workspace",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id", name="pk_usage_events"),
        sa.UniqueConstraint(
            "workspace_id", "execution_id", "idempotency_identifier", name="uq_usage_idempotency"
        ),
    )
    op.create_index(
        "ix_usage_events_workspace_execution", "usage_events", ["workspace_id", "execution_id"]
    )
    _install_immutability_triggers()


def downgrade() -> None:
    _remove_immutability_triggers()
    op.drop_index("ix_usage_events_workspace_execution", table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_index("ix_execution_records_workspace_started", table_name="execution_records")
    op.drop_table("execution_records")
    op.drop_index("ix_execution_predictions_workspace_created", table_name="execution_predictions")
    op.drop_table("execution_predictions")
