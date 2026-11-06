"""Persist immutable prediction identity and link it through telemetry.

Revision ID: 0005_prediction_identity
Revises: 0004_provider_usage_metadata
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_prediction_identity"
down_revision: str | None = "0004_provider_usage_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_ledger_triggers() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS trg_prediction_immutable ON execution_predictions")
        op.execute("DROP TRIGGER IF EXISTS trg_usage_append_only ON usage_events")
    elif bind.dialect.name == "sqlite":
        for table, suffix in (
            ("execution_predictions", "update"),
            ("execution_predictions", "delete"),
            ("usage_events", "update"),
            ("usage_events", "delete"),
        ):
            op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_{suffix}_immutable")


def _restore_ledger_triggers() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE TRIGGER trg_prediction_immutable BEFORE UPDATE OR DELETE "
            "ON execution_predictions FOR EACH ROW "
            "EXECUTE FUNCTION citadel_reject_ledger_mutation()"
        )
        op.execute(
            "CREATE TRIGGER trg_usage_append_only BEFORE UPDATE OR DELETE "
            "ON usage_events FOR EACH ROW "
            "EXECUTE FUNCTION citadel_reject_ledger_mutation()"
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


def upgrade() -> None:
    _drop_ledger_triggers()

    with op.batch_alter_table("execution_predictions") as batch_op:
        batch_op.add_column(sa.Column("provider", sa.String(length=64), nullable=True))
        batch_op.add_column(
            sa.Column("exact_provider_model", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("identity_schema_version", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "identity_status",
                sa.String(length=32),
                nullable=False,
                server_default="legacy_unresolved",
            )
        )
        batch_op.add_column(
            sa.Column("token_target_semantics", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("estimator_artifact_version", sa.String(length=256), nullable=True)
        )
        batch_op.add_column(
            sa.Column("estimator_artifact_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("prediction_identity_hash", sa.String(length=64), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_prediction_identity_status",
            "identity_status IN ('resolved', 'legacy_unresolved')",
        )

    with op.batch_alter_table("execution_records") as batch_op:
        batch_op.alter_column(
            "exact_provider_model",
            existing_type=sa.String(length=128),
            nullable=True,
        )
        batch_op.add_column(
            sa.Column("identity_schema_version", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "identity_status",
                sa.String(length=32),
                nullable=False,
                server_default="unlinked",
            )
        )
        batch_op.add_column(
            sa.Column("token_target_semantics", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("prediction_identity_hash", sa.String(length=64), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_execution_identity_status",
            "identity_status IN ('resolved', 'legacy_unresolved', 'unlinked')",
        )

    op.execute(
        sa.text(
            "UPDATE execution_records SET identity_status = 'legacy_unresolved' "
            "WHERE prediction_id IS NOT NULL"
        )
    )

    with op.batch_alter_table("usage_events") as batch_op:
        batch_op.alter_column(
            "exact_provider_model",
            existing_type=sa.String(length=128),
            nullable=True,
        )
        batch_op.add_column(
            sa.Column("identity_schema_version", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "identity_status",
                sa.String(length=32),
                nullable=False,
                server_default="unlinked",
            )
        )
        batch_op.add_column(
            sa.Column("token_target_semantics", sa.String(length=128), nullable=True)
        )
        batch_op.add_column(
            sa.Column("prediction_identity_hash", sa.String(length=64), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_usage_identity_status",
            "identity_status IN ('resolved', 'legacy_unresolved', 'unlinked')",
        )

    op.execute(
        sa.text(
            "UPDATE usage_events SET identity_status = ("
            "SELECT execution_records.identity_status FROM execution_records "
            "WHERE execution_records.execution_id = usage_events.execution_id "
            "AND execution_records.workspace_id = usage_events.workspace_id)"
        )
    )

    _restore_ledger_triggers()


def downgrade() -> None:
    bind = op.get_bind()
    identity_rows = bind.execute(
        sa.text(
            "SELECT "
            "(SELECT count(*) FROM execution_predictions WHERE "
            "provider IS NOT NULL OR exact_provider_model IS NOT NULL OR "
            "identity_schema_version IS NOT NULL OR token_target_semantics IS NOT NULL OR "
            "estimator_artifact_version IS NOT NULL OR estimator_artifact_sha256 IS NOT NULL OR "
            "prediction_identity_hash IS NOT NULL) + "
            "(SELECT count(*) FROM execution_records WHERE "
            "identity_schema_version IS NOT NULL OR token_target_semantics IS NOT NULL OR "
            "prediction_identity_hash IS NOT NULL OR exact_provider_model IS NULL) + "
            "(SELECT count(*) FROM usage_events WHERE "
            "identity_schema_version IS NOT NULL OR token_target_semantics IS NOT NULL OR "
            "prediction_identity_hash IS NOT NULL OR exact_provider_model IS NULL)"
        )
    ).scalar_one()
    if identity_rows:
        raise RuntimeError(
            "Cannot downgrade prediction identity while identity-bearing or unresolved-model "
            "records exist; preserve the immutable identity chain."
        )

    _drop_ledger_triggers()

    with op.batch_alter_table("usage_events") as batch_op:
        batch_op.drop_constraint("ck_usage_identity_status", type_="check")
        batch_op.drop_column("prediction_identity_hash")
        batch_op.drop_column("token_target_semantics")
        batch_op.drop_column("identity_status")
        batch_op.drop_column("identity_schema_version")
        batch_op.alter_column(
            "exact_provider_model",
            existing_type=sa.String(length=128),
            nullable=False,
        )
    with op.batch_alter_table("execution_records") as batch_op:
        batch_op.drop_constraint("ck_execution_identity_status", type_="check")
        batch_op.drop_column("prediction_identity_hash")
        batch_op.drop_column("token_target_semantics")
        batch_op.drop_column("identity_status")
        batch_op.drop_column("identity_schema_version")
        batch_op.alter_column(
            "exact_provider_model",
            existing_type=sa.String(length=128),
            nullable=False,
        )
    with op.batch_alter_table("execution_predictions") as batch_op:
        batch_op.drop_constraint("ck_prediction_identity_status", type_="check")
        batch_op.drop_column("prediction_identity_hash")
        batch_op.drop_column("estimator_artifact_sha256")
        batch_op.drop_column("estimator_artifact_version")
        batch_op.drop_column("token_target_semantics")
        batch_op.drop_column("identity_status")
        batch_op.drop_column("identity_schema_version")
        batch_op.drop_column("exact_provider_model")
        batch_op.drop_column("provider")

    _restore_ledger_triggers()
