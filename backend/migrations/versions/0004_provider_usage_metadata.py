"""Preserve provider-specific usage semantics and optional token counters.

Revision ID: 0004_provider_usage_metadata
Revises: 0003_cancelled_execution_status
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_provider_usage_metadata"
down_revision: str | None = "0003_cancelled_execution_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _restore_sqlite_usage_triggers() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    for operation, suffix in (("UPDATE", "update"), ("DELETE", "delete")):
        trigger_name = f"trg_usage_events_{suffix}_immutable"
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
        op.execute(
            f"CREATE TRIGGER {trigger_name} BEFORE {operation} ON usage_events "
            "BEGIN SELECT RAISE(ABORT, 'ledger rows are append-only'); END"
        )


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "execution_records",
        sa.Column("reasoning_effort", sa.String(length=16), nullable=True),
    )
    with op.batch_alter_table("usage_events") as batch_op:
        batch_op.alter_column("input_tokens", existing_type=sa.BigInteger(), nullable=True)
        batch_op.alter_column("output_tokens", existing_type=sa.BigInteger(), nullable=True)
        batch_op.add_column(sa.Column("cached_input_tokens", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("reasoning_tokens", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("tool_use_prompt_tokens", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("total_tokens", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("token_semantics", sa.String(length=128), nullable=True))
        batch_op.add_column(
            sa.Column(
                "usage_completeness",
                sa.String(length=16),
                nullable=False,
                server_default="unknown",
            )
        )
        batch_op.add_column(sa.Column("provider_counters", json_type, nullable=True))
    _restore_sqlite_usage_triggers()

    if op.get_bind().dialect.name == "postgresql":
        op.create_check_constraint(
            "ck_usage_cached_nonnegative",
            "usage_events",
            "cached_input_tokens IS NULL OR cached_input_tokens >= 0",
        )
        op.create_check_constraint(
            "ck_usage_reasoning_nonnegative",
            "usage_events",
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
        )
        op.create_check_constraint(
            "ck_usage_tool_prompt_nonnegative",
            "usage_events",
            "tool_use_prompt_tokens IS NULL OR tool_use_prompt_tokens >= 0",
        )
        op.create_check_constraint(
            "ck_usage_total_nonnegative",
            "usage_events",
            "total_tokens IS NULL OR total_tokens >= 0",
        )
        op.create_check_constraint(
            "ck_usage_event_completeness",
            "usage_events",
            "usage_completeness IN ('complete', 'incomplete', 'unknown')",
        )


def downgrade() -> None:
    bind = op.get_bind()
    null_counters: int = bind.execute(
        sa.text(
            "SELECT count(*) FROM usage_events "
            "WHERE input_tokens IS NULL OR output_tokens IS NULL"
        )
    ).scalar_one()
    if null_counters:
        raise RuntimeError(
            "Cannot downgrade provider usage metadata while nullable token counters exist; "
            "preserve the original records."
        )

    provider_usage_rows: int = bind.execute(
        sa.text(
            "SELECT count(*) FROM usage_events WHERE "
            "cached_input_tokens IS NOT NULL OR reasoning_tokens IS NOT NULL OR "
            "tool_use_prompt_tokens IS NOT NULL OR total_tokens IS NOT NULL OR "
            "token_semantics IS NOT NULL OR usage_completeness != 'unknown' OR "
            "provider_counters IS NOT NULL"
        )
    ).scalar_one()
    executions_with_effort: int = bind.execute(
        sa.text(
            "SELECT count(*) FROM execution_records WHERE reasoning_effort IS NOT NULL"
        )
    ).scalar_one()
    if provider_usage_rows or executions_with_effort:
        raise RuntimeError(
            "Cannot downgrade provider usage metadata while provider-specific data exists; "
            "preserve the original records."
        )

    if bind.dialect.name == "postgresql":
        for name in (
            "ck_usage_event_completeness",
            "ck_usage_total_nonnegative",
            "ck_usage_tool_prompt_nonnegative",
            "ck_usage_reasoning_nonnegative",
            "ck_usage_cached_nonnegative",
        ):
            op.drop_constraint(name, "usage_events", type_="check")

    with op.batch_alter_table("usage_events") as batch_op:
        batch_op.drop_column("provider_counters")
        batch_op.drop_column("usage_completeness")
        batch_op.drop_column("token_semantics")
        batch_op.drop_column("total_tokens")
        batch_op.drop_column("tool_use_prompt_tokens")
        batch_op.drop_column("reasoning_tokens")
        batch_op.drop_column("cached_input_tokens")
        batch_op.alter_column("input_tokens", existing_type=sa.BigInteger(), nullable=False)
        batch_op.alter_column("output_tokens", existing_type=sa.BigInteger(), nullable=False)
    _restore_sqlite_usage_triggers()
    op.drop_column("execution_records", "reasoning_effort")
