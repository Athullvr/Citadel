"""Add execution telemetry completeness and provenance metadata.

Revision ID: 0002_evaluation_metadata
Revises: 0001_execution_feedback
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_evaluation_metadata"
down_revision: str | None = "0001_execution_feedback"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "execution_records",
        sa.Column("expected_usage_event_count", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "execution_records",
        sa.Column("telemetry_provenance", sa.String(length=32), nullable=True),
    )
    if op.get_bind().dialect.name == "postgresql":
        op.create_check_constraint(
            "ck_execution_expected_usage_event_count_nonnegative",
            "execution_records",
            "expected_usage_event_count IS NULL OR expected_usage_event_count >= 0",
        )
        op.create_check_constraint(
            "ck_execution_telemetry_provenance",
            "execution_records",
            "telemetry_provenance IS NULL OR telemetry_provenance IN "
            "('verified_provider', 'unverified_submitted', 'synthetic_benchmark')",
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint(
            "ck_execution_telemetry_provenance", "execution_records", type_="check"
        )
        op.drop_constraint(
            "ck_execution_expected_usage_event_count_nonnegative",
            "execution_records",
            type_="check",
        )
    op.drop_column("execution_records", "telemetry_provenance")
    op.drop_column("execution_records", "expected_usage_event_count")
