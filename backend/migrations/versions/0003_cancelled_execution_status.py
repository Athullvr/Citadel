"""Allow executions to end in the cancelled state.

Revision ID: 0003_cancelled_execution_status
Revises: 0002_evaluation_metadata
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_cancelled_execution_status"
down_revision: str | None = "0002_evaluation_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_OLD_STATUS_CHECK = "status IN ('started', 'completed', 'failed')"
_NEW_STATUS_CHECK = "status IN ('started', 'completed', 'failed', 'cancelled')"


def _replace_status_check(expression: str) -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("execution_records") as batch_op:
            batch_op.drop_constraint("ck_execution_status", type_="check")
            batch_op.create_check_constraint("ck_execution_status", expression)
    else:
        op.drop_constraint("ck_execution_status", "execution_records", type_="check")
        op.create_check_constraint("ck_execution_status", "execution_records", expression)


def upgrade() -> None:
    _replace_status_check(_NEW_STATUS_CHECK)


def downgrade() -> None:
    cancelled_count = op.get_bind().execute(
        sa.text("SELECT count(*) FROM execution_records WHERE status = 'cancelled'")
    ).scalar_one()
    if cancelled_count:
        raise RuntimeError(
            "Cannot downgrade while cancelled executions exist; preserve their terminal status."
        )
    _replace_status_check(_OLD_STATUS_CHECK)
