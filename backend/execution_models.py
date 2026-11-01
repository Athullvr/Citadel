"""SQLAlchemy records for predictions, executions, and provider usage events."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from database import Base
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC-aware timestamps consistently on PostgreSQL and SQLite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, _dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Timezone-aware timestamps are required")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, _dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


SnapshotJSON = JSON().with_variant(JSONB(), "postgresql")


class PredictionRecord(Base):
    __tablename__ = "execution_predictions"
    __table_args__ = (
        UniqueConstraint("prediction_id", "workspace_id", name="uq_prediction_workspace"),
        Index("ix_execution_predictions_workspace_created", "workspace_id", "created_at"),
    )

    prediction_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    model_artifact_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    feature_schema_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    predicted_low_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    predicted_expected_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    predicted_high_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    estimated_low_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_expected_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_high_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    pricing_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prediction_payload_snapshot: Mapped[dict[str, Any]] = mapped_column(
        SnapshotJSON, nullable=False
    )
    task_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ExecutionRecord(Base):
    __tablename__ = "execution_records"
    __table_args__ = (
        ForeignKeyConstraint(
            ["prediction_id", "workspace_id"],
            ["execution_predictions.prediction_id", "execution_predictions.workspace_id"],
            name="fk_execution_prediction_workspace",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["parent_execution_id", "workspace_id"],
            ["execution_records.execution_id", "execution_records.workspace_id"],
            name="fk_execution_parent_workspace",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("execution_id", "workspace_id", name="uq_execution_workspace"),
        CheckConstraint("status IN ('started', 'completed', 'failed')", name="ck_execution_status"),
        Index("ix_execution_records_workspace_started", "workspace_id", "started_at"),
    )

    execution_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    prediction_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    integration_name: Mapped[str] = mapped_column(String(128), nullable=False)
    integration_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    exact_provider_model: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="started")
    external_run_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    parent_execution_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class UsageEvent(Base):
    __tablename__ = "usage_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["execution_id", "workspace_id"],
            ["execution_records.execution_id", "execution_records.workspace_id"],
            name="fk_usage_execution_workspace",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "workspace_id", "execution_id", "idempotency_identifier", name="uq_usage_idempotency"
        ),
        CheckConstraint("input_tokens >= 0", name="ck_usage_input_nonnegative"),
        CheckConstraint("output_tokens >= 0", name="ck_usage_output_nonnegative"),
        CheckConstraint(
            "cache_creation_input_tokens IS NULL OR cache_creation_input_tokens >= 0",
            name="ck_usage_cache_creation_nonnegative",
        ),
        CheckConstraint(
            "cache_read_input_tokens IS NULL OR cache_read_input_tokens >= 0",
            name="ck_usage_cache_read_nonnegative",
        ),
        Index("ix_usage_events_workspace_execution", "workspace_id", "execution_id"),
    )

    event_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    execution_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    exact_provider_model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    idempotency_identifier: Mapped[str] = mapped_column(String(36), nullable=False)
