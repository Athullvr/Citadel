"""Database configuration for the execution feedback ledger.

The inference API remains usable without a database. Execution routes require
``CITADEL_DATABASE_URL`` and return 503 when it is not configured or reachable.
"""

import os
from collections.abc import Generator
from functools import lru_cache

from fastapi import HTTPException, status
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    """Base metadata for the execution feedback tables."""


def normalize_database_url(database_url: str) -> str:
    """Prefer psycopg 3 when a conventional PostgreSQL URL is supplied."""
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


def _is_in_memory_sqlite(database_url: str) -> bool:
    normalized_url = normalize_database_url(database_url)
    return normalized_url.startswith("sqlite") and (
        normalized_url == "sqlite://" or ":memory:" in normalized_url
    )


def create_database_engine(database_url: str) -> Engine:
    """Create a process-local SQLAlchemy engine for PostgreSQL or SQLite."""
    normalized_url = normalize_database_url(database_url)
    options: dict[str, object] = {"pool_pre_ping": True}
    if normalized_url.startswith("sqlite:"):
        if _is_in_memory_sqlite(normalized_url):
            raise ValueError(
                "Execution feedback requires file-backed SQLite or PostgreSQL; "
                "in-memory storage is not durable."
            )
        options["connect_args"] = {"check_same_thread": False, "timeout": 30}

    engine = create_engine(normalized_url, **options)
    if normalized_url.startswith("sqlite:"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache(maxsize=4)
def _engine_for_url(database_url: str) -> Engine:
    return create_database_engine(database_url)


def get_db() -> Generator[Session, None, None]:
    """Yield a connected session or fail closed when tracking storage is absent."""
    database_url = os.environ.get("CITADEL_DATABASE_URL", "").strip()
    if not database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution feedback storage is not configured.",
        )
    if _is_in_memory_sqlite(database_url):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution feedback requires file-backed SQLite or PostgreSQL storage.",
        )

    try:
        engine = _engine_for_url(database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        with Session(engine) as session:
            try:
                yield session
            except SQLAlchemyError as exc:
                session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Execution feedback storage is unavailable.",
                ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Execution feedback storage is unavailable.",
        ) from exc
