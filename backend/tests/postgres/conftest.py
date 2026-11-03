"""Disposable PostgreSQL fixtures for release-gate integration tests."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from database import create_database_engine, normalize_database_url
from sqlalchemy import Engine, text
from sqlalchemy.engine import URL, make_url

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
ALEMBIC_CONFIG = BACKEND_DIR / "alembic.ini"
LOCAL_DATABASE_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class PostgreSQLDatabase:
    """A uniquely named schema inside a disposable local test database."""

    schema: str
    admin_url: str
    database_url: str


@dataclass
class PostgreSQLWorkers:
    """Two independent FastAPI processes sharing one PostgreSQL schema."""

    base_urls: tuple[str, str]
    api_key: str
    workspace_id: str
    processes: tuple[subprocess.Popen, subprocess.Popen]

    def request(
        self,
        worker_index: int,
        method: str,
        path: str,
        *,
        payload: dict | None = None,
        timeout: float = 45.0,
    ) -> httpx.Response:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            return client.request(
                method,
                f"{self.base_urls[worker_index]}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )


@pytest.fixture(scope="session")
def postgres_test_url() -> Iterator[URL]:
    """Require an explicit loopback-only test URL; never infer DATABASE_URL."""
    raw_url = os.environ.get("CITADEL_TEST_POSTGRES_URL", "").strip()
    if not raw_url:
        pytest.skip("PostgreSQL integration tests require CITADEL_TEST_POSTGRES_URL")

    try:
        import psycopg
    except ImportError as exc:
        pytest.fail(
            f"PostgreSQL tests require psycopg 3 from backend/requirements.txt ({exc})",
            pytrace=False,
        )

    database_url = make_url(normalize_database_url(raw_url))
    if database_url.drivername != "postgresql+psycopg":
        pytest.fail("CITADEL_TEST_POSTGRES_URL must use the psycopg 3 driver")
    if database_url.host not in LOCAL_DATABASE_HOSTS:
        pytest.fail("PostgreSQL tests only accept a loopback host to prevent production access")
    if not database_url.database or not database_url.database.lower().endswith("_test"):
        pytest.fail("PostgreSQL test database name must end with '_test'")

    dsn = database_url.set(drivername="postgresql").render_as_string(hide_password=False)
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=2) as connection:
                version = connection.execute("SHOW server_version").fetchone()[0]
            print(f"PostgreSQL release-gate server version: {version}")
            break
        except psycopg.OperationalError as exc:
            last_error = exc
            time.sleep(1)
    else:
        error_type = type(last_error).__name__ if last_error else "unknown error"
        pytest.fail(f"PostgreSQL test service did not become ready ({error_type})")

    yield database_url


def _schema_url(base_url: URL, schema: str, application_name: str | None = None) -> URL:
    query = dict(base_url.query)
    search_path = f"-csearch_path={schema}"
    existing_options = query.get("options")
    query["options"] = f"{existing_options} {search_path}" if existing_options else search_path
    if application_name:
        query["application_name"] = application_name
    return base_url.set(query=query)


def _create_test_schema(base_url: URL) -> PostgreSQLDatabase:
    schema = f"citadel_test_{uuid4().hex}"
    admin_url = base_url.render_as_string(hide_password=False)
    admin_engine = create_database_engine(admin_url)
    try:
        with admin_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    finally:
        admin_engine.dispose()
    scoped_url = _schema_url(base_url, schema).render_as_string(hide_password=False)
    return PostgreSQLDatabase(schema, admin_url, scoped_url)


def _drop_test_schema(database: PostgreSQLDatabase) -> None:
    admin_engine = create_database_engine(database.admin_url)
    try:
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{database.schema}" CASCADE'))
    finally:
        admin_engine.dispose()


@contextmanager
def _migration_database_url(database_url: str) -> Iterator[None]:
    previous_url = os.environ.get("CITADEL_DATABASE_URL")
    os.environ["CITADEL_DATABASE_URL"] = database_url
    try:
        yield
    finally:
        if previous_url is None:
            os.environ.pop("CITADEL_DATABASE_URL", None)
        else:
            os.environ["CITADEL_DATABASE_URL"] = previous_url


def run_migration(database_url: str, revision: str) -> None:
    with _migration_database_url(database_url):
        command.upgrade(Config(str(ALEMBIC_CONFIG)), revision)


def downgrade_migration(database_url: str, revision: str) -> None:
    with _migration_database_url(database_url):
        command.downgrade(Config(str(ALEMBIC_CONFIG)), revision)


@pytest.fixture(scope="session")
def postgres_database(postgres_test_url: URL) -> Iterator[PostgreSQLDatabase]:
    """Migrate a unique schema and drop it after every PostgreSQL test finishes."""
    database = _create_test_schema(postgres_test_url)
    try:
        run_migration(database.database_url, "head")
        yield database
    finally:
        _drop_test_schema(database)


@pytest.fixture
def postgres_migration_database(postgres_test_url: URL) -> Iterator[PostgreSQLDatabase]:
    """Provide an empty schema for upgrade/downgrade verification."""
    database = _create_test_schema(postgres_test_url)
    try:
        yield database
    finally:
        _drop_test_schema(database)


@pytest.fixture(scope="session")
def postgres_engine(postgres_database: PostgreSQLDatabase) -> Iterator[Engine]:
    engine = create_database_engine(postgres_database.database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_log(log_file) -> str:
    log_file.flush()
    log_file.seek(0)
    return log_file.read().decode("utf-8", errors="replace")


@pytest.fixture(scope="session")
def postgres_workers(
    postgres_database: PostgreSQLDatabase,
) -> Iterator[PostgreSQLWorkers]:
    """Start two independent API processes with separate pools and the same DB."""
    api_key = secrets.token_urlsafe(32)
    workspace_id = "citadel-postgres-release-gate"
    env = os.environ.copy()
    env["DATA_COLLECTION_DIR"] = str(DATA_COLLECTION_DIR)
    env["CITADEL_EXECUTION_WORKSPACE_ID"] = workspace_id
    env["CITADEL_API_KEY"] = api_key
    env["API_KEY"] = ""
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(BACKEND_DIR), str(DATA_COLLECTION_DIR), existing_pythonpath) if part
    )

    processes: list[subprocess.Popen] = []
    log_files = []
    base_urls: list[str] = []
    try:
        for worker_index in range(2):
            port = _free_loopback_port()
            base_urls.append(f"http://127.0.0.1:{port}")
            env["CITADEL_DATABASE_URL"] = _schema_url(
                make_url(postgres_database.admin_url),
                postgres_database.schema,
                application_name=f"citadel_pg_gate_worker_{worker_index}",
            ).render_as_string(hide_password=False)
            log_file = tempfile.TemporaryFile(mode="w+b")
            log_files.append(log_file)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--workers",
                    "1",
                    "--log-level",
                    "warning",
                ],
                cwd=BACKEND_DIR,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)

            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(
                        f"PostgreSQL API worker {worker_index} exited during startup:\n"
                        f"{_read_log(log_file)}"
                    )
                try:
                    response = httpx.get(
                        f"{base_urls[-1]}/api/health", timeout=1, trust_env=False
                    )
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail(
                    f"PostgreSQL API worker {worker_index} did not become ready:\n"
                    f"{_read_log(log_file)}"
                )

        yield PostgreSQLWorkers(
            (base_urls[0], base_urls[1]),
            api_key,
            workspace_id,
            (processes[0], processes[1]),
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for log_file in log_files:
            log_file.close()
