"""Shared fixtures for AutoAgent tests.

Provides:
- ``mock_llm``: patches ChatGroq / ChatOpenAI so tests never hit real APIs.
- ``mock_embeddings``: patches embed_text to return a fixed 384-dim vector,
  avoiding the 80MB model download.
"""

import os
from unittest.mock import patch, MagicMock

import pytest
from dotenv import load_dotenv

# Load .env so tests can read POSTGRES_* / API keys when needed
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


@pytest.fixture
def mock_llm():
    """Mock the LLM so no real API calls are made during tests.

    Both ChatGroq and ChatOpenAI are patched to return a MagicMock response
    whose ``content`` is ``"TOOL: finish\\nREASON: Test"`` and ``usage_metadata``
    is None.  Tests that need different responses can re-patch per-test.
    """
    mock_response = MagicMock()
    mock_response.content = "TOOL: finish\nREASON: Test response"
    mock_response.usage_metadata = None

    with patch("backend.agent.graph.ChatGroq") as mock_groq, \
         patch("backend.agent.graph.ChatOpenAI") as mock_openai:
        mock_groq.return_value.invoke.return_value = mock_response
        mock_openai.return_value.invoke.return_value = mock_response
        yield mock_groq, mock_openai


@pytest.fixture
def mock_embeddings():
    """Mock the embedding model to return a fixed 384-dim vector.

    Avoids downloading the ~80MB ``all-MiniLM-L6-v2`` model in tests that
    only need to verify dedup/TTL/importance logic.
    """
    fixed_vector = [0.01] * 384
    with patch("backend.memory.store.embed_text", return_value=fixed_vector):
        yield fixed_vector


@pytest.fixture
def db_conn():
    """Provide a PostgreSQL connection for DB-dependent tests.

    Creates a fresh ``autoagent_test`` database, runs all SQL migration
    files from ``docker/``, and yields a connection.  The database is
    dropped on teardown so tests are fully isolated.

    Set ``POSTGRES_HOST``, ``POSTGRES_PORT``, ``POSTGRES_USER``,
    ``POSTGRES_PASSWORD`` env vars to override defaults.
    """
    import psycopg2
    from pgvector.psycopg2 import register_vector

    test_db = os.getenv("POSTGRES_TEST_DB", "autoagent_test")
    user = os.getenv("POSTGRES_USER", "autoagent")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    main_db = os.getenv("POSTGRES_DB", "autoagent_db")

    def _connect(dbname):
        return psycopg2.connect(
            host=host, port=port, dbname=dbname,
            user=user, password=password,
        )

    # Create test database from the main one
    admin_conn = _connect(main_db)
    admin_conn.autocommit = True
    try:
        with admin_conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {test_db}")
            cur.execute(f"CREATE DATABASE {test_db}")
    finally:
        admin_conn.close()

    # Connect to test DB, run migrations (creates the vector extension), then
    # register the vector type — must happen in that order, or register_vector
    # fails because the extension it needs doesn't exist yet.
    test_conn = _connect(test_db)
    try:
        _run_migrations(test_conn)
        register_vector(test_conn)
    except Exception:
        test_conn.close()
        # Clean up on setup failure
        ac = _connect(main_db)
        ac.autocommit = True
        with ac.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {test_db}")
        ac.close()
        raise

    # Point all code at the test DB for the duration of the test
    old_db = os.environ.get("POSTGRES_DB")
    os.environ["POSTGRES_DB"] = test_db

    yield test_conn

    # Teardown: restore env, close conn, drop test DB
    if old_db is not None:
        os.environ["POSTGRES_DB"] = old_db
    elif "POSTGRES_DB" in os.environ:
        del os.environ["POSTGRES_DB"]

    test_conn.close()

    ac = _connect(main_db)
    ac.autocommit = True
    try:
        with ac.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {test_db}")
    finally:
        ac.close()


def _run_migrations(conn):
    """Run all SQL migration files from ``docker/`` against *conn*."""
    docker_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docker")
    migrations = sorted(f for f in os.listdir(docker_dir) if f.endswith(".sql"))
    with conn.cursor() as cur:
        for mig in migrations:
            path = os.path.join(docker_dir, mig)
            with open(path) as f:
                sql = f.read()
            if sql.strip():
                cur.execute(sql)
    conn.commit()
