"""Authentication helpers — user CRUD, password hashing, JWT tokens.

Depends on the same PostgreSQL database used by the memory store.
The users table is created automatically on first use.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
import psycopg2
from dotenv import load_dotenv

# Read env from the project root so Docker and local dev share one .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (_PROJECT_ROOT / ".env").exists():
    load_dotenv(_PROJECT_ROOT / ".env")

SECRET_KEY = os.getenv("NEXTAUTH_SECRET", "change-me-to-a-random-string")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def _get_connection():
    """Create a PostgreSQL connection and ensure the users table exists."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "autoagent_db"),
        user=os.getenv("POSTGRES_USER", "autoagent"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    _ensure_users_table(conn)
    return conn


def _ensure_users_table(conn):
    """Create the users table if it does not already exist.

    We use gen_random_uuid() which requires the pgcrypto extension.
    If pgcrypto is unavailable, we fall back to a text ID filled by Python.
    """
    with conn.cursor() as cur:
        # Try to create the extension first (idempotent)
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            conn.commit()
            use_gen_random = True
        except Exception:
            conn.rollback()
            use_gen_random = False

        if use_gen_random:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email      TEXT UNIQUE NOT NULL,
                    name       TEXT NOT NULL,
                    password   TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        else:
            # Fallback: use TEXT for id, filled by Python's uuid4()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id         TEXT PRIMARY KEY,
                    email      TEXT UNIQUE NOT NULL,
                    name       TEXT NOT NULL,
                    password   TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )

    conn.commit()


def _hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _create_token(user_id: str, email: str) -> str:
    """Create a signed JWT access token."""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ── Public API ─────────────────────────────────────────


def create_user(email: str, name: str, password: str) -> dict:
    """Register a new user. Returns dict with id, email, name.

    Raises:
        ValueError: if email already exists.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                raise ValueError("Email already registered")

            user_id = str(uuid.uuid4())
            hashed_pw = _hash_password(password)
            cur.execute(
                "INSERT INTO users (id, email, name, password) VALUES (%s, %s, %s, %s) RETURNING id, email, name, created_at",
                (user_id, email, name, hashed_pw),
            )
            row = cur.fetchone()
        conn.commit()

        return {
            "id": str(row[0]),
            "email": row[1],
            "name": row[2],
            "created_at": str(row[3]),
        }
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> dict:
    """Verify credentials and return a dict with token + user info.

    Raises:
        ValueError: if credentials are invalid.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name, password FROM users WHERE email = %s", (email,))
            row = cur.fetchone()

        if not row or not _verify_password(password, row[3]):
            raise ValueError("Invalid email or password")

        user_id, user_email, user_name = str(row[0]), row[1], row[2]
        token = _create_token(user_id, user_email)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"id": user_id, "email": user_email, "name": user_name},
        }
    finally:
        conn.close()


def get_user_by_id(user_id: str) -> dict | None:
    """Fetch a user by ID (returns None if not found)."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, email, name FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()

        if not row:
            return None
        return {"id": str(row[0]), "email": row[1], "name": row[2]}
    finally:
        conn.close()
