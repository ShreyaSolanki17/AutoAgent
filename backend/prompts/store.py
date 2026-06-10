"""Prompt versioning store — manage versioned prompts in PostgreSQL.

Lifecycle: draft → staging → production → archived

Follows the same _get_connection() pattern as backend/memory/store.py.
"""

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _get_connection():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "autoagent_db"),
        user=os.getenv("POSTGRES_USER", "autoagent"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def get_active_prompt(prompt_key: str, environment: str = "production") -> str | None:
    """Retrieve the active prompt text for a given key and environment.

    For staging environment: checks 'staging' first, falls back to 'production'.
    For production environment: only checks 'production'.

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM', 'OBSERVER_SYSTEM'
        environment: 'production' or 'staging'

    Returns:
        The prompt text, or None if no active version found.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        if environment == "staging":
            # Staging: prefer staging-tagged, fall back to production
            cur.execute(
                """SELECT content FROM prompt_versions
                   WHERE prompt_key = %s AND status IN ('staging', 'production')
                   ORDER BY status DESC  -- 'staging' > 'production' alphabetically
                   LIMIT 1""",
                (prompt_key,),
            )
        else:
            # Production: only production-tagged
            cur.execute(
                """SELECT content FROM prompt_versions
                   WHERE prompt_key = %s AND status = 'production'
                   LIMIT 1""",
                (prompt_key,),
            )

        row = cur.fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def create_prompt_version(
    prompt_key: str,
    content: str,
    change_summary: str = "",
    created_by: str = "system",
) -> int:
    """Create a new prompt version as 'draft'.

    Auto-increments the version number based on the current max for this key.

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM'
        content: The new prompt text
        change_summary: Human-readable description of the change
        created_by: Who created this version

    Returns:
        The new version number.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        # Get next version number for this key
        cur.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM prompt_versions WHERE prompt_key = %s",
            (prompt_key,),
        )
        new_version = cur.fetchone()[0]

        cur.execute(
            """INSERT INTO prompt_versions (prompt_key, version, content, status, change_summary, created_by)
               VALUES (%s, %s, %s, 'draft', %s, %s)""",
            (prompt_key, new_version, content, change_summary, created_by),
        )
        conn.commit()
        return new_version
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def promote_prompt(prompt_key: str, version: int, target_status: str) -> bool:
    """Promote a prompt version to a new status.

    Valid transitions:
    - draft → staging
    - staging → production (auto-archives current production)

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM'
        version: The version number to promote
        target_status: 'staging' or 'production'

    Returns:
        True if promotion succeeded, False otherwise.
    """
    if target_status not in ("staging", "production"):
        return False

    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        # Verify the version exists and is in the right current status
        cur.execute(
            "SELECT status FROM prompt_versions WHERE prompt_key = %s AND version = %s",
            (prompt_key, version),
        )
        row = cur.fetchone()
        if not row:
            return False

        current_status = row[0]

        # Validate transition
        if current_status == "draft" and target_status == "staging":
            pass  # valid
        elif current_status == "staging" and target_status == "production":
            pass  # valid
        else:
            return False  # invalid transition

        # If promoting to production, archive current production
        if target_status == "production":
            cur.execute(
                """UPDATE prompt_versions SET status = 'archived'
                   WHERE prompt_key = %s AND status = 'production'""",
                (prompt_key,),
            )

        # Promote the version
        cur.execute(
            """UPDATE prompt_versions SET status = %s
               WHERE prompt_key = %s AND version = %s""",
            (target_status, prompt_key, version),
        )
        conn.commit()
        return True
    except Exception:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def rollback_prompt(prompt_key: str) -> bool:
    """Rollback to the previous production version.

    Archives the current production prompt and re-activates the most recent
    archived version (if any).

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM'

    Returns:
        True if rollback succeeded, False otherwise.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        # Check there IS a current production to rollback from
        cur.execute(
            "SELECT version FROM prompt_versions WHERE prompt_key = %s AND status = 'production'",
            (prompt_key,),
        )
        row = cur.fetchone()
        if not row:
            return False

        # Archive current production
        cur.execute(
            """UPDATE prompt_versions SET status = 'archived'
               WHERE prompt_key = %s AND status = 'production'""",
            (prompt_key,),
        )

        # Re-activate the most recent archived version
        cur.execute(
            """UPDATE prompt_versions SET status = 'production'
               WHERE prompt_key = %s AND version = (
                   SELECT MAX(version) FROM prompt_versions
                   WHERE prompt_key = %s AND status = 'archived'
               )""",
            (prompt_key, prompt_key),
        )

        conn.commit()
        return True
    except Exception:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


# ── Query helpers ──────────────────────────────────────────

def list_versions(prompt_key: str, limit: int = 50) -> list[dict]:
    """List version history for a prompt key, newest first.

    Returns:
        List of dicts with keys: id, prompt_key, version, content, status,
        change_summary, created_by, created_at, promoted_at
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, prompt_key, version, content, status, change_summary,
                      created_by, created_at, promoted_at
               FROM prompt_versions
               WHERE prompt_key = %s
               ORDER BY version DESC
               LIMIT %s""",
            (prompt_key, limit),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


def list_all_active(environment: str = "production") -> dict[str, str]:
    """Get all active prompts for an environment.

    For staging: returns staging-tagged prompts, falling back to production for keys
    that have no staging version.

    Returns:
        Dict mapping prompt_key → content
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()

        # Get production prompts first
        cur.execute(
            """SELECT DISTINCT ON (prompt_key) prompt_key, content, status
               FROM prompt_versions
               WHERE status IN ('production', 'staging')
               ORDER BY prompt_key,
                        CASE WHEN status = 'staging' THEN 0 ELSE 1 END""",
        )
        results = {}
        for row in cur.fetchall():
            prompt_key = row[0]
            if prompt_key not in results:
                results[prompt_key] = row[2]  # store status temporarily

        # Re-query to get content for the right status per environment
        prompts = {}
        cur.execute(
            "SELECT prompt_key, content, status FROM prompt_versions WHERE status IN ('production', 'staging')",
        )
        rows = cur.fetchall()

        # Group by key
        by_key: dict[str, list[tuple]] = {}
        for row in rows:
            key = row[0]
            if key not in by_key:
                by_key[key] = []
            by_key[key].append(row)

        for key, entries in by_key.items():
            if environment == "staging":
                # Prefer staging
                for entry in entries:
                    if entry[2] == "staging":
                        prompts[key] = entry[1]
                        break
                else:
                    for entry in entries:
                        if entry[2] == "production":
                            prompts[key] = entry[1]
                            break
            else:
                for entry in entries:
                    if entry[2] == "production":
                        prompts[key] = entry[1]
                        break

        return prompts
    except Exception:
        return {}
    finally:
        if conn:
            conn.close()


def delete_version(prompt_key: str, version: int) -> bool:
    """Delete a prompt version. Only draft versions can be deleted.

    Returns:
        True if deleted, False if not found or not a draft.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """DELETE FROM prompt_versions
               WHERE prompt_key = %s AND version = %s AND status = 'draft'""",
            (prompt_key, version),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def get_version(prompt_key: str, version: int) -> dict | None:
    """Get a specific version by key and version number.

    Returns:
        Dict with version data, or None if not found.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, prompt_key, version, content, status, change_summary,
                      created_by, created_at, promoted_at
               FROM prompt_versions
               WHERE prompt_key = %s AND version = %s""",
            (prompt_key, version),
        )
        row = cur.fetchone()
        if not row:
            return None
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    except Exception:
        return None
    finally:
        if conn:
            conn.close()
