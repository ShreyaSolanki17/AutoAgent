"""A/B testing for prompt variants.

Resolves which prompt variant a user gets based on deterministic hashing,
and provides helpers for managing A/B test lifecycle in the database.

Usage:
    from backend.prompts.ab_testing import get_prompt_for_ab_test, create_ab_test
    content, variant = get_prompt_for_ab_test("PLANNER_SYSTEM", user_id)
"""

import hashlib

from backend.prompts.store import _get_connection, get_active_prompt, get_version
from backend.agent.prompts import _get_environment

# ── Internal helpers ───────────────────────────────────────


def _assign_variant(user_id: str, test_name: str) -> str:
    """Deterministically assign user to variant A or B using SHA-256 hash.

    Uses the first hex character of SHA256(user_id + ":" + test_name):
      - Values 0-7  → variant A (50% of users)
      - Values 8-f  → variant B (50% of users)

    This is a pure function — no DB lookup needed — so the same user always
    gets the same variant for a given test_name, across restarts and servers.
    Different test_names produce independent assignments for the same user.

    Args:
        user_id: Unique user identifier.
        test_name: Unique A/B test identifier (used as hash salt).

    Returns:
        'A' or 'B'.
    """
    h = hashlib.sha256(f"{user_id}:{test_name}".encode()).hexdigest()
    return "A" if int(h[0], 16) < 8 else "B"


# ── DB operations ──────────────────────────────────────────


def get_ab_test(prompt_key: str) -> dict | None:
    """Look up the active A/B test for a given prompt key.

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM'.

    Returns:
        Dict with test metadata, or None if no active test exists.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, test_name, prompt_key, variant_a_version,
                      variant_b_version, status, description, created_by, created_at
               FROM ab_tests
               WHERE prompt_key = %s AND status = 'active'
               LIMIT 1""",
            (prompt_key,),
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


def create_ab_test(
    test_name: str,
    prompt_key: str,
    variant_a_version: int,
    variant_b_version: int,
    description: str = "",
    created_by: str = "admin",
) -> bool:
    """Create an A/B test record in the database.

    If a test with the same name already exists and is not 'completed',
    this will fail (unique constraint).

    Args:
        test_name: Unique identifier for the test.
        prompt_key: The prompt key being tested.
        variant_a_version: Version number for variant A.
        variant_b_version: Version number for variant B.
        description: Human-readable description.
        created_by: Who created the test.

    Returns:
        True if created successfully, False otherwise.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ab_tests
                   (test_name, prompt_key, variant_a_version, variant_b_version, description, created_by)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (test_name, prompt_key, variant_a_version, variant_b_version, description, created_by),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        if conn:
            conn.close()


def stop_ab_test(test_name: str) -> bool:
    """Stop an A/B test (status → 'completed', set ended_at).

    Args:
        test_name: The test to stop.

    Returns:
        True if a test was found and stopped.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """UPDATE ab_tests
               SET status = 'completed', ended_at = NOW()
               WHERE test_name = %s AND status = 'active'""",
            (test_name,),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        return False
    finally:
        if conn:
            conn.close()


def list_ab_tests(prompt_key: str | None = None) -> list[dict]:
    """List A/B tests, optionally filtered by prompt_key.

    Returns newest-first.
    """
    conn = None
    try:
        conn = _get_connection()
        cur = conn.cursor()
        if prompt_key:
            cur.execute(
                """SELECT id, test_name, prompt_key, variant_a_version,
                          variant_b_version, status, description, created_by,
                          created_at, ended_at
                   FROM ab_tests
                   WHERE prompt_key = %s
                   ORDER BY created_at DESC""",
                (prompt_key,),
            )
        else:
            cur.execute(
                """SELECT id, test_name, prompt_key, variant_a_version,
                          variant_b_version, status, description, created_by,
                          created_at, ended_at
                   FROM ab_tests
                   ORDER BY created_at DESC""",
            )
        rows = cur.fetchall()
        if not rows:
            return []
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


# ── Prompt resolution with A/B awareness ───────────────────


def get_prompt_for_ab_test(
    prompt_key: str,
    user_id: str,
) -> tuple[str | None, str | None]:
    """Resolve the prompt content for a user, considering active A/B tests.

    If an active A/B test exists for the prompt_key, the user is deterministically
    assigned to variant A or B, and that specific version's content is returned.
    Otherwise, standard single-prompt resolution is used.

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM'.
        user_id: Unique user identifier for deterministic variant assignment.

    Returns:
        Tuple of (prompt_content, ab_variant).
        ab_variant is 'A', 'B', or None (when no active A/B test).
        prompt_content is None if nothing can be resolved.
    """
    ab_test = get_ab_test(prompt_key)
    if ab_test is None:
        # No active A/B test — fall back to standard resolution
        content = get_active_prompt(prompt_key, environment=_get_environment())
        return content, None

    # Active A/B test — deterministically assign user
    variant = _assign_variant(user_id, ab_test["test_name"])
    version = ab_test["variant_a_version"] if variant == "A" else ab_test["variant_b_version"]

    # Fetch the specific version content
    ver = get_version(prompt_key, version)
    if ver and ver.get("content"):
        return ver["content"], variant

    # Specified version doesn't exist — fall back to standard resolution
    content = get_active_prompt(prompt_key, environment=_get_environment())
    return content, None
