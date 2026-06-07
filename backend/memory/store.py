"""Memory store — save and retrieve memories using pgvector similarity search.

Includes:
- Similarity-based dedup: skips saving near-duplicate memories, updates
  the existing one's timestamp instead.
- TTL/expiry: memories auto-expire after 90 days by default.
- Cleanup helpers: delete expired memories, merge similar ones.
"""

import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

from backend.memory.embeddings import embed_text

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Tuning constants ──────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.95   # cosine similarity above which two memories are "the same"
DEFAULT_TTL_DAYS = 90               # memories expire after this many days


def _get_connection():
    """Create a new PostgreSQL connection with pgvector registered."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "autoagent_db"),
        user=os.getenv("POSTGRES_USER", "autoagent"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )
    register_vector(conn)
    return conn


def save_memory(content: str, user_id: str = "default", ttl_days: int = DEFAULT_TTL_DAYS) -> int:
    """Save a memory with its embedding vector, with dedup.

    Before inserting, searches for an existing memory by the same user
    whose cosine similarity >= DEDUP_SIMILARITY_THRESHOLD.  If found,
    that existing row's content is updated (and its timestamps refreshed)
    instead of inserting a duplicate.

    Args:
        content: Text content to store.
        user_id: User identifier for filtering.
        ttl_days: Number of days until this memory expires.

    Returns:
        The ID of the saved (or updated) memory row.
    """
    embedding = embed_text(content)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # ── Dedup: look for a highly similar existing memory ──
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE user_id = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (vec_str, user_id, vec_str),
            )
            row = cur.fetchone()

            if row and row[2] >= DEDUP_SIMILARITY_THRESHOLD:
                # Near-duplicate found — refresh the existing row
                existing_id = row[0]
                cur.execute(
                    """
                    UPDATE memories
                    SET content = %s,
                        embedding = %s::vector,
                        updated_at = NOW(),
                        expires_at = NOW() + INTERVAL '%s days'
                    WHERE id = %s
                    RETURNING id
                    """,
                    (content, vec_str, ttl_days, existing_id),
                )
                memory_id = cur.fetchone()[0]
                conn.commit()
                return memory_id

            # ── No duplicate — insert fresh ──
            cur.execute(
                """
                INSERT INTO memories (user_id, content, embedding, expires_at)
                VALUES (%s, %s, %s::vector, NOW() + INTERVAL '%s days')
                RETURNING id
                """,
                (user_id, content, vec_str, ttl_days),
            )
            memory_id = cur.fetchone()[0]
        conn.commit()
        return memory_id
    finally:
        conn.close()


def search_memories(query: str, user_id: str = "default", top_k: int = 5) -> list[dict]:
    """Search memories by semantic similarity (excludes expired memories).

    Args:
        query: Search query text.
        user_id: Filter memories for this user.
        top_k: Number of top results to return.

    Returns:
        List of dicts with 'id', 'content', 'similarity' keys.
    """
    query_embedding = embed_text(query)
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE user_id = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_str, user_id, vec_str, top_k),
            )
            rows = cur.fetchall()

        return [
            {"id": row[0], "content": row[1], "similarity": round(row[2], 4)}
            for row in rows
        ]
    finally:
        conn.close()


def get_all_memories(user_id: str = "default", include_expired: bool = False) -> list[dict]:
    """Get all memories for a user, newest first.

    Args:
        user_id: User identifier.
        include_expired: If True, also return expired memories.

    Returns:
        List of dicts with 'id', 'content', 'created_at', 'updated_at', 'expires_at' keys.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if include_expired:
                cur.execute(
                    """SELECT id, content, created_at, updated_at, expires_at
                    FROM memories WHERE user_id = %s ORDER BY created_at DESC""",
                    (user_id,),
                )
            else:
                cur.execute(
                    """SELECT id, content, created_at, updated_at, expires_at
                    FROM memories
                    WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY created_at DESC""",
                    (user_id,),
                )
            rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "created_at": str(row[2]),
                "updated_at": str(row[3]) if row[3] else None,
                "expires_at": str(row[4]) if row[4] else None,
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_memory(memory_id: int, user_id: str | None = None) -> bool:
    """Delete a memory by ID.

    Args:
        memory_id: The memory row ID.
        user_id: If provided, only delete the memory if it belongs to this user.

    Returns:
        True if a row was deleted.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND user_id = %s",
                    (memory_id, user_id),
                )
            else:
                cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


# ── Cleanup helpers ──────────────────────────────────────

def delete_expired_memories(user_id: str | None = None) -> int:
    """Delete all expired memories.

    Args:
        user_id: If provided, only delete expired memories for this user.

    Returns:
        Number of rows deleted.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    "DELETE FROM memories WHERE user_id = %s AND expires_at <= NOW()",
                    (user_id,),
                )
            else:
                cur.execute("DELETE FROM memories WHERE expires_at <= NOW()")
            count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def find_duplicate_groups(user_id: str, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list[list[dict]]:
    """Find groups of near-duplicate memories for a user.

    Uses a simple O(n²) comparison within each user's memories.
    Suitable for moderate memory counts (< 1000 per user).

    Args:
        user_id: User to check.
        threshold: Similarity above which two memories are considered duplicates.

    Returns:
        List of groups, where each group is a list of memory dicts
        that are all similar to each other.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, embedding, created_at, updated_at, expires_at
                FROM memories
                WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at ASC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        memories = [
            {
                "id": r[0],
                "content": r[1],
                "embedding": r[2],
                "created_at": str(r[3]),
                "updated_at": str(r[4]) if r[4] else None,
                "expires_at": str(r[5]) if r[5] else None,
            }
            for r in rows
        ]

        # Build duplicate groups via Union-Find
        parent = list(range(len(memories)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Compare all pairs using pgvector cosine similarity
        if len(memories) > 1:
            vec_strs = [
                "[" + ",".join(str(v) for v in m["embedding"]) + "]"
                for m in memories
            ]
            with conn.cursor() as cur:
                for i in range(len(memories)):
                    for j in range(i + 1, len(memories)):
                        cur.execute(
                            "SELECT 1 - (%s::vector <=> %s::vector)",
                            (vec_strs[i], vec_strs[j]),
                        )
                        sim = cur.fetchone()[0]
                        if sim >= threshold:
                            union(i, j)

        # Collect groups
        groups: dict[int, list[dict]] = {}
        for idx, mem in enumerate(memories):
            root = find(idx)
            groups.setdefault(root, []).append(mem)

        # Only return groups that actually have duplicates (size > 1)
        return [members for members in groups.values() if len(members) > 1]
    finally:
        conn.close()


def merge_duplicate_group(keep_id: int, remove_ids: list[int], user_id: str) -> bool:
    """Merge a group of duplicates: keep one, delete the rest.

    Args:
        keep_id: ID of the memory to keep.
        remove_ids: IDs of memories to delete.
        user_id: User ID for safety check.

    Returns:
        True if the merge succeeded.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # Verify the kept memory belongs to this user
            cur.execute("SELECT id FROM memories WHERE id = %s AND user_id = %s", (keep_id, user_id))
            if not cur.fetchone():
                return False

            # Refresh the kept memory's timestamps
            cur.execute(
                "UPDATE memories SET updated_at = NOW(), expires_at = NOW() + INTERVAL '%s days' WHERE id = %s",
                (DEFAULT_TTL_DAYS, keep_id),
            )

            # Delete the others (only if they belong to this user)
            if remove_ids:
                cur.execute(
                    "DELETE FROM memories WHERE id = ANY(%s) AND user_id = %s",
                    (remove_ids, user_id),
                )
        conn.commit()
        return True
    finally:
        conn.close()


def get_memory_stats(user_id: str) -> dict:
    """Get memory statistics for a user.

    Returns:
        Dict with 'total', 'expired', 'active' counts.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = %s",
                (user_id,),
            )
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = %s AND expires_at <= NOW()",
                (user_id,),
            )
            expired = cur.fetchone()[0]

        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
        }
    finally:
        conn.close()
