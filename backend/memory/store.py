"""Memory store — save and retrieve memories using pgvector similarity search.

Includes:
- Similarity-based dedup: skips saving near-duplicate memories, updates
  the existing one's timestamp instead.
- TTL/expiry: memories auto-expire after 90 days by default.
- Importance scoring: personal-identity facts score higher than casual ones.
- Pruning: cap memories per user, evict low-importance + old entries.
- Cleanup helpers: delete expired memories, merge similar ones.
- Auto-cleanup scheduler: background job that periodically purges expired/trimmed memories.
"""

import os
import re
import threading
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

from backend.memory.embeddings import embed_text

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Tuning constants ──────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.95   # Tier 1: near-identical — refresh existing row
MERGE_SIMILARITY_THRESHOLD = 0.85   # Tier 2: semantic duplicate — merge (keep longer + bump importance)
DEFAULT_TTL_DAYS = 90               # memories expire after this many days
MAX_MEMORIES_PER_USER = 200         # cap: when exceeded, prune lowest-importance entries
IMPORTANCE_BUMP = 0.05              # how much to increase importance on each re-statement
IMPORTANCE_MAX = 1.0                # hard cap for importance_score

# ── Importance scoring patterns ───────────────────────────
# Higher = more important (identity facts > preferences > casual statements)
_HIGH_IMPORTANCE_PATTERNS = [
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bmy birthday\b",
    r"\bmy email\b",
]
_MEDIUM_IMPORTANCE_PATTERNS = [
    r"\bi like\b",
    r"\bi love\b",
    r"\bi prefer\b",
    r"\bi hate\b",
    r"\bi enjoy\b",
    r"\bmy favorite\b",
    r"\bremember (that|to|this)\b",
    r"\bi work\b",
    r"\bi live\b",
    r"\bi have a\b",
    r"\bi need\b",
    r"\bi always\b",
    r"\bi never\b",
]
_LOW_IMPORTANCE_PATTERNS = [
    r"\bi want\b",
    r"\bi think\b",
    r"\bi feel\b",
    r"\bi might\b",
    r"\bcan you\b",
    r"\bplease\b",
]

_HIGH_IMPORTANCE = 0.9
_MEDIUM_IMPORTANCE = 0.7
_LOW_IMPORTANCE = 0.5
_DEFAULT_IMPORTANCE = 0.3   # no pattern match → likely a one-off query


def compute_importance(content: str) -> float:
    """Compute an importance score (0.0–1.0) for a memory based on content heuristics.

    Scoring tiers:
    - 0.9 : Identity facts ("My name is…", "I am…")
    - 0.7 : Preferences / lasting traits ("I like…", "I work at…")
    - 0.5 : Casual statements ("I want…", "I think…")
    - 0.3 : No pattern match (likely ephemeral)

    Args:
        content: The memory text to score.

    Returns:
        Importance score between 0.0 and 1.0.
    """
    text = content.lower().strip()

    for pattern in _HIGH_IMPORTANCE_PATTERNS:
        if re.search(pattern, text):
            return _HIGH_IMPORTANCE

    for pattern in _MEDIUM_IMPORTANCE_PATTERNS:
        if re.search(pattern, text):
            return _MEDIUM_IMPORTANCE

    for pattern in _LOW_IMPORTANCE_PATTERNS:
        if re.search(pattern, text):
            return _LOW_IMPORTANCE

    return _DEFAULT_IMPORTANCE


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


def _bump_importance(current: float) -> float:
    """Bump importance by IMPORTANCE_BUMP, capped at IMPORTANCE_MAX."""
    return round(min(current + IMPORTANCE_BUMP, IMPORTANCE_MAX), 2)


def save_memory(
    content: str,
    user_id: str = "default",
    ttl_days: int = DEFAULT_TTL_DAYS,
    importance_override: float | None = None,
) -> int:
    """Save a memory with its embedding vector, with two-tier dedup and importance scoring.

    Two-tier dedup strategy:
    - Tier 1 (similarity >= 0.95): Near-identical — refresh existing row in place.
    - Tier 2 (0.85 <= similarity < 0.95): Semantic duplicate — merge by keeping the
      longer (more informative) text and bumping importance.
    - Below 0.85: No match — insert as a fresh memory.

    Importance is auto-computed via content heuristics unless overridden.

    Args:
        content: Text content to store.
        user_id: User identifier for filtering.
        ttl_days: Number of days until this memory expires.
        importance_override: If provided, use this score instead of auto-computing.

    Returns:
        The ID of the saved (or updated) memory row.
    """
    importance = importance_override if importance_override is not None else compute_importance(content)

    embedding = embed_text(content)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # ── Find the most similar existing memory ──
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity,
                       importance_score
                FROM memories
                WHERE user_id = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (vec_str, user_id, vec_str),
            )
            row = cur.fetchone()

            if row:
                similarity = row[2]

                # ── Tier 1: Near-identical (≥ 0.95) — refresh in place ──
                if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                    existing_id = row[0]
                    cur.execute(
                        """
                        UPDATE memories
                        SET content = %s,
                            embedding = %s::vector,
                            updated_at = NOW(),
                            expires_at = NOW() + INTERVAL '%s days',
                            importance_score = GREATEST(importance_score, %s)
                        WHERE id = %s
                        RETURNING id
                        """,
                        (content, vec_str, ttl_days, importance, existing_id),
                    )
                    memory_id = cur.fetchone()[0]
                    conn.commit()
                    return memory_id

                # ── Tier 2: Semantic duplicate (0.85–0.94) ──
                # Merge: keep the longer (more informative) text, bump importance
                if similarity >= MERGE_SIMILARITY_THRESHOLD:
                    existing_id = row[0]
                    existing_content = row[1]
                    existing_importance = row[3] if row[3] is not None else 0.5

                    # Keep the longer text — more informative
                    merged_content = content if len(content) > len(existing_content) else existing_content
                    merged_importance = _bump_importance(max(importance, existing_importance))

                    merged_embedding = embed_text(merged_content)
                    merged_vec_str = "[" + ",".join(str(v) for v in merged_embedding) + "]"

                    cur.execute(
                        """
                        UPDATE memories
                        SET content = %s,
                            embedding = %s::vector,
                            updated_at = NOW(),
                            expires_at = NOW() + INTERVAL '%s days',
                            importance_score = %s
                        WHERE id = %s
                        RETURNING id
                        """,
                        (merged_content, merged_vec_str, ttl_days, merged_importance, existing_id),
                    )
                    memory_id = cur.fetchone()[0]
                    conn.commit()
                    return memory_id

            # ── No duplicate — insert fresh ──
            cur.execute(
                """
                INSERT INTO memories (user_id, content, embedding, expires_at, importance_score)
                VALUES (%s, %s, %s::vector, NOW() + INTERVAL '%s days', %s)
                RETURNING id
                """,
                (user_id, content, vec_str, ttl_days, importance),
            )
            memory_id = cur.fetchone()[0]
        conn.commit()

        # ── Prune if over cap ──
        _prune_if_needed(user_id)

        return memory_id
    finally:
        conn.close()


def search_memories(query: str, user_id: str = "default", top_k: int = 5) -> list[dict]:
    """Search memories by semantic similarity (excludes expired memories).

    Results are ranked by a combined score: 70% similarity + 30% importance.
    Updates last_accessed timestamp on returned memories.

    Args:
        query: Search query text.
        user_id: Filter memories for this user.
        top_k: Number of top results to return.

    Returns:
        List of dicts with 'id', 'content', 'similarity', 'importance' keys.
    """
    query_embedding = embed_text(query)
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            # Combined ranking: similarity-weighted + importance-boosted
            cur.execute(
                """
                SELECT id, content,
                       1 - (embedding <=> %s::vector) AS similarity,
                       importance_score
                FROM memories
                WHERE user_id = %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY (0.7 * (1 - (embedding <=> %s::vector))
                          + 0.3 * importance_score) DESC
                LIMIT %s
                """,
                (vec_str, user_id, vec_str, top_k),
            )
            rows = cur.fetchall()

            # Update last_accessed for returned memories
            ids = [row[0] for row in rows]
            if ids:
                cur.execute(
                    "UPDATE memories SET last_accessed = NOW() WHERE id = ANY(%s)",
                    (ids,),
                )
                conn.commit()

        return [
            {
                "id": row[0],
                "content": row[1],
                "similarity": round(row[2], 4),
                "importance": round(row[3], 2) if row[3] is not None else 0.5,
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_all_memories(user_id: str = "default", include_expired: bool = False) -> list[dict]:
    """Get all memories for a user, ranked by importance then recency.

    Args:
        user_id: User identifier.
        include_expired: If True, also return expired memories.

    Returns:
        List of dicts with 'id', 'content', 'created_at', 'updated_at',
        'expires_at', 'importance', 'last_accessed' keys.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if include_expired:
                cur.execute(
                    """SELECT id, content, created_at, updated_at, expires_at,
                              importance_score, last_accessed
                    FROM memories WHERE user_id = %s
                    ORDER BY importance_score DESC, created_at DESC""",
                    (user_id,),
                )
            else:
                cur.execute(
                    """SELECT id, content, created_at, updated_at, expires_at,
                              importance_score, last_accessed
                    FROM memories
                    WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY importance_score DESC, created_at DESC""",
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
                "importance": round(row[5], 2) if row[5] is not None else 0.5,
                "last_accessed": str(row[6]) if row[6] else None,
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
        Dict with 'total', 'expired', 'active', 'avg_importance' counts.
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

            cur.execute(
                "SELECT COALESCE(AVG(importance_score), 0) FROM memories WHERE user_id = %s",
                (user_id,),
            )
            avg_importance = cur.fetchone()[0]

        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
            "avg_importance": round(float(avg_importance), 3),
        }
    finally:
        conn.close()


# ── Pruning ───────────────────────────────────────────────

def _prune_if_needed(user_id: str, max_memories: int = MAX_MEMORIES_PER_USER) -> int:
    """Evict low-importance memories if the user exceeds the cap.

    Strategy: delete the lowest-importance, oldest memories first
    until count is back under the cap. Never prunes memories with
    importance >= 0.8 (identity facts are protected).

    Args:
        user_id: User whose memories to prune.
        max_memories: Maximum allowed active memories.

    Returns:
        Number of memories pruned.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # Count active (non-expired) memories
            cur.execute(
                """SELECT COUNT(*) FROM memories
                WHERE user_id = %s AND (expires_at IS NULL OR expires_at > NOW())""",
                (user_id,),
            )
            active_count = cur.fetchone()[0]

            if active_count <= max_memories:
                return 0

            to_delete = active_count - max_memories

            # Delete lowest-importance, oldest first — protect high-importance (>= 0.8)
            cur.execute(
                """DELETE FROM memories
                WHERE id IN (
                    SELECT id FROM memories
                    WHERE user_id = %s
                      AND (expires_at IS NULL OR expires_at > NOW())
                      AND importance_score < 0.8
                    ORDER BY importance_score ASC, last_accessed ASC NULLS FIRST
                    LIMIT %s
                )
                RETURNING id""",
                (user_id, to_delete),
            )
            pruned = cur.rowcount
        conn.commit()
        return pruned
    finally:
        conn.close()


def prune_user_memories(user_id: str, max_memories: int = MAX_MEMORIES_PER_USER) -> dict:
    """Public API for manual pruning. See _prune_if_needed.

    Returns:
        Dict with 'pruned' count and 'remaining' count.
    """
    pruned = _prune_if_needed(user_id, max_memories)
    stats = get_memory_stats(user_id)
    return {"pruned": pruned, "remaining": stats["active"]}


# ── Auto-cleanup scheduler ────────────────────────────────

_cleanup_timer: threading.Timer | None = None
_CLEANUP_INTERVAL_SECONDS = 3600  # run every hour
_cleanup_started = False


def _cleanup_job():
    """Background job: delete expired memories globally, then start next timer."""
    global _cleanup_timer
    try:
        deleted = delete_expired_memories()
        if deleted > 0:
            from backend.logging_config import get_logger
            logger = get_logger(__name__)
            logger.info("memory.auto_cleanup", deleted=deleted)
    except Exception:
        pass  # Never crash the scheduler — just skip this run
    finally:
        # Schedule next run
        _cleanup_timer = threading.Timer(_CLEANUP_INTERVAL_SECONDS, _cleanup_job)
        _cleanup_timer.daemon = True
        _cleanup_timer.start()


def start_auto_cleanup():
    """Start the background cleanup scheduler. Idempotent — safe to call multiple times.

    Spawns a daemon timer that runs every hour to purge expired memories.
    The thread is marked daemon so it won't prevent the app from shutting down.
    """
    global _cleanup_started, _cleanup_timer
    if _cleanup_started:
        return
    _cleanup_started = True
    _cleanup_timer = threading.Timer(_CLEANUP_INTERVAL_SECONDS, _cleanup_job)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()


def stop_auto_cleanup():
    """Stop the background cleanup scheduler."""
    global _cleanup_started, _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
        _cleanup_timer = None
    _cleanup_started = False


def get_cleanup_status() -> dict:
    """Check the scheduler status.

    Returns:
        Dict with 'running' bool and 'interval_seconds'.
    """
    return {
        "running": _cleanup_started,
        "interval_seconds": _CLEANUP_INTERVAL_SECONDS,
    }
