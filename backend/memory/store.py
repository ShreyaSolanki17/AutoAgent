"""Memory store — save and retrieve memories using pgvector similarity search.
"""

import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

from backend.memory.embeddings import embed_text

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


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


def save_memory(content: str, user_id: str = "default") -> int:
    """Save a memory with its embedding vector.

    Args:
        content: Text content to store.
        user_id: User identifier for filtering.

    Returns:
        The ID of the saved memory row.
    """
    embedding = embed_text(content)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (user_id, content, embedding) VALUES (%s, %s, %s::vector) RETURNING id",
                (user_id, content, vec_str),
            )
            memory_id = cur.fetchone()[0]
        conn.commit()
        return memory_id
    finally:
        conn.close()


def search_memories(query: str, user_id: str = "default", top_k: int = 5) -> list[dict]:
    """Search memories by semantic similarity.

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
            # Format vector as string '[v1,v2,...]' for pgvector
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE user_id = %s
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


def get_all_memories(user_id: str = "default") -> list[dict]:
    """Get all memories for a user, newest first.

    Args:
        user_id: User identifier.

    Returns:
        List of dicts with 'id', 'content', 'created_at' keys.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, created_at FROM memories WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()

        return [
            {"id": row[0], "content": row[1], "created_at": str(row[2])}
            for row in rows
        ]
    finally:
        conn.close()


def delete_memory(memory_id: int) -> bool:
    """Delete a memory by ID.

    Args:
        memory_id: The memory row ID.

    Returns:
        True if a row was deleted.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()
