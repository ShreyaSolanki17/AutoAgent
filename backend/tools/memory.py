"""Memory tool — lets the agent save and retrieve long-term memories.
"""

from backend.memory.store import save_memory, search_memories


def memory_save(content: str, user_id: str = "default") -> str:
    """Save a fact to long-term memory.

    Args:
        content: The fact to remember (e.g. "User's favorite language is Python").
        user_id: User identifier for filtering memories.

    Returns:
        Confirmation message.
    """
    try:
        memory_id = save_memory(content, user_id)
        return f"Memory saved [id={memory_id}]: {content}"
    except Exception as e:
        return f"Error saving memory: {e}"


def memory_search(query: str, user_id: str = "default", top_k: int = 5) -> str:
    """Search long-term memory by semantic similarity.

    Args:
        query: What to search for (e.g. "user's favorite food").
        user_id: User identifier for filtering.
        top_k: Number of results to return.

    Returns:
        Formatted string of matching memories, or "No memories found."
    """
    try:
        results = search_memories(query, user_id, top_k)

        if not results:
            return "No relevant memories found."

        formatted = []
        for r in results:
            formatted.append(f"  [{r['similarity']:.2f}] {r['content']}")

        return "Relevant memories:\n" + "\n".join(formatted)

    except Exception as e:
        return f"Error searching memory: {e}"
