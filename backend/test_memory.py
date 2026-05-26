"""Test: verify memory save + semantic search works.
"""

from backend.memory.store import save_memory, search_memories, get_all_memories

if __name__ == "__main__":
    print("=== Memory Store Test ===\n")

    # Save some test memories
    test_memories = [
        "The user's favorite programming language is Python.",
        "The user loves Italian food, especially pizza and pasta.",
        "The user is building an AI agent project called AutoAgent.",
        "The user owns two cats named Luna and Max.",
        "The user works as a software engineer at a tech startup.",
    ]

    print("Saving test memories...")
    for mem in test_memories:
        mid = save_memory(mem)
        print(f"  Saved [id={mid}]: {mem}")

    # Search by semantic similarity
    print("\n--- Search: 'What food does the user like?' ---")
    results = search_memories("What food does the user like?", top_k=3)
    for r in results:
        print(f"  [{r['similarity']:.2f}] {r['content']}")

    print("\n--- Search: 'Tell me about pets' ---")
    results = search_memories("Tell me about pets", top_k=3)
    for r in results:
        print(f"  [{r['similarity']:.2f}] {r['content']}")

    print("\n--- Search: 'What project is the user working on?' ---")
    results = search_memories("What project is the user working on?", top_k=3)
    for r in results:
        print(f"  [{r['similarity']:.2f}] {r['content']}")

    print("\n--- All memories ---")
    all_mem = get_all_memories()
    for m in all_mem:
        print(f"  [{m['id']}] {m['content']} ({m['created_at']})")

    print(f"\n[OK] Total memories stored: {len(all_mem)}")
