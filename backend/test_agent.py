"""Test: verify memory-augmented agent with auto-save and retrieval.
"""

from backend.agent.graph import run_agent
from backend.memory.store import get_all_memories

if __name__ == "__main__":
    print("=== Memory-Augmented Agent Test ===\n")

    # Session 1: Tell the agent some personal facts
    print("--- Session 1: Saving personal facts ---")
    facts = [
        "My name is Alex",
        "My favorite programming language is Python",
        "I have two cats named Luna and Max",
    ]

    for fact in facts:
        print(f"\nUser: {fact}")
        result = run_agent(fact)
        answer = result.get("final_answer", "(no answer)")
        print(f"Agent: {answer}")

    # Check what was saved
    print("\n--- Memories stored ---")
    all_mem = get_all_memories()
    for m in all_mem:
        print(f"  [{m['id']}] {m['content']}")

    # Session 2: Ask about remembered facts
    print("\n\n--- Session 2: Retrieving memories ---")
    questions = [
        "What is my name?",
        "What is my favorite programming language?",
        "Do I have any pets?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        result = run_agent(q)
        answer = result.get("final_answer", "(no answer)")
        print(f"Agent: {answer}")

    print(f"\n[OK] Total memories: {len(all_mem)}")
