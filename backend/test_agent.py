"""Test: verify the LangGraph agent with real tools.
"""

from backend.agent.graph import run_agent

if __name__ == "__main__":
    print("=== AutoAgent + Real Tools Test ===\n")

    goal = "What is the capital of France? Answer in one sentence."
    print(f"User goal: {goal}\n")

    try:
        result = run_agent(goal)

        print("--- Agent Trace ---")
        for msg in result["messages"]:
            print(f"  {msg['content']}")

        print(f"\n[OK] Final answer: {result.get('final_answer', '(none)')}")
        print(f"     Iterations: {result['iterations']}")

    except RuntimeError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")
