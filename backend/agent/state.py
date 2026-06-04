"""AgentState — the shared state that flows through every node in the graph.
"""

from typing import TypedDict


class AgentState(TypedDict):
    user_goal: str              # The original task the user wants done
    user_id: str                # Authenticated user ID (for per-user memory)
    messages: list[dict]        # Full agent trace (plan/act/observe steps)
    chat_history: list[dict]    # Previous user/assistant messages from prior turns
    plan: str                   # Current plan: which tool to use and why
    tool_name: str              # Name of the tool selected for this step
    tool_input: str             # Input to pass to the tool
    observation: str            # Result returned by the tool
    final_answer: str           # Set when the agent decides it's done
    iterations: int             # How many act-observe cycles have run
    used_tools: list[dict]      # History of tools called: [{"tool": "wikipedia", "input": "..."}]
