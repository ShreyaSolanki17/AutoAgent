"""LangGraph agent — Retrieve Memory → Plan → Act → Observe → (loop or finish).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.prompts import PLANNER_SYSTEM, OBSERVER_SYSTEM
from backend.tools.registry import execute_tool
from backend.memory.store import search_memories, save_memory

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MAX_ITERATIONS = 5


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=4096,
    )


# ── Node 0: Retrieve Memory ────────────────────────────────
def retrieve_memory_node(state: AgentState) -> AgentState:
    """Search long-term memory for context relevant to the user's goal.

    Runs once at the start (iteration 0) to inject past context
    into the planner's prompt.
    """
    # Only retrieve on first iteration to avoid redundant calls
    if state["iterations"] > 0:
        return state

    try:
        results = search_memories(state["user_goal"], top_k=3)
        if results:
            context_lines = [f"  - {r['content']}" for r in results]
            memory_context = "Relevant memories from past sessions:\n" + "\n".join(context_lines)
        else:
            memory_context = ""
    except Exception:
        memory_context = ""

    return {
        **state,
        "messages": state["messages"] + [
            {"role": "system", "content": memory_context}
        ] if memory_context else state["messages"],
    }


# ── Node 1: Plan ──────────────────────────────────────────
def plan_node(state: AgentState) -> AgentState:
    """Decide which tool to use next based on the current state."""
    llm = _get_llm()

    user_msg = f"User goal: {state['user_goal']}\n"

    # Include memory context if available
    for msg in state["messages"]:
        if msg.get("role") == "system" and msg.get("content", "").startswith("Relevant memories"):
            user_msg = msg["content"] + "\n\n" + user_msg
            break

    if state["observation"]:
        user_msg += f"Previous observation: {state['observation']}\n"

    response = llm.invoke([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ])

    content = response.content.strip()

    tool_name = "finish"
    tool_input = ""
    reason = content

    for line in content.splitlines():
        if line.startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
        elif line.startswith("INPUT:"):
            tool_input = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {
        **state,
        "plan": reason,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "messages": state["messages"] + [
            {"role": "assistant", "content": f"[Plan] {tool_name}: {reason}"}
        ],
    }


# ── Node 2: Act ───────────────────────────────────────────
def act_node(state: AgentState) -> AgentState:
    """Execute the chosen tool using the tool registry."""
    tool = state["tool_name"]
    tool_input = state["tool_input"]

    if tool == "finish":
        observation = f"Agent is ready to provide the final answer for: {state['user_goal']}"
    else:
        observation = execute_tool(tool, tool_input)

    return {
        **state,
        "observation": observation,
        "messages": state["messages"] + [
            {"role": "assistant", "content": f"[Act] {tool} -> {observation}"}
        ],
    }


# ── Node 3: Observe ───────────────────────────────────────
def observe_node(state: AgentState) -> AgentState:
    """Evaluate the result. Decide if the agent is done or should loop.
    Also auto-saves important user facts to long-term memory."""
    llm = _get_llm()

    user_msg = (
        f"User goal: {state['user_goal']}\n"
        f"Tool used: {state['tool_name']}\n"
        f"Tool result: {state['observation']}\n"
        f"Iterations so far: {state['iterations']}"
    )

    response = llm.invoke([
        {"role": "system", "content": OBSERVER_SYSTEM},
        {"role": "user", "content": user_msg},
    ])

    content = response.content.strip()

    new_state = {**state, "iterations": state["iterations"] + 1}

    if content.startswith("DONE:"):
        final_answer = content.split(":", 1)[1].strip()
        new_state["final_answer"] = final_answer
    else:
        new_state["final_answer"] = ""

    new_state["messages"] = state["messages"] + [
        {"role": "assistant", "content": f"[Observe] {content}"}
    ]

    # Auto-save: if the user goal contains personal facts, remember them
    _auto_save_memory(state["user_goal"], state["observation"])

    return new_state


def _auto_save_memory(user_goal: str, observation: str):
    """Heuristic: save to memory if the goal looks like a personal fact.

    Triggers on patterns like "I like...", "My name is...", "Remember that..."
    """
    goal_lower = user_goal.lower().strip()
    triggers = [
        "my name is", "i like", "i love", "i prefer",
        "i work", "i live", "i have", "remember that",
        "my favorite", "i am", "i'm",
    ]

    should_save = any(t in goal_lower for t in triggers)

    if should_save:
        try:
            # Clean up the fact (remove question marks, trailing punctuation)
            fact = user_goal.strip().rstrip("?!").strip()
            save_memory(fact)
        except Exception:
            pass  # Don't break the agent if memory save fails


# ── Routing logic ─────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Route to END or back to plan."""
    if state["final_answer"]:
        return "end"
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    return "continue"


# ── Build the graph ───────────────────────────────────────
def build_agent():
    """Create and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory", retrieve_memory_node)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges(
        "observe",
        should_continue,
        {
            "continue": "plan",
            "end": END,
        },
    )

    return graph.compile()


def run_agent(user_goal: str) -> AgentState:
    """Convenience function: build graph, run it, return final state."""
    agent = build_agent()
    initial_state: AgentState = {
        "user_goal": user_goal,
        "messages": [],
        "plan": "",
        "tool_name": "",
        "tool_input": "",
        "observation": "",
        "final_answer": "",
        "iterations": 0,
    }
    return agent.invoke(initial_state)
