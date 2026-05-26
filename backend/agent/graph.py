"""LangGraph agent — Plan → Act → Observe → (loop or finish).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.prompts import PLANNER_SYSTEM, OBSERVER_SYSTEM
from backend.tools.registry import execute_tool

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MAX_ITERATIONS = 5  # safety limit to prevent infinite loops


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=4096,
    )


# ── Node 1: Plan ──────────────────────────────────────────
def plan_node(state: AgentState) -> AgentState:
    """Decide which tool to use next based on the current state."""
    llm = _get_llm()

    user_msg = f"User goal: {state['user_goal']}\n"
    if state["observation"]:
        user_msg += f"Previous observation: {state['observation']}\n"

    response = llm.invoke([
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ])

    content = response.content.strip()

    # Parse the structured response
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
    """Evaluate the result. Decide if the agent is done or should loop."""
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
    return new_state


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

    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)

    graph.set_entry_point("plan")
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
