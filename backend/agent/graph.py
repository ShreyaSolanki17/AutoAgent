"""LangGraph agent — Retrieve Memory → Plan → Act → Observe → (loop or finish).

All memory operations are scoped to a user_id so each user has their own
private long-term memory space in the shared pgvector table.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState
from backend.agent.prompts import get_prompt
from backend.tools.registry import execute_tool
from backend.memory.store import search_memories, save_memory
from backend.logging_config import get_logger
from backend.config import (
    PROVIDERS,
    MAX_ITERATIONS,
    MAX_OBSERVATION_CHARS,
    AUTO_SAVE_TRIGGERS,
)

logger = get_logger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _get_llm() -> ChatGroq | ChatOpenAI:
    """Return the first available LLM client (tries all providers).

    Priority: Groq → Google AI Studio → OpenRouter
    All are free-tier and use OpenAI-compatible APIs.

    Reloads .env on every call so key changes are picked up
    without restarting the server.

    Raises:
        RuntimeError: If no provider API key is configured.
    """
    # Reload .env so key changes are picked up without server restart
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

    from backend.config import TEMPERATURE, MAX_TOKENS

    for provider in PROVIDERS:
        api_key = os.getenv(provider["api_key_env"])
        if not api_key:
            continue

        if provider["base_url"] is None:
            # Groq — native langchain-groq
            return ChatGroq(
                model=provider["model"],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        else:
            # Google / OpenRouter — OpenAI-compatible via langchain-openai
            return ChatOpenAI(
                model=provider["model"],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_key=api_key,
                base_url=provider["base_url"],
            )

    raise RuntimeError(
        "All LLM providers failed. No API key found.\n"
        "Set at least one of these in .env: "
        + ", ".join(p["api_key_env"] for p in PROVIDERS)
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
        results = search_memories(state["user_goal"], user_id=state["user_id"], top_k=3)
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

    # Include conversation history (previous Q&A) so the agent can
    # resolve follow-up questions that reference earlier topics.
    chat_history = state.get("chat_history", [])
    if chat_history:
        user_msg = (
            "Conversation so far:\n"
            + "\n".join(f"  {m['role'].capitalize()}: {m['content']}" for m in chat_history)
            + "\n\n"
            + user_msg
        )

    # Include memory context if available (as background info, not to quote back)
    for msg in state["messages"]:
        if msg.get("role") == "system" and msg.get("content", "").startswith("Relevant memories"):
            user_msg = "Background info about the user (use naturally in conversation, do NOT quote or mention explicitly):\n" + msg["content"] + "\n\n" + user_msg
            break

    if state["observation"]:
        user_msg += f"Previous observation: {state['observation']}\n"

    # Tell the planner which tools have already been tried (truncated summaries)
    used_tools = state.get("used_tools", [])
    if used_tools:
        user_msg += "\nTools already tried:\n"
        for entry in used_tools:
            tool = entry["tool"]
            snippet = entry["result"][:80].replace("\n", " ")
            user_msg += f"  - {tool}: {snippet}…\n"
        user_msg += "Pick a DIFFERENT tool or use 'finish'.\n"

    response = llm.invoke([
        {"role": "system", "content": get_prompt("PLANNER_SYSTEM")},
        {"role": "user", "content": user_msg},
    ])

    content = response.content.strip()

    tool_name = "finish"
    tool_input = ""
    reason = content

    for line in content.splitlines():
        # Case-insensitive prefix matching for robustness
        upper_line = line.strip().upper()
        if upper_line.startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
        elif upper_line.startswith("INPUT:"):
            tool_input = line.split(":", 1)[1].strip()
        elif upper_line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    # If the LLM didn't use the structured format at all, try to detect
    # a bare tool name or treat the whole response as a finish answer.
    if tool_name == "finish" and reason == content:
        # Check if the entire response is just a known tool name
        known_tools = {"search", "wikipedia", "calculator", "file_read",
                       "file_write", "memory_save", "memory_search"}
        stripped = content.strip().lower()
        if stripped in known_tools:
            tool_name = stripped
            tool_input = state["user_goal"]
            reason = f"Using {tool_name} to find information"

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
        observation = f"Respond directly to the user. No tool needed."
    else:
        observation = execute_tool(tool, tool_input)

    # Track which tools have been used and their results
    used_tools = list(state.get("used_tools", []))
    if tool != "finish":
        used_tools.append({"tool": tool, "input": tool_input, "result": observation})

    # Truncate observation in trace to prevent token bloat across iterations
    obs_for_trace = observation[:MAX_OBSERVATION_CHARS] + "…" if len(observation) > MAX_OBSERVATION_CHARS else observation

    return {
        **state,
        "observation": observation,
        "used_tools": used_tools,
        "messages": state["messages"] + [
            {"role": "assistant", "content": f"[Act] {tool} -> {obs_for_trace}"}
        ],
    }


def _build_fallback_answer(state: AgentState) -> str:
    """Build a fallback answer when the agent exhausts its tools."""
    used = state.get("used_tools", [])
    tried = ", ".join(f'"{e["tool"]}"' for e in used)

    # Collect any non-error observations to include partial findings
    findings = []
    for entry in used:
        result = entry.get("result", "")
        if result and not result.startswith("Error") and not result.startswith("No ") and len(result) > 20:
            # Extract first meaningful sentence
            first_line = result.split("\n")[0].strip()
            if len(first_line) > 10:
                findings.append(first_line[:200])

    base = (
        f"I searched using {tried} but couldn't find a definitive answer "
        f"for your question."
    )
    if findings:
        base += " Here's what I found so far:\n" + "\n".join(f"• {f}" for f in findings)
    base += "\nCould you provide more context or rephrase your question?"
    return base


# ── Node 3: Observe ───────────────────────────────────────
def observe_node(state: AgentState) -> AgentState:
    """Evaluate the result. Decide if the agent is done or should loop.
    Also auto-saves important user facts to long-term memory."""
    llm = _get_llm()

    used_tools = state.get("used_tools", [])
    tool_failure_count = sum(1 for e in used_tools if e["tool"] == state["tool_name"])

    # When planner chose "finish", respond directly as a conversationalist
    if state["tool_name"] == "finish":
        user_msg = f"User: {state['user_goal']}\n"
        chat_history = state.get("chat_history", [])
        if chat_history:
            user_msg = (
                "Previous conversation:\n"
                + "\n".join(f"  {m['role'].capitalize()}: {m['content']}" for m in chat_history)
                + "\n\n"
                + user_msg
            )
        # Include memory context as background (don't quote it back)
        memory_context = ""
        for msg in state["messages"]:
            if msg.get("role") == "system" and msg.get("content", "").startswith("Relevant memories"):
                memory_context = msg["content"]
                break
        if memory_context:
            user_msg = f"Background about the user (use naturally, never quote or mention explicitly):\n{memory_context}\n\n" + user_msg
        user_msg += "Respond directly to the user in a friendly, natural way. Be concise (1-2 sentences). Never mention tool results, memory IDs, or internal details."

        response = llm.invoke([
            {"role": "system", "content": "You are a friendly, helpful assistant. Never mention tool names, memory IDs, or internal system details in your responses."},
            {"role": "user", "content": user_msg},
        ])
        content = f"DONE: {response.content.strip()}"
    else:
        # Build the user message with tool history for normal tool evaluation
        user_msg = f"User goal: {state['user_goal']}\n"

        chat_history = state.get("chat_history", [])
        if chat_history:
            user_msg = (
                "Previous conversation:\n"
                + "\n".join(f"  {m['role'].capitalize()}: {m['content']}" for m in chat_history)
                + "\n\n"
                + user_msg
            )

        # Truncate observation in prompt to save tokens
        obs_for_prompt = state['observation'][:MAX_OBSERVATION_CHARS] + "…" if len(state['observation']) > MAX_OBSERVATION_CHARS else state['observation']

        user_msg += (
            f"Tool used: {state['tool_name']}\n"
            f"Tool result: {obs_for_prompt}\n"
            f"Iterations so far: {state['iterations']}"
        )

        if tool_failure_count > 1:
            user_msg += (
                f"\n\n⚠️  WARNING: The tool '{state['tool_name']}' has already been "
                f"called {tool_failure_count} times.\n"
                f"You MUST respond with 'DONE: <answer>' — do NOT say CONTINUE."
            )

        response = llm.invoke([
            {"role": "system", "content": get_prompt("OBSERVER_SYSTEM")},
            {"role": "user", "content": user_msg},
        ])
        content = response.content.strip()

    new_state = {**state, "iterations": state["iterations"] + 1}

    if content.upper().startswith("DONE:"):
        final_answer = content.split(":", 1)[1].strip()
        new_state["final_answer"] = final_answer
    else:
        # SAFETY OVERRIDE: if this tool has already been tried 2+ times
        # and observer still says CONTINUE, force a DONE with fallback.
        # >= 2 (not >= 1) so the agent gets to try a different tool first.
        if tool_failure_count >= 2 and state["tool_name"] != "finish":
            new_state["final_answer"] = _build_fallback_answer(new_state)
        else:
            new_state["final_answer"] = ""

    new_state["messages"] = state["messages"] + [
        {"role": "assistant", "content": f"[Observe] {content}"}
    ]

    # Auto-save: if the user goal contains personal facts, remember them
    _auto_save_memory(state["user_goal"], state["observation"], user_id=state["user_id"])

    return new_state


def _auto_save_memory(user_goal: str, observation: str, user_id: str = "default"):
    """Heuristic: save to memory if the goal looks like a personal fact.

    Triggers on patterns like "I like...", "My name is...", "Remember that..."
    """
    goal_lower = user_goal.lower().strip()

    should_save = any(t in goal_lower for t in AUTO_SAVE_TRIGGERS)

    if should_save:
        try:
            # Clean up the fact (remove question marks, trailing punctuation)
            fact = user_goal.strip().rstrip("?!").strip()
            save_memory(fact, user_id=user_id)
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


def run_agent(user_goal: str, user_id: str = "default", history: list[dict] | None = None) -> AgentState:
    """Convenience function: build graph, run it, return final state with usage stats."""
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    provider_used = ""
    model_used = ""

    logger.info(
        "agent.run_started",
        user_id=user_id,
        goal=user_goal[:100],
        history_length=len(history) if history else 0,
    )

    agent = build_agent()
    initial_state: AgentState = {
        "user_goal": user_goal,
        "user_id": user_id,
        "messages": [],
        "chat_history": history or [],
        "plan": "",
        "tool_name": "",
        "tool_input": "",
        "observation": "",
        "final_answer": "",
        "iterations": 0,
        "used_tools": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "provider": "", "model": "", "cost_usd": 0.0},
    }
    result = agent.invoke(initial_state)

    # If the agent hit max iterations without producing an answer, synthesize one
    if not result.get("final_answer"):
        logger.warning(
            "agent.max_iterations_reached",
            user_id=user_id,
            iterations=result["iterations"],
            tools_used=[e["tool"] for e in result.get("used_tools", [])],
        )
        llm = _get_llm()
        trace_summary = "\n".join(
            msg["content"] for msg in result.get("messages", [])
            if msg.get("role") == "assistant"
        )
        try:
            synth = llm.invoke([
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. The agent loop did not produce "
                        "a direct answer. Based on the trace below, give a concise, "
                        "friendly response to the user. If it was a greeting, just greet "
                        "back warmly. If the tools returned no relevant results, honestly "
                        "say you couldn't find the information. Keep it under 3 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User said: {user_goal}\n\nTrace:\n{trace_summary}",
                },
            ])
            result["final_answer"] = synth.content.strip()
            # Count fallback synthesis tokens
            if hasattr(synth, 'usage_metadata') and synth.usage_metadata:
                total_input_tokens += synth.usage_metadata.get("input_tokens", 0)
                total_output_tokens += synth.usage_metadata.get("output_tokens", 0)
        except Exception:
            result["final_answer"] = f"I wasn't able to find an answer for: {user_goal}"

    elapsed_ms = round((time.time() - start_time) * 1000)

    # Detect which provider/model was used
    for provider_cfg in PROVIDERS:
        if os.getenv(provider_cfg["api_key_env"]):
            provider_used = provider_cfg["name"]
            model_used = provider_cfg["model"]
            break

    # Calculate cost
    from backend.middleware.usage import get_cost
    cost = get_cost(provider_used, model_used, total_input_tokens, total_output_tokens)

    result["usage"] = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "provider": provider_used,
        "model": model_used,
        "cost_usd": cost,
        "response_time_ms": elapsed_ms,
    }

    logger.info(
        "agent.run_completed",
        user_id=user_id,
        iterations=result["iterations"],
        tools_used=[e["tool"] for e in result.get("used_tools", [])],
        elapsed_ms=elapsed_ms,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=cost,
        answer_length=len(result.get("final_answer", "")),
    )

    return result
