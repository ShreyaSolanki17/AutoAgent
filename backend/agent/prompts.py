"""System prompts for each node in the agent graph.

Hardcoded defaults are used as fallback when the DB is unavailable or no
versioned prompt exists. Use get_prompt() to load from DB with auto-fallback.
"""

import os
import time

PLANNER_SYSTEM = """You are a planning module. Analyze the user's goal and decide the NEXT single action.

Tools: calculator, wikipedia, search, file_read, file_write, memory_save, memory_search, finish.

Rules:
- Greetings/casual chat → finish immediately
- Resolve pronouns (he/his/their/it) from conversation history. Include the resolved subject in the tool input.
- Never call the same tool twice. If a tool fails, switch or finish.
- When in doubt, finish with the best answer you have.

Format:
TOOL: <tool_name>
INPUT: <input>
REASON: <one sentence>
"""

OBSERVER_SYSTEM = """You are an observation module. Evaluate the tool result and decide:

- Goal answered? → DONE: <final answer written in a natural, conversational way>
- Need more? → CONTINUE: <what's still needed>

Rules:
- Be very willing to say DONE. Write the final answer as if talking to a friend — natural, concise, no bullet points, no raw tool output, no metadata.
- NEVER include tool names, memory IDs, JSON, or internal system details in your DONE response.
- If the same tool was called more than once, ALWAYS say DONE.
- If iterations >= 3 without a clear answer, say DONE with the best available response.
- For greetings, reply DONE immediately with just a friendly greeting.
"""

# ── Prompt cache (key → {content, ts}) ────────────────────
_prompt_cache: dict[str, dict] = {}
_CACHE_TTL = 60  # seconds

# Hardcoded defaults for fallback
_DEFAULTS = {
    "PLANNER_SYSTEM": PLANNER_SYSTEM,
    "OBSERVER_SYSTEM": OBSERVER_SYSTEM,
}


def _get_environment() -> str:
    """Read the current environment from APP_ENV (default: 'production')."""
    return os.getenv("APP_ENV", "production")


def get_prompt(prompt_key: str) -> str:
    """Load a prompt by key, with DB → cache → hardcoded fallback.

    Resolution order:
    1. In-memory cache (if not expired)
    2. Database (active prompt for current environment)
    3. Hardcoded default from this module

    The cache TTL is 60 seconds, so prompt changes in the DB are picked up
    within a minute without requiring a server restart.

    Args:
        prompt_key: e.g. 'PLANNER_SYSTEM', 'OBSERVER_SYSTEM'

    Returns:
        The prompt string, or empty string if key is unknown.
    """
    # 1. Check cache
    cached = _prompt_cache.get(prompt_key)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        return cached["content"]

    # 2. Try DB
    try:
        from backend.prompts.store import get_active_prompt
        content = get_active_prompt(prompt_key, environment=_get_environment())
        if content:
            _prompt_cache[prompt_key] = {"content": content, "ts": time.time()}
            return content
    except Exception:
        pass

    # 3. Fallback to hardcoded default
    return _DEFAULTS.get(prompt_key, "")


def refresh_cache():
    """Clear the prompt cache to force reload from DB on next call."""
    _prompt_cache.clear()
