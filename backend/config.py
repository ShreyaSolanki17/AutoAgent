"""Shared configuration constants — single source of truth.

All provider lists, model names, LLM generation params, loop limits,
and auto-save triggers are defined here.  Both ``backend.agent.graph``
and ``backend.llm`` import from this module so there is no drift.
"""

# ── LLM Provider Config (free-tier only) ──────────────────
# All providers are tried in order; first one with a valid key wins.
PROVIDERS: list[dict] = [
    {
        "name": "groq",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "base_url": None,  # native Groq SDK via langchain-groq
    },
    {
        "name": "google",
        "model": "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    {
        "name": "openrouter",
        "model": "openai/gpt-oss-20b:free",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
]

# ── LLM Generation Params ─────────────────────────────────
TEMPERATURE: float = 0.1
MAX_TOKENS: int = 4096

# ── Agent Loop Limits ──────────────────────────────────────
MAX_ITERATIONS: int = 3
MAX_OBSERVATION_CHARS: int = 300  # truncate tool outputs stored in trace

# ── Memory Auto-Save Triggers ──────────────────────────────
# If the user's message contains any of these substrings (case-insensitive),
# the observe node automatically saves it as a long-term memory.
AUTO_SAVE_TRIGGERS: list[str] = [
    "my name is",
    "i like",
    "i love",
    "i prefer",
    "i work",
    "i live",
    "i have",
    "remember that",
    "my favorite",
    "i am",
    "i'm",
]
