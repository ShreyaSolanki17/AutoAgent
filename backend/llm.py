"""LLM client — multi-provider with automatic fallback (free-tier only).

Primary:  Groq        (llama-3.3-70b-versatile)
Fallback: Google AI   (gemini-2.0-flash)                  — free tier, no credit card
Fallback: OpenRouter  (openai/gpt-oss-20b:free)                  — free tier, no credit card

Add to .env (at least one):
  GROQ_API_KEY=gsk_your_key
  GOOGLE_API_KEY=your_google_key
  OPENROUTER_API_KEY=your_openrouter_key
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from project root (two levels up from this file: backend/llm.py -> root)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Provider configuration ────────────────────────────────
_PROVIDERS: list[dict] = [
    {
        "name": "groq",
        "model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "base_url": None,          # uses native Groq SDK
    },
    {
        "name": "google",
        "model": "gemini-2.0-flash",
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

# Shared generation settings
_TEMPERATURE = 0.1
_MAX_TOKENS = 4096


def _call_groq(messages: list[dict[str, str]], api_key: str) -> str:
    """Call Groq API (native SDK)."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
    )
    return response.choices[0].message.content


def _call_openai_compatible(
    messages: list[dict[str, str]],
    api_key: str,
    model: str,
    base_url: str,
) -> str:
    """Call any OpenAI-compatible endpoint (Google, OpenRouter, etc.)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
    )
    return response.choices[0].message.content


def call_llm(messages: list[dict[str, str]]) -> str:
    """Send chat messages to the LLM with automatic provider fallback.

    Tries each configured provider in order:
      1. Groq        (llama-3.3-70b-versatile)
      2. Google AI   (gemini-2.0-flash)                 — free tier
      3. OpenRouter  (openai/gpt-oss-20b:free)                 — free tier

    Reloads .env on every call so changes to API keys are picked up
    without restarting the server.

    Args:
        messages: OpenAI-format chat messages, e.g.
                  [{"role": "user", "content": "Hello"}]

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If no provider API key is set or all providers fail.
    """
    # Reload .env so key changes are picked up without server restart
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

    errors: list[str] = []

    for provider in _PROVIDERS:
        api_key = os.getenv(provider["api_key_env"])
        if not api_key:
            continue

        try:
            if provider["base_url"] is None:
                result = _call_groq(messages, api_key)
            else:
                result = _call_openai_compatible(
                    messages, api_key, provider["model"], provider["base_url"]
                )
            if len(errors):
                print(f"[LLM] {provider['name']} succeeded after {len(errors)} failure(s)")
            else:
                print(f"[LLM] {provider['name']} responded successfully")
            return result
        except Exception as e:
            error_msg = f"{provider['name']}: {e}"
            errors.append(error_msg)
            print(f"[LLM] {error_msg} — trying next provider...")
            continue

    # All providers exhausted
    raise RuntimeError(
        "All LLM providers failed. Errors:\n"
        + "\n".join(f"  • {err}" for err in errors)
        + f"\n\nSet at least one API key in .env: "
        + ", ".join(p["api_key_env"] for p in _PROVIDERS)
    )
