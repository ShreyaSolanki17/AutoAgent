"""Groq LLM client — single function to call Llama 3.3 70B via Groq API.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from project root (two levels up from this file: backend/llm.py -> root)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def call_llm(messages: list[dict[str, str]]) -> str:
    """Send a list of chat messages to Groq and return the response text.

    Args:
        messages: OpenAI-format chat messages, e.g.
                  [{"role": "user", "content": "Hello"}]

    Returns:
        The model's response text.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
        groq.GroqError: If the API call fails.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,       # low temp for consistent, focused responses
        max_tokens=4096,
    )

    return response.choices[0].message.content
