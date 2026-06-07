"""Test LLM fallback chain — mocks each provider to verify cascade order.

Run:  python -m pytest backend/test_fallback.py -v
"""

import os
import unittest
from unittest.mock import patch, MagicMock

# Ensure .env is loaded before importing llm module
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend.llm import call_llm, _PROVIDERS


class TestFallbackChain(unittest.TestCase):
    """Verify that providers are tried in order and errors cascade."""

    def _fake_messages(self):
        return [{"role": "user", "content": "Hello"}]

    # ── Test 1: First provider (Groq) succeeds ────────────────
    @patch("backend.llm.Groq")
    def test_groq_succeeds_no_fallback(self, mock_groq):
        """If Groq works, no fallback should be attempted."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "groq reply"
        mock_groq.return_value = mock_client

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake_groq_key"}):
            result = call_llm(self._fake_messages())

        self.assertEqual(result, "groq reply")
        mock_groq.assert_called_once()

    # ── Test 2: Groq fails → Google succeeds ──────────────────
    @patch("backend.llm.Groq")
    @patch("openai.OpenAI")
    def test_groq_fails_google_succeeds(self, mock_openai, mock_groq):
        """If Groq throws, should fall back to Google."""
        mock_groq.side_effect = Exception("Groq rate limited")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "google reply"
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {
            "GROQ_API_KEY": "fake_groq_key",
            "GOOGLE_API_KEY": "fake_google_key",
        }):
            result = call_llm(self._fake_messages())

        self.assertEqual(result, "google reply")

    # ── Test 3: Groq + Google fail → OpenRouter succeeds ──────
    @patch("backend.llm.Groq")
    @patch("openai.OpenAI")
    def test_groq_google_fail_openrouter_succeeds(self, mock_openai, mock_groq):
        """If Groq and Google both fail, should fall back to OpenRouter."""
        mock_groq.side_effect = Exception("Groq rate limited")

        call_count = 0
        def openai_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_client = MagicMock()
            if call_count == 1:
                # First call (Google) fails
                raise Exception("Google quota exceeded")
            # Second call (OpenRouter) succeeds
            mock_client.chat.completions.create.return_value.choices[0].message.content = "openrouter reply"
            return mock_client

        mock_openai.side_effect = openai_side_effect

        with patch.dict(os.environ, {
            "GROQ_API_KEY": "fake_groq_key",
            "GOOGLE_API_KEY": "fake_google_key",
            "OPENROUTER_API_KEY": "fake_openrouter_key",
        }):
            result = call_llm(self._fake_messages())

        self.assertEqual(result, "openrouter reply")

    # ── Test 4: All providers fail → RuntimeError ─────────────
    @patch("backend.llm.Groq")
    @patch("openai.OpenAI")
    def test_all_fail_raises_runtime_error(self, mock_openai, mock_groq):
        """If every provider fails, should raise RuntimeError with all errors."""
        mock_groq.side_effect = Exception("Groq down")
        mock_openai.side_effect = Exception("All down")

        with patch.dict(os.environ, {
            "GROQ_API_KEY": "fake_groq_key",
            "GOOGLE_API_KEY": "fake_google_key",
            "OPENROUTER_API_KEY": "fake_openrouter_key",
        }):
            with self.assertRaises(RuntimeError) as ctx:
                call_llm(self._fake_messages())

        error_msg = str(ctx.exception)
        self.assertIn("All LLM providers failed", error_msg)
        self.assertIn("groq", error_msg.lower())
        self.assertIn("google", error_msg.lower())
        self.assertIn("openrouter", error_msg.lower())

    # ── Test 5: No API keys configured → RuntimeError ─────────
    def test_no_keys_raises_runtime_error(self):
        """If no API keys are set, should raise RuntimeError."""
        env_clean = {k: v for k, v in os.environ.items()
                     if not k.endswith("_API_KEY")}
        with patch.dict(os.environ, env_clean, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                call_llm(self._fake_messages())

        self.assertIn("All LLM providers failed", str(ctx.exception))

    # ── Test 6: Provider order is correct ─────────────────────
    def test_provider_priority_order(self):
        """Providers should be ordered: groq → google → openrouter."""
        names = [p["name"] for p in _PROVIDERS]
        self.assertEqual(names, ["groq", "google", "openrouter"])

    # ── Test 7: Only Google key set (no Groq) → uses Google ──
    @patch("openai.OpenAI")
    def test_only_google_key(self, mock_openai):
        """If only GOOGLE_API_KEY is set, should use Google directly."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "google reply"
        mock_openai.return_value = mock_client

        env_clean = {k: v for k, v in os.environ.items()
                     if not k.endswith("_API_KEY")}
        env_clean["GOOGLE_API_KEY"] = "fake_google_key"

        with patch.dict(os.environ, env_clean, clear=True):
            result = call_llm(self._fake_messages())

        self.assertEqual(result, "google reply")


if __name__ == "__main__":
    unittest.main(verbosity=2)
