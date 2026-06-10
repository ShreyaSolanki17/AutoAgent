"""Config consistency tests — validate shared constants."""

import unittest

from backend.config import (
    PROVIDERS,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_ITERATIONS,
    MAX_OBSERVATION_CHARS,
    AUTO_SAVE_TRIGGERS,
)


class TestProviderConfig(unittest.TestCase):
    def test_provider_order(self):
        names = [p["name"] for p in PROVIDERS]
        self.assertEqual(names, ["groq", "google", "openrouter"])

    def test_all_providers_have_api_key_env(self):
        for p in PROVIDERS:
            self.assertIn("api_key_env", p)
            self.assertTrue(p["api_key_env"].endswith("_API_KEY"))

    def test_all_providers_have_models(self):
        for p in PROVIDERS:
            self.assertIn("model", p)
            self.assertTrue(len(p["model"]) > 0)

    def test_groq_has_no_base_url(self):
        groq = next(p for p in PROVIDERS if p["name"] == "groq")
        self.assertIsNone(groq["base_url"])

    def test_fallback_providers_have_base_url(self):
        for p in PROVIDERS:
            if p["name"] != "groq":
                self.assertIsNotNone(p["base_url"])


class TestGenerationParams(unittest.TestCase):
    def test_temperature_in_valid_range(self):
        self.assertGreaterEqual(TEMPERATURE, 0.0)
        self.assertLessEqual(TEMPERATURE, 2.0)

    def test_max_tokens_positive(self):
        self.assertGreater(MAX_TOKENS, 0)


class TestLoopLimits(unittest.TestCase):
    def test_max_iterations_positive(self):
        self.assertGreater(MAX_ITERATIONS, 0)
        self.assertLessEqual(MAX_ITERATIONS, 10)

    def test_max_observation_chars_positive(self):
        self.assertGreater(MAX_OBSERVATION_CHARS, 0)


class TestAutoSaveTriggers(unittest.TestCase):
    def test_triggers_is_list(self):
        self.assertIsInstance(AUTO_SAVE_TRIGGERS, list)

    def test_triggers_non_empty(self):
        self.assertGreater(len(AUTO_SAVE_TRIGGERS), 0)

    def test_key_triggers_present(self):
        for trigger in ("my name is", "i like", "i am", "i'm"):
            self.assertIn(trigger, AUTO_SAVE_TRIGGERS)
