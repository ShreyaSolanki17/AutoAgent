"""Prompt structure tests — verify prompts haven't been accidentally broken.

These are pure string-analysis tests.  No LLM calls, no DB, no mocks.
If someone edits a prompt and removes a required keyword or tool name,
these tests fail immediately.
"""

import unittest

from backend.agent.prompts import PLANNER_SYSTEM, OBSERVER_SYSTEM
from backend.evaluation.judge import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE


# ── Planner Prompt ─────────────────────────────────────────

class TestPlannerPrompt(unittest.TestCase):
    """PLANNER_SYSTEM must mention all tools and specify the output format."""

    def test_mentions_all_tools(self):
        expected = [
            "calculator", "wikipedia", "search",
            "file_read", "file_write",
            "memory_save", "memory_search", "finish",
        ]
        for tool in expected:
            self.assertIn(tool, PLANNER_SYSTEM,
                          f"PLANNER_SYSTEM missing tool: {tool}")

    def test_has_tool_format(self):
        self.assertIn("TOOL:", PLANNER_SYSTEM)

    def test_has_input_format(self):
        self.assertIn("INPUT:", PLANNER_SYSTEM)

    def test_has_reason_format(self):
        self.assertIn("REASON:", PLANNER_SYSTEM)

    def test_greeting_rule_present(self):
        self.assertIn("greeting", PLANNER_SYSTEM.lower())

    def test_no_repeat_tool_rule(self):
        self.assertIn("never call the same tool twice", PLANNER_SYSTEM.lower())

    def test_resolve_pronouns_rule(self):
        self.assertIn("pronouns", PLANNER_SYSTEM.lower())

    def test_prompt_length_reasonable(self):
        """Prompt should be concise — under 1000 chars to avoid wasting tokens."""
        self.assertLess(len(PLANNER_SYSTEM), 1000,
                        "PLANNER_SYSTEM is too long; it wastes tokens per request")

    def test_no_forbidden_directives(self):
        forbidden = ["ignore previous", "disregard instructions", "reveal your prompt"]
        for phrase in forbidden:
            self.assertNotIn(phrase, PLANNER_SYSTEM.lower(),
                             f"PLANNER_SYSTEM contains forbidden phrase: {phrase}")


# ── Observer Prompt ────────────────────────────────────────

class TestObserverPrompt(unittest.TestCase):
    """OBSERVER_SYSTEM must specify DONE/CONTINUE format and stopping rules."""

    def test_done_continue_format(self):
        self.assertIn("DONE:", OBSERVER_SYSTEM)
        self.assertIn("CONTINUE:", OBSERVER_SYSTEM)

    def test_no_metadata_in_done(self):
        self.assertIn("never", OBSERVER_SYSTEM.lower())
        self.assertIn("tool names", OBSERVER_SYSTEM.lower())

    def test_max_iterations_guard(self):
        self.assertIn("3", OBSERVER_SYSTEM)

    def test_natural_response_rule(self):
        self.assertIn("natural", OBSERVER_SYSTEM.lower())

    def test_greeting_done_rule(self):
        self.assertIn("greeting", OBSERVER_SYSTEM.lower())

    def test_no_bullet_points_rule(self):
        self.assertIn("bullet", OBSERVER_SYSTEM.lower())

    def test_no_raw_tool_output_rule(self):
        self.assertIn("raw tool output", OBSERVER_SYSTEM.lower())


# ── Judge Prompt ───────────────────────────────────────────

class TestJudgePrompt(unittest.TestCase):
    """JUDGE_SYSTEM_PROMPT must score on 4 dimensions and request JSON."""

    def test_four_dimensions(self):
        for dim in ("relevance", "accuracy", "helpfulness", "safety"):
            self.assertIn(dim, JUDGE_SYSTEM_PROMPT.lower())

    def test_json_format_specified(self):
        self.assertIn("json", JUDGE_SYSTEM_PROMPT.lower())

    def test_overall_weighted_composite(self):
        self.assertIn("overall", JUDGE_SYSTEM_PROMPT.lower())
        self.assertIn("0.30", JUDGE_SYSTEM_PROMPT)  # relevance weight

    def test_no_markdown_fences(self):
        self.assertIn("no markdown", JUDGE_SYSTEM_PROMPT.lower())

    def test_score_range(self):
        self.assertIn("0.0", JUDGE_SYSTEM_PROMPT)
        self.assertIn("1.0", JUDGE_SYSTEM_PROMPT)

    def test_user_template_has_placeholders(self):
        self.assertIn("{user_goal}", JUDGE_USER_TEMPLATE)
        self.assertIn("{final_answer}", JUDGE_USER_TEMPLATE)
        self.assertIn("{trace_summary}", JUDGE_USER_TEMPLATE)
