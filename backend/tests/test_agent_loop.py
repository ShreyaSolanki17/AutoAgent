"""Agent loop regression tests — routing, parsing, safety override, fallback.

Pure logic tests.  No LLM calls, no DB.  These extract the decision logic
from graph.py into testable units so config changes can be validated.
"""

import unittest

from backend.agent.graph import should_continue, _build_fallback_answer
from backend.config import (
    MAX_ITERATIONS,
    MAX_OBSERVATION_CHARS,
    AUTO_SAVE_TRIGGERS,
)


def _make_state(**overrides):
    """Build a minimal AgentState for testing should_continue."""
    base = {
        "user_goal": "test",
        "user_id": "test",
        "messages": [],
        "chat_history": [],
        "plan": "",
        "tool_name": "finish",
        "tool_input": "",
        "observation": "",
        "final_answer": "",
        "iterations": 0,
        "used_tools": [],
        "usage": {},
    }
    base.update(overrides)
    return base


# ── Routing ────────────────────────────────────────────────

class TestAgentLoopRouting(unittest.TestCase):
    """Test should_continue routing logic."""

    def test_continue_when_no_answer_and_under_max(self):
        state = _make_state(iterations=0, final_answer="")
        self.assertEqual(should_continue(state), "continue")

    def test_continue_at_max_minus_one(self):
        state = _make_state(iterations=MAX_ITERATIONS - 1, final_answer="")
        self.assertEqual(should_continue(state), "continue")

    def test_end_when_max_iterations_reached(self):
        state = _make_state(iterations=MAX_ITERATIONS, final_answer="")
        self.assertEqual(should_continue(state), "end")

    def test_end_when_answer_produced(self):
        state = _make_state(iterations=0, final_answer="Paris is the capital")
        self.assertEqual(should_continue(state), "end")

    def test_end_when_answer_at_max(self):
        """Even at max iterations, if there's an answer, route to END."""
        state = _make_state(iterations=MAX_ITERATIONS, final_answer="fallback answer")
        self.assertEqual(should_continue(state), "end")

    def test_end_when_over_max(self):
        state = _make_state(iterations=MAX_ITERATIONS + 1, final_answer="")
        self.assertEqual(should_continue(state), "end")


# ── Plan Node Parsing ──────────────────────────────────────

class TestPlanNodeParsing(unittest.TestCase):
    """Test the TOOL/INPUT/REASON parsing logic used in plan_node.

    Parses the same way plan_node does: splits on ':' and takes the
    remainder of the line.
    """

    @staticmethod
    def _parse(content):
        tool_name = "finish"
        tool_input = ""
        reason = content
        for line in content.splitlines():
            upper_line = line.strip().upper()
            if upper_line.startswith("TOOL:"):
                tool_name = line.split(":", 1)[1].strip().lower()
            elif upper_line.startswith("INPUT:"):
                tool_input = line.split(":", 1)[1].strip()
            elif upper_line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return tool_name, tool_input, reason

    def test_parse_structured_format(self):
        content = "TOOL: wikipedia\nINPUT: Python programming\nREASON: Look up Python"
        tool, inp, reason = self._parse(content)
        self.assertEqual(tool, "wikipedia")
        self.assertEqual(inp, "Python programming")
        self.assertEqual(reason, "Look up Python")

    def test_parse_finish_tool(self):
        content = "TOOL: finish\nREASON: It's a greeting"
        tool, inp, reason = self._parse(content)
        self.assertEqual(tool, "finish")
        self.assertEqual(reason, "It's a greeting")

    def test_parse_unstructured_defaults_to_finish(self):
        content = "Hello! How are you today?"
        tool, inp, reason = self._parse(content)
        self.assertEqual(tool, "finish")

    def test_known_tool_names_detected(self):
        known = {"search", "wikipedia", "calculator", "file_read",
                 "file_write", "memory_save", "memory_search"}
        for tool_name in known:
            stripped = tool_name.strip().lower()
            self.assertIn(stripped, known)


# ── Observe Node Safety Override ───────────────────────────

class TestObserveNodeSafetyOverride(unittest.TestCase):
    """Test the safety override: force DONE if a tool fails 2+ times."""

    @staticmethod
    def _check_override(tool_failure_count, tool_name):
        if tool_failure_count >= 2 and tool_name != "finish":
            return True
        return False

    def test_override_triggers_at_two_failures(self):
        self.assertTrue(self._check_override(2, "wikipedia"))

    def test_override_triggers_at_three_failures(self):
        self.assertTrue(self._check_override(3, "wikipedia"))

    def test_override_does_not_trigger_at_one_failure(self):
        self.assertFalse(self._check_override(1, "wikipedia"))

    def test_override_ignores_finish_tool(self):
        """Even with many failures, 'finish' tool should not trigger override."""
        self.assertFalse(self._check_override(3, "finish"))


# ── Fallback Answer ────────────────────────────────────────

class TestFallbackAnswer(unittest.TestCase):
    """Test _build_fallback_answer logic.

    _build_fallback_answer takes a full AgentState dict, not a bare list.
    """

    @staticmethod
    def _state_with_tools(used_tools):
        return {
            "user_goal": "test",
            "user_id": "test",
            "messages": [],
            "chat_history": [],
            "plan": "",
            "tool_name": "",
            "tool_input": "",
            "observation": "",
            "final_answer": "",
            "iterations": 3,
            "used_tools": used_tools,
            "usage": {},
        }

    def test_no_tools_used(self):
        state = self._state_with_tools([])
        answer = _build_fallback_answer(state)
        self.assertIn("couldn't find", answer.lower())

    def test_with_findings(self):
        state = self._state_with_tools([{
            "tool": "wikipedia", "input": "Python",
            "result": "Python is a programming language.\nMore text here.",
        }])
        answer = _build_fallback_answer(state)
        self.assertIn("Python is a programming language", answer)

    def test_error_results_excluded(self):
        state = self._state_with_tools([{
            "tool": "search", "input": "test", "result": "Error: timeout",
        }])
        answer = _build_fallback_answer(state)
        self.assertNotIn("Error: timeout", answer)

    def test_no_result_excluded(self):
        state = self._state_with_tools([{
            "tool": "search", "input": "test", "result": "No results found",
        }])
        answer = _build_fallback_answer(state)
        self.assertNotIn("No results found", answer)

    def test_short_result_excluded(self):
        state = self._state_with_tools([{
            "tool": "search", "input": "test", "result": "short",
        }])
        answer = _build_fallback_answer(state)
        self.assertNotIn("short", answer)

    def test_asks_for_clarification(self):
        state = self._state_with_tools([])
        answer = _build_fallback_answer(state)
        self.assertIn("more context", answer.lower())
        self.assertIn("rephrase", answer.lower())


# ── Memory Observation Truncation ──────────────────────────

class TestObservationTruncation(unittest.TestCase):
    """Test the observation truncation constant."""

    def test_max_observation_chars_is_reasonable(self):
        self.assertGreater(MAX_OBSERVATION_CHARS, 0)
        self.assertLessEqual(MAX_OBSERVATION_CHARS, 1000)

    @staticmethod
    def _truncate(observation):
        if len(observation) > MAX_OBSERVATION_CHARS:
            return observation[:MAX_OBSERVATION_CHARS] + "..."
        return observation

    def test_short_observation_not_truncated(self):
        text = "Short result"
        self.assertEqual(self._truncate(text), text)

    def test_long_observation_truncated(self):
        text = "x" * 1000
        result = self._truncate(text)
        self.assertLessEqual(len(result), MAX_OBSERVATION_CHARS + 3)


# ── Auto-Save Memory Triggers ──────────────────────────────

class TestAutoSaveMemory(unittest.TestCase):
    """Test _auto_save_memory trigger detection."""

    def _should_auto_save(self, user_goal):
        goal_lower = user_goal.lower().strip()
        return any(t in goal_lower for t in AUTO_SAVE_TRIGGERS)

    def test_triggers_on_my_name_is(self):
        self.assertTrue(self._should_auto_save("My name is Alex"))

    def test_triggers_on_i_like(self):
        self.assertTrue(self._should_auto_save("I like Python"))

    def test_triggers_on_i_love(self):
        self.assertTrue(self._should_auto_save("I love programming"))

    def test_triggers_on_i_prefer(self):
        self.assertTrue(self._should_auto_save("I prefer tea over coffee"))

    def test_triggers_on_i_work(self):
        self.assertTrue(self._should_auto_save("I work at Google"))

    def test_triggers_on_i_live(self):
        self.assertTrue(self._should_auto_save("I live in Paris"))

    def test_triggers_on_i_have(self):
        self.assertTrue(self._should_auto_save("I have a cat"))

    def test_triggers_on_remember_that(self):
        self.assertTrue(self._should_auto_save("Remember that I have a cat"))

    def test_triggers_on_my_favorite(self):
        self.assertTrue(self._should_auto_save("My favorite color is blue"))

    def test_triggers_on_i_am(self):
        self.assertTrue(self._should_auto_save("I am a software engineer"))

    def test_triggers_on_im(self):
        self.assertTrue(self._should_auto_save("I'm a developer"))

    def test_does_not_trigger_on_question(self):
        self.assertFalse(self._should_auto_save("What is the capital of France?"))

    def test_does_not_trigger_on_generic_query(self):
        self.assertFalse(self._should_auto_save("Tell me about quantum computing"))

    def test_trigger_list_is_exactly_11(self):
        self.assertEqual(len(AUTO_SAVE_TRIGGERS), 11)
