"""Evaluation tests — judge scoring with mocked LLM, JSON parsing, trace hashing."""

import json
import unittest
from unittest.mock import patch, MagicMock

from backend.evaluation.judge import (
    evaluate_response,
    compute_trace_hash,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
)
from backend.evaluation.store import save_evaluation, get_evaluation_stats


# ── Judge Prompt Structure ─────────────────────────────────

class TestJudgePrompt(unittest.TestCase):
    def test_judge_prompt_has_weighted_formula(self):
        self.assertIn("0.30", JUDGE_SYSTEM_PROMPT)
        self.assertIn("0.25", JUDGE_SYSTEM_PROMPT)
        self.assertIn("0.15", JUDGE_SYSTEM_PROMPT)

    def test_judge_user_template_has_placeholders(self):
        self.assertIn("{user_goal}", JUDGE_USER_TEMPLATE)
        self.assertIn("{final_answer}", JUDGE_USER_TEMPLATE)
        self.assertIn("{trace_summary}", JUDGE_USER_TEMPLATE)


# ── Judge Scoring (mocked LLM) ─────────────────────────────

class TestEvaluationScoring(unittest.TestCase):

    @patch("backend.evaluation.judge.call_llm")
    def test_successful_evaluation(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "relevance": 0.9, "accuracy": 0.8, "helpfulness": 0.85,
            "safety": 1.0, "overall": 0.88, "reasoning": "Good response",
        })
        result = evaluate_response("What is Python?", "Python is a language.", [])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["relevance"], 0.9)
        self.assertAlmostEqual(result["safety"], 1.0)

    @patch("backend.evaluation.judge.call_llm")
    def test_scores_clamped_to_range(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "relevance": 1.5, "accuracy": -0.2, "helpfulness": 0.5,
            "safety": 0.5, "overall": 2.0, "reasoning": "test",
        })
        result = evaluate_response("q", "a", [])
        for key in ("relevance", "accuracy", "helpfulness", "safety", "overall"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    @patch("backend.evaluation.judge.call_llm")
    def test_json_with_markdown_fences(self, mock_llm):
        mock_llm.return_value = (
            '```json\n'
            '{"relevance": 0.8, "accuracy": 0.7, "helpfulness": 0.6, '
            '"safety": 1.0, "overall": 0.75, "reasoning": "ok"}\n'
            '```'
        )
        result = evaluate_response("q", "a", [])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["relevance"], 0.8)

    @patch("backend.evaluation.judge.call_llm")
    def test_invalid_json_returns_none(self, mock_llm):
        mock_llm.return_value = "not json at all"
        result = evaluate_response("q", "a", [])
        self.assertIsNone(result)

    @patch("backend.evaluation.judge.call_llm")
    def test_llm_failure_returns_none(self, mock_llm):
        mock_llm.side_effect = RuntimeError("All providers down")
        result = evaluate_response("q", "a", [])
        self.assertIsNone(result)

    @patch("backend.evaluation.judge.call_llm")
    def test_missing_overall_recomputed(self, mock_llm):
        """If the judge forgets the overall field, we recompute it."""
        mock_llm.return_value = json.dumps({
            "relevance": 0.5, "accuracy": 0.5, "helpfulness": 0.5,
            "safety": 0.5, "reasoning": "ok",
            # no "overall"
        })
        result = evaluate_response("q", "a", [])
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["overall"])
        # overall = 0.30*0.5 + 0.30*0.5 + 0.25*0.5 + 0.15*0.5 = 0.5
        self.assertAlmostEqual(result["overall"], 0.5, places=2)

    @patch("backend.evaluation.judge.call_llm")
    def test_trace_is_summarized(self, mock_llm):
        """Long traces should be truncated in the judge prompt."""
        long_trace = ["Step " + str(i) + " x" * 100 for i in range(20)]
        mock_llm.return_value = json.dumps({
            "relevance": 0.5, "accuracy": 0.5, "helpfulness": 0.5,
            "safety": 0.5, "overall": 0.5, "reasoning": "ok",
        })
        result = evaluate_response("q", "a", long_trace)
        self.assertIsNotNone(result)


# ── Trace Hashing ──────────────────────────────────────────

class TestTraceHashing(unittest.TestCase):
    def test_hash_deterministic(self):
        trace = ["step1", "step2"]
        self.assertEqual(compute_trace_hash(trace), compute_trace_hash(trace))

    def test_hash_different_for_different_traces(self):
        self.assertNotEqual(compute_trace_hash(["a"]), compute_trace_hash(["b"]))

    def test_hash_is_16_chars(self):
        h = compute_trace_hash(["test"])
        self.assertEqual(len(h), 16)


# ── Evaluation Store ───────────────────────────────────────

class TestEvaluationStats(unittest.TestCase):
    def test_stats_error_response_has_period_days(self):
        """When DB/table is unavailable, the error response still has period_days."""
        stats = get_evaluation_stats(user_id="nonexistent_user")
        self.assertIn("period_days", stats)
        # Either we got data or an error — both are valid responses
        if "error" not in stats:
            self.assertIn("total_evaluations", stats)

    def test_stats_error_is_string(self):
        """If there's an error, it should be a string describing the issue."""
        stats = get_evaluation_stats(user_id="nonexistent_user")
        if "error" in stats:
            self.assertIsInstance(stats["error"], str)
