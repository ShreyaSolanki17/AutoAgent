"""Multi-agent debate mode tests — generation, critique cross-visibility,
judge/synthesis, pipeline composition, and triggering/fallback.

LLM calls are mocked throughout (via backend.agent.debate.call_llm_as and
backend.evaluation.judge.call_llm) — no real provider calls, no cost.
"""

import json
import os
import unittest
from unittest.mock import patch

from backend.agent.debate import generate_candidates, critique_round, run_debate
from backend.agent.graph import run_agent
from backend.config import PROVIDERS
from backend.evaluation.judge import judge_debate


# ── Step 2: Independent Generation ──────────────────────────

class TestGenerateCandidates(unittest.TestCase):

    @patch("backend.agent.debate.call_llm_as")
    def test_all_providers_called(self, mock_call):
        mock_call.side_effect = lambda provider, messages: f"answer from {provider['name']}"
        results = generate_candidates("question")
        self.assertEqual(len(results), len(PROVIDERS))
        self.assertEqual({r["provider"] for r in results}, {p["name"] for p in PROVIDERS})
        for r in results:
            self.assertIsNone(r["error"])
            self.assertIn("answer from", r["answer"])

    @patch("backend.agent.debate.call_llm_as")
    def test_one_provider_failing_does_not_affect_others(self, mock_call):
        def fake_call(provider, messages):
            if provider["name"] == PROVIDERS[1]["name"]:
                raise RuntimeError("boom")
            return "ok"
        mock_call.side_effect = fake_call

        results = generate_candidates("question")
        by_name = {r["provider"]: r for r in results}
        failed_name = PROVIDERS[1]["name"]
        self.assertIsNotNone(by_name[failed_name]["error"])
        self.assertEqual(by_name[failed_name]["answer"], "")
        for name, r in by_name.items():
            if name != failed_name:
                self.assertIsNone(r["error"])


# ── Step 3: Critique Round ──────────────────────────────────

class TestCritiqueRound(unittest.TestCase):

    @patch("backend.agent.debate.call_llm_as")
    def test_each_agent_sees_the_others_answers_not_its_own_as_other(self, mock_call):
        """The whole point of debate mode: critique prompts must actually
        contain the OTHER agents' answers, not be 3 isolated calls."""
        captured = {}

        def fake_call(provider, messages):
            captured[provider["name"]] = messages[1]["content"]
            return f"{provider['name']} revised"

        mock_call.side_effect = fake_call
        names = [p["name"] for p in PROVIDERS]
        candidates = [
            {"provider": n, "model": "m", "answer": f"{n.upper()}_ANSWER", "error": None}
            for n in names
        ]

        critique_round("some question", candidates)

        for n in names:
            others = [o for o in names if o != n]
            for other in others:
                self.assertIn(
                    f"{other.upper()}_ANSWER", captured[n],
                    f"{n}'s critique prompt should contain {other}'s answer",
                )

    @patch("backend.agent.debate.call_llm_as")
    def test_failed_candidate_passed_through_unchanged(self, mock_call):
        mock_call.side_effect = lambda provider, messages: "revised"
        names = [p["name"] for p in PROVIDERS]
        candidates = [
            {"provider": names[0], "model": "m", "answer": "A1", "error": None},
            {"provider": names[1], "model": "m", "answer": "A2", "error": None},
            {"provider": names[2], "model": "m", "answer": "", "error": "provider down"},
        ]
        result = critique_round("question", candidates)
        by_name = {r["provider"]: r for r in result}
        self.assertEqual(by_name[names[2]]["revised_answer"], "")
        self.assertEqual(by_name[names[2]]["critique_error"], "provider down")
        self.assertEqual(by_name[names[0]]["revised_answer"], "revised")

    def test_skips_critique_with_fewer_than_two_successful(self):
        candidates = [
            {"provider": "p1", "model": "m", "answer": "A1", "error": None},
            {"provider": "p2", "model": "m", "answer": "", "error": "down"},
            {"provider": "p3", "model": "m", "answer": "", "error": "down"},
        ]
        result = critique_round("question", candidates)
        # nothing to critique against -> revised_answer falls back to the original answer
        p1 = next(r for r in result if r["provider"] == "p1")
        self.assertEqual(p1["revised_answer"], "A1")
        self.assertIn("skipped", p1["critique_error"])


# ── Step 4: Judge / Synthesis ───────────────────────────────

class TestJudgeDebate(unittest.TestCase):

    @patch("backend.evaluation.judge.call_llm")
    def test_successful_synthesis(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "scores": {
                "groq": {"relevance": 0.9, "accuracy": 0.9, "helpfulness": 0.9, "safety": 1.0, "overall": 0.91},
                "google": {"relevance": 0.8, "accuracy": 0.8, "helpfulness": 0.8, "safety": 1.0, "overall": 0.81},
            },
            "synthesized_answer": "The combined best answer.",
            "reasoning": "groq was strongest",
        })
        candidates = [
            {"provider": "groq", "revised_answer": "groq's answer"},
            {"provider": "google", "revised_answer": "google's answer"},
        ]
        result = judge_debate("question", candidates)
        self.assertEqual(result["judge_model"], "inline")
        self.assertEqual(result["synthesized_answer"], "The combined best answer.")
        self.assertIn("groq", result["scores"])

    @patch("backend.evaluation.judge.call_llm")
    def test_bad_json_falls_back_to_first_candidate(self, mock_llm):
        mock_llm.return_value = "not valid json"
        candidates = [
            {"provider": "groq", "revised_answer": "groq's fallback-worthy answer"},
            {"provider": "google", "revised_answer": "google's answer"},
        ]
        result = judge_debate("question", candidates)
        self.assertEqual(result["judge_model"], "fallback")
        self.assertEqual(result["synthesized_answer"], "groq's fallback-worthy answer")

    @patch("backend.evaluation.judge.call_llm")
    def test_empty_synthesized_answer_triggers_fallback(self, mock_llm):
        mock_llm.return_value = json.dumps({"scores": {}, "synthesized_answer": "", "reasoning": "r"})
        candidates = [{"provider": "groq", "revised_answer": "groq's answer"}]
        result = judge_debate("question", candidates)
        self.assertEqual(result["judge_model"], "fallback")

    def test_no_valid_candidates_returns_none(self):
        candidates = [{"provider": "groq", "revised_answer": "", "error": "failed"}]
        self.assertIsNone(judge_debate("question", candidates))

    @patch("backend.evaluation.judge.call_llm")
    def test_scores_clamped_to_range(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "scores": {"groq": {"relevance": 1.5, "accuracy": -0.5, "helpfulness": 0.5, "safety": 0.5, "overall": 0.5}},
            "synthesized_answer": "answer",
            "reasoning": "r",
        })
        result = judge_debate("question", [{"provider": "groq", "revised_answer": "a"}])
        self.assertLessEqual(result["scores"]["groq"]["relevance"], 1.0)
        self.assertGreaterEqual(result["scores"]["groq"]["accuracy"], 0.0)


# ── Step 5: Pipeline Composition (run_debate) ───────────────

class TestRunDebate(unittest.TestCase):

    @patch("backend.evaluation.judge.judge_debate")
    @patch("backend.agent.debate.critique_round")
    @patch("backend.agent.debate.generate_candidates")
    def test_composes_pipeline_into_run_agent_shape(self, mock_gen, mock_critique, mock_judge):
        mock_gen.return_value = [{"provider": "groq", "model": "m", "answer": "a1", "error": None}]
        mock_critique.return_value = [{"provider": "groq", "model": "m", "answer": "a1", "error": None, "revised_answer": "a1r", "critique_error": None}]
        mock_judge.return_value = {
            "scores": {"groq": {"overall": 0.9}},
            "synthesized_answer": "final synthesized answer",
            "reasoning": "groq wins",
            "judge_model": "inline",
        }

        result = run_debate("question", user_id="u1")

        mock_gen.assert_called_once_with("question")
        mock_critique.assert_called_once_with("question", mock_gen.return_value)
        self.assertEqual(result["final_answer"], "final synthesized answer")
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["used_tools"], [])
        self.assertIsNone(result["ab_variant"])
        self.assertEqual(result["usage"]["provider"], "debate")
        self.assertEqual(result["debate"]["judge_model"], "inline")

    @patch("backend.evaluation.judge.judge_debate")
    @patch("backend.agent.debate.critique_round")
    @patch("backend.agent.debate.generate_candidates")
    def test_all_candidates_failed_returns_apology_not_crash(self, mock_gen, mock_critique, mock_judge):
        mock_gen.return_value = [{"provider": "groq", "model": "m", "answer": "", "error": "down"}]
        mock_critique.return_value = [{"provider": "groq", "model": "m", "answer": "", "error": "down", "revised_answer": "", "critique_error": "skipped"}]
        mock_judge.return_value = None

        result = run_debate("question")
        self.assertIn("failed", result["final_answer"].lower())
        self.assertEqual(result["debate"]["judge_model"], "none")


# ── Step 5: Triggering / Fallback ───────────────────────────

_FAKE_PROVIDERS_ALL_PRESENT = [
    {"name": "p1", "model": "m1", "api_key_env": "TEST_DEBATE_KEY_1", "base_url": None},
    {"name": "p2", "model": "m2", "api_key_env": "TEST_DEBATE_KEY_2", "base_url": None},
    {"name": "p3", "model": "m3", "api_key_env": "TEST_DEBATE_KEY_3", "base_url": None},
]

_FAKE_PROVIDERS_ONE_MISSING = [
    {"name": "p1", "model": "m1", "api_key_env": "TEST_DEBATE_KEY_1", "base_url": None},
    {"name": "p2", "model": "m2", "api_key_env": "TEST_DEBATE_KEY_2", "base_url": None},
    {"name": "p3", "model": "m3", "api_key_env": "DEFINITELY_MISSING_KEY_FOR_TEST", "base_url": None},
]


class TestDebateTriggering(unittest.TestCase):

    def test_routes_to_debate_when_all_three_keys_present(self):
        env = {p["api_key_env"]: "fake-key" for p in _FAKE_PROVIDERS_ALL_PRESENT}
        with patch("backend.agent.graph.PROVIDERS", _FAKE_PROVIDERS_ALL_PRESENT), \
             patch.dict(os.environ, env), \
             patch("backend.agent.debate.run_debate") as mock_run_debate:
            mock_run_debate.return_value = {"final_answer": "debated!", "usage": {}, "debate": {}}
            result = run_agent("question", user_id="u1", debate=True)
            mock_run_debate.assert_called_once_with("question", user_id="u1")
            self.assertEqual(result["final_answer"], "debated!")

    def test_debate_false_never_touches_run_debate(self):
        with patch("backend.agent.debate.run_debate") as mock_run_debate, \
             patch("backend.agent.graph.ChatGroq") as mock_groq:
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.content = "TOOL: finish\nREASON: Test"
            mock_response.usage_metadata = None
            mock_groq.return_value.invoke.return_value = mock_response
            run_agent("question", user_id="u1", debate=False)
            mock_run_debate.assert_not_called()

    def test_falls_back_to_single_agent_when_key_missing(self):
        env = {"TEST_DEBATE_KEY_1": "x", "TEST_DEBATE_KEY_2": "y"}  # 3rd deliberately unset
        with patch("backend.agent.graph.PROVIDERS", _FAKE_PROVIDERS_ONE_MISSING), \
             patch.dict(os.environ, env), \
             patch("backend.agent.debate.run_debate") as mock_run_debate, \
             patch("backend.agent.graph.ChatGroq") as mock_groq:
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.content = "TOOL: finish\nREASON: Test"
            mock_response.usage_metadata = None
            mock_groq.return_value.invoke.return_value = mock_response

            result = run_agent("question", user_id="u1", debate=True)

            mock_run_debate.assert_not_called()
            self.assertTrue(result["final_answer"])  # normal ReAct path still produced an answer


if __name__ == "__main__":
    unittest.main()
