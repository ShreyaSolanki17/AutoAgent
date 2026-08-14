"""Multi-agent debate mode.

Step 2: independent candidate generation — each of the 3 configured
providers answers the same query concurrently, with no visibility into
each other's answers.

Step 3: critique round — each agent is shown the other two's answers and
produces one revised answer. Capped at exactly 1 round.

Step 4: judge/synthesis (backend.evaluation.judge.judge_debate) — scores
the revised answers and synthesizes the final response.

Step 5 (current): run_debate() chains all three into one call with a
run_agent()-compatible return shape, so backend.agent.graph.run_agent()
can route into it when debate=True.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from backend.config import PROVIDERS
from backend.llm import call_llm_as
from backend.logging_config import get_logger

logger = get_logger(__name__)

DEBATER_SYSTEM_PROMPT = (
    "You are one of several independent experts answering the same question. "
    "Give your best, complete answer. Be concise but thorough — you may be asked "
    "to defend or revise this answer against other experts' answers next."
)


def _generate_one(provider: dict, user_goal: str) -> dict:
    """Generate a single provider's independent answer. Never raises —
    a failed/unconfigured provider comes back with answer="" and error set,
    so one bad provider can't take down the whole debate."""
    try:
        answer = call_llm_as(provider, [
            {"role": "system", "content": DEBATER_SYSTEM_PROMPT},
            {"role": "user", "content": user_goal},
        ])
        return {"provider": provider["name"], "model": provider["model"], "answer": answer.strip(), "error": None}
    except Exception as e:
        logger.warning("debate.generate_failed", provider=provider["name"], error=str(e))
        return {"provider": provider["name"], "model": provider["model"], "answer": "", "error": str(e)}


def generate_candidates(user_goal: str) -> list[dict]:
    """Run all configured providers concurrently to get independent answers.

    Args:
        user_goal: the user's question/goal, sent identically to all providers.

    Returns:
        One dict per provider in PROVIDERS order:
        {"provider", "model", "answer", "error"}.
        A provider with no configured key (or a failed call) has answer=""
        and error set — callers decide whether that's fatal.
    """
    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        results = list(pool.map(lambda p: _generate_one(p, user_goal), PROVIDERS))

    logger.info(
        "debate.candidates_generated",
        succeeded=[r["provider"] for r in results if not r["error"]],
        failed=[r["provider"] for r in results if r["error"]],
    )
    return results


CRITIC_SYSTEM_PROMPT = (
    "You are one of several independent experts who each already answered the "
    "same question. You will now see the other experts' answers. Critically "
    "compare them against your own — note anything they got wrong, missed, or "
    "did better. Then respond with ONLY your revised final answer, keeping what "
    "holds up and fixing what doesn't."
)


def _critique_one(candidate: dict, others: list[dict], user_goal: str) -> dict:
    """Have one agent critique+revise its answer given the other agents' answers.

    Falls back to the original answer (not an empty one) if the critique call
    itself fails, so a single flaky provider doesn't lose that agent's voice
    for the judge step.
    """
    others_text = "\n\n".join(f"{o['provider']}'s answer:\n{o['answer']}" for o in others)
    user_msg = (
        f"Original question: {user_goal}\n\n"
        f"Your original answer:\n{candidate['answer']}\n\n"
        f"Other experts' answers:\n{others_text}\n\n"
        "Give your revised final answer."
    )
    provider = next(p for p in PROVIDERS if p["name"] == candidate["provider"])
    try:
        revised = call_llm_as(provider, [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        return {**candidate, "revised_answer": revised.strip(), "critique_error": None}
    except Exception as e:
        logger.warning("debate.critique_failed", provider=candidate["provider"], error=str(e))
        return {**candidate, "revised_answer": candidate["answer"], "critique_error": str(e)}


def critique_round(user_goal: str, candidates: list[dict]) -> list[dict]:
    """Run one critique round: each successful candidate sees the other
    successful candidates' answers and revises its own, concurrently.

    Args:
        user_goal: same question passed to generate_candidates.
        candidates: output of generate_candidates().

    Returns:
        Same shape as candidates, each dict gaining "revised_answer" and
        "critique_error". Candidates that failed generation are passed
        through unchanged (revised_answer="", critique_error=original error).
        If fewer than 2 candidates succeeded, there's nothing to critique
        against — all candidates pass through with revised_answer=answer.
    """
    successful = [c for c in candidates if not c["error"]]

    if len(successful) < 2:
        logger.warning("debate.critique_skipped", successful_count=len(successful))
        return [
            {**c, "revised_answer": c["answer"], "critique_error": "skipped: fewer than 2 successful candidates to critique against"}
            for c in candidates
        ]

    tasks = [(c, [o for o in successful if o["provider"] != c["provider"]]) for c in successful]

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        revised = list(pool.map(lambda t: _critique_one(t[0], t[1], user_goal), tasks))

    revised_by_provider = {r["provider"]: r for r in revised}
    return [
        revised_by_provider.get(c["provider"], {**c, "revised_answer": "", "critique_error": c["error"]})
        for c in candidates
    ]


def run_debate(user_goal: str, user_id: str = "default") -> dict:
    """Run the full debate pipeline: independent generation -> critique -> judge.

    Returns a dict shaped like graph.run_agent()'s return value (final_answer,
    messages, iterations, used_tools, usage, ab_variant) so the /chat route
    needs no mode-specific branching for those fields — plus a "debate" key
    with the raw candidates/scores for callers that want the panel detail.
    """
    from backend.evaluation.judge import judge_debate  # local import: keeps the
    # judge <-> debate module boundary one-directional (judge.py never imports debate.py)

    start = time.time()
    logger.info("debate.run_started", user_id=user_id, goal=user_goal[:100])

    candidates = generate_candidates(user_goal)
    revised = critique_round(user_goal, candidates)
    judged = judge_debate(user_goal, revised)

    messages = [
        {"role": "assistant", "content": f"[Debate:{c['provider']}:initial] {c['answer'][:200] or c['error']}"}
        for c in candidates
    ] + [
        {"role": "assistant", "content": f"[Debate:{r['provider']}:revised] {r['revised_answer'][:200]}"}
        for r in revised
    ]

    if judged:
        final_answer = judged["synthesized_answer"]
        messages.append({"role": "assistant", "content": f"[Debate:judge] {judged['reasoning']}"})
    else:
        # Every candidate failed generation — genuinely nothing to judge or answer with.
        final_answer = "All debate agents failed to respond. Please try again."
        judged = {"scores": {}, "reasoning": "no candidates succeeded", "judge_model": "none"}

    elapsed_ms = round((time.time() - start) * 1000)
    logger.info(
        "debate.run_completed",
        user_id=user_id,
        elapsed_ms=elapsed_ms,
        succeeded=[c["provider"] for c in candidates if not c["error"]],
    )

    return {
        "final_answer": final_answer,
        "messages": messages,
        "iterations": 1,
        "used_tools": [],
        "ab_variant": None,
        "usage": {
            # ponytail: token/cost accounting isn't wired up for debate calls yet —
            # call_llm_as() discards each SDK response's usage_metadata. Real numbers
            # get measured properly in Step 7, not guessed here.
            "input_tokens": 0,
            "output_tokens": 0,
            "provider": "debate",
            "model": "+".join(p["name"] for p in PROVIDERS),
            "cost_usd": 0.0,
            "response_time_ms": elapsed_ms,
        },
        "debate": {
            "candidates": candidates,
            "revised": revised,
            "judge_scores": judged["scores"],
            "judge_reasoning": judged["reasoning"],
            "judge_model": judged["judge_model"],
        },
    }
