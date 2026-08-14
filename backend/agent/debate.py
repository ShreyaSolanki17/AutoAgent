"""Multi-agent debate mode.

Step 2: independent candidate generation — each of the 3 configured
providers answers the same query concurrently, with no visibility into
each other's answers.

Step 3 (current): critique round — each agent is shown the other two's
answers and produces one revised answer. Capped at exactly 1 round.

Later steps add: judge/synthesis, and graph wiring + triggering. Kept as
a standalone module (not yet graph nodes) since it has nowhere to route
to until those steps land.
"""

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
