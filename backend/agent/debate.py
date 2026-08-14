"""Multi-agent debate mode.

Step 2 (current): independent candidate generation — each of the 3
configured providers answers the same query concurrently, with no
visibility into each other's answers yet.

Later steps add: critique (each agent sees the other two's answers),
judge/synthesis, and graph wiring + triggering. Kept as a standalone
module (not yet a graph node) since it has nothing to route to until
those steps land.
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
