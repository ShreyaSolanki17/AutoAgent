"""FastAPI route handlers — chat, memories, auth, health.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.logging_config import get_logger

logger = get_logger(__name__)

from backend.models.schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    LoginRequest,
    MemoryItem,
    MemoryListResponse,
    MemoryStatsResponse,
    RegisterRequest,
    UserResponse,
)
from backend.agent.graph import run_agent
from backend.memory.store import (
    _get_connection,
    get_all_memories,
    delete_memory,
    delete_expired_memories,
    find_duplicate_groups,
    merge_duplicate_group,
    get_memory_stats,
    prune_user_memories,
    compute_importance,
    get_cleanup_status,
    start_auto_cleanup,
    stop_auto_cleanup,
)
from backend.middleware.safety import sanitize_input, filter_output, _MAX_MESSAGE_LENGTH, _MAX_HISTORY_LENGTH
from backend.middleware.usage import record_request, get_usage_stats, get_cost
from backend.evaluation.judge import evaluate_response, compute_trace_hash
from backend.evaluation.store import save_evaluation, get_evaluation_stats, get_recent_evaluations
from backend.models.schemas import (
    SubmitFeedbackRequest,
    EvaluationScores,
    EvaluationResponse,
)
from backend.auth import (
    authenticate_user,
    create_user,
    get_user_by_id,
    _decode_token,
)

router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """Extract and validate the JWT from the Authorization header.

    Raises 401 if the token is missing or invalid.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing access token")

    payload = _decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


# ── Chat ───────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 10  # Keep last 5 user/assistant exchanges to limit prompt growth

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Send a message to the agent and get a response."""
    # ── Input sanitization ──
    clean_message, issues = sanitize_input(request.message)
    if any("blocked" in issue for issue in issues):
        logger.warning(
            "chat.input_blocked",
            user_id=current_user["id"],
            issues=issues,
        )
        raise HTTPException(
            status_code=400,
            detail="Message blocked: potential prompt injection detected.",
        )

    if issues:
        logger.info(
            "chat.input_sanitized",
            user_id=current_user["id"],
            issues=issues,
        )

    # ── History validation ──
    history = request.history or []
    if len(history) > _MAX_HISTORY_LENGTH:
        history = history[-_MAX_HISTORY_LENGTH:]

    # Trim history to prevent unbounded prompt growth across turns
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    logger.info(
        "chat.request_received",
        user_id=current_user["id"],
        message_length=len(clean_message),
        history_length=len(trimmed_history),
    )

    result = run_agent(clean_message, user_id=current_user["id"], history=trimmed_history)

    trace = [msg["content"] for msg in result.get("messages", [])]

    # ── Output filtering ──
    raw_answer = result.get("final_answer", "")
    filtered_answer, was_filtered = filter_output(raw_answer)
    if was_filtered:
        logger.warning(
            "chat.output_filtered",
            user_id=current_user["id"],
            reason="potential_system_leak",
        )
        trace = ["[Output filtered for safety]"]

    # ── Generate message_id for correlation ──
    import hashlib as _hashlib
    import time as _time
    message_id = _hashlib.sha256(
        f"{current_user['id']}:{clean_message}:{_time.time_ns()}".encode()
    ).hexdigest()[:16]

    # ── LLM-as-judge evaluation (non-blocking — never breaks chat) ──
    judge_scores = None
    try:
        judge_scores = evaluate_response(
            user_goal=clean_message,
            final_answer=filtered_answer,
            trace=trace,
        )
    except Exception as e:
        logger.warning("chat.judge_failed", error=str(e))

    # ── Persist evaluation to DB (non-blocking — never breaks chat) ──
    trace_hash = compute_trace_hash(trace) if trace else None
    try:
        save_evaluation(
            user_id=current_user["id"],
            message_id=message_id,
            user_goal=clean_message,
            final_answer=filtered_answer,
            trace_hash=trace_hash,
            judge_scores=judge_scores,
        )
    except Exception as e:
        logger.warning("chat.eval_save_failed", error=str(e))

    # ── Record usage analytics ──
    usage = result.get("usage", {})
    record_request(
        user_id=current_user["id"],
        endpoint="/api/chat",
        provider=usage.get("provider") or None,
        model=usage.get("model") or None,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cost_usd=usage.get("cost_usd", 0.0),
        response_time_ms=usage.get("response_time_ms", 0),
        status_code=200,
    )

    logger.info(
        "chat.response_sent",
        user_id=current_user["id"],
        iterations=result["iterations"],
        tools_used=[e["tool"] for e in result.get("used_tools", [])],
        answer_length=len(filtered_answer),
        filtered=was_filtered,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cost_usd=usage.get("cost_usd", 0.0),
        response_time_ms=usage.get("response_time_ms", 0),
    )

    # ── Build inline evaluation for response (if judge ran) ──
    inline_eval = None
    if judge_scores and judge_scores.get("overall") is not None:
        inline_eval = EvaluationResponse(
            message_id=message_id,
            scores=EvaluationScores(
                relevance=judge_scores.get("relevance"),
                accuracy=judge_scores.get("accuracy"),
                helpfulness=judge_scores.get("helpfulness"),
                safety=judge_scores.get("safety"),
                overall=judge_scores.get("overall"),
            ),
            judge_reasoning=judge_scores.get("reasoning"),
            evaluation_type="auto",
        )

    return ChatResponse(
        final_answer=filtered_answer,
        trace=trace,
        message_id=message_id,
        evaluation=inline_eval,
    )


# ── Memory ─────────────────────────────────────────────

@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    current_user: dict = Depends(get_current_user),
    include_expired: bool = False,
):
    """List all stored memories for the current user."""
    raw = get_all_memories(user_id=current_user["id"], include_expired=include_expired)
    memories = [
        MemoryItem(
            id=m["id"],
            content=m["content"],
            created_at=m["created_at"],
            updated_at=m.get("updated_at"),
            expires_at=m.get("expires_at"),
            importance=m.get("importance", 0.5),
            last_accessed=m.get("last_accessed"),
        )
        for m in raw
    ]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.delete("/memories/{memory_id}")
async def remove_memory(
    memory_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Delete a memory by ID (only if it belongs to the current user)."""
    deleted = delete_memory(memory_id, user_id=current_user["id"])

    if not deleted:
        return {"error": f"Memory {memory_id} not found"}
    return {"status": "deleted", "id": memory_id}


@router.get("/memories/stats", response_model=MemoryStatsResponse)
async def memory_stats(current_user: dict = Depends(get_current_user)):
    """Get memory statistics for the current user."""
    stats = get_memory_stats(user_id=current_user["id"])
    return MemoryStatsResponse(**stats)


@router.post("/memories/cleanup")
async def cleanup_expired_memories(current_user: dict = Depends(get_current_user)):
    """Delete all expired memories for the current user.

    Returns the number of deleted memories.
    """
    count = delete_expired_memories(user_id=current_user["id"])
    return {"status": "ok", "deleted": count}


@router.get("/memories/dedup/groups")
async def find_memory_duplicates(
    current_user: dict = Depends(get_current_user),
    threshold: float = 0.95,
):
    """Find groups of near-duplicate memories.

    Returns groups of memories that are semantically similar
    above the given threshold (default 0.95).
    """
    groups = find_duplicate_groups(user_id=current_user["id"], threshold=threshold)
    return {"groups": groups, "group_count": len(groups)}


@router.post("/memories/dedup/merge")
async def merge_memory_duplicates(
    keep_id: int,
    remove_ids: list[int],
    current_user: dict = Depends(get_current_user),
):
    """Merge a group of duplicate memories.

    Keeps the memory with `keep_id` and deletes all memories
    in `remove_ids`. All must belong to the current user.
    """
    success = merge_duplicate_group(
        keep_id=keep_id,
        remove_ids=remove_ids,
        user_id=current_user["id"],
    )
    if not success:
        raise HTTPException(status_code=404, detail="Keep memory not found or does not belong to you")
    return {"status": "merged", "kept": keep_id, "removed": remove_ids}


@router.post("/memories/prune")
async def prune_memories(
    current_user: dict = Depends(get_current_user),
    max_memories: int = 200,
):
    """Manually trigger memory pruning for the current user.

    Deletes lowest-importance memories until count is under max_memories.
    High-importance memories (score >= 0.8) are never pruned.
    """
    result = prune_user_memories(user_id=current_user["id"], max_memories=max_memories)
    return {"status": "ok", **result}


@router.get("/memories/importance/preview")
async def preview_importance(content: str):
    """Preview what importance score a piece of text would receive.

    Useful for debugging / tuning importance patterns.
    """
    score = compute_importance(content)
    return {"content": content, "importance": score}


@router.get("/memories/cleanup/status")
async def cleanup_scheduler_status(current_user: dict = Depends(get_current_user)):
    """Get the status of the automatic cleanup scheduler."""
    return get_cleanup_status()


@router.post("/memories/cleanup/start")
async def manual_start_cleanup(current_user: dict = Depends(get_current_user)):
    """Start the auto cleanup scheduler (admin endpoint)."""
    start_auto_cleanup()
    return {"status": "started", **get_cleanup_status()}


@router.post("/memories/cleanup/stop")
async def manual_stop_cleanup(current_user: dict = Depends(get_current_user)):
    """Stop the auto cleanup scheduler (admin endpoint)."""
    stop_auto_cleanup()
    return {"status": "stopped", **get_cleanup_status()}


# ── Usage Analytics ─────────────────────────────────────

@router.get("/usage/stats")
async def usage_stats(
    current_user: dict = Depends(get_current_user),
    days: int = 7,
):
    """Get usage analytics for the current user.

    Returns token counts, costs, response times, and per-day breakdown.
    """
    stats = get_usage_stats(user_id=current_user["id"], days=days)
    return stats


@router.get("/usage/cost/estimate")
async def cost_estimate(
    input_tokens: int,
    output_tokens: int,
    provider: str = "groq",
    model: str = "llama-3.3-70b-versatile",
):
    """Estimate the cost of an LLM call without making it.

    Useful for frontend cost previews.
    """
    cost = get_cost(provider, model, input_tokens, output_tokens)
    return {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
    }


# ── Evaluation / Feedback ─────────────────────────────

@router.post("/evaluations/feedback")
async def submit_feedback(
    request: SubmitFeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit thumbs up/down feedback for a specific response.

    The message_id is the correlation ID returned in the ChatResponse.
    Upserts into the evaluations table — safe to call multiple times.
    """
    if request.feedback not in (-1, 1):
        raise HTTPException(status_code=400, detail="feedback must be 1 (up) or -1 (down)")

    eval_id = save_evaluation(
        user_id=current_user["id"],
        message_id=request.message_id,
        user_goal="",  # filled from existing record on upsert
        final_answer="",
        user_feedback=request.feedback,
        user_comment=request.comment,
    )

    logger.info(
        "eval.feedback_submitted",
        user_id=current_user["id"],
        message_id=request.message_id,
        feedback=request.feedback,
        evaluation_id=eval_id,
    )

    return {"status": "recorded", "evaluation_id": eval_id}


@router.get("/evaluations/stats")
async def evaluation_stats(
    current_user: dict = Depends(get_current_user),
    days: int = 7,
):
    """Get evaluation statistics for the current user.

    Returns totals, per-dimension averages, daily breakdown, and score distribution.
    """
    stats = get_evaluation_stats(user_id=current_user["id"], days=days)
    return stats


@router.get("/evaluations/recent")
async def recent_evaluations(
    current_user: dict = Depends(get_current_user),
    limit: int = 20,
):
    """Get recent evaluations for the current user (for feedback loop analysis)."""
    evals = get_recent_evaluations(user_id=current_user["id"], limit=limit)
    return {"evaluations": evals, "total": len(evals)}


# ── Auth ───────────────────────────────────────────────

@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """Register a new user account and return an access token."""
    try:
        user = create_user(request.email, request.name, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {exc}")

    token_data = authenticate_user(request.email, request.password)
    return AuthResponse(**token_data)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Log in with email and password. Returns an access token."""
    try:
        token_data = authenticate_user(request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    return AuthResponse(**token_data)


@router.get("/auth/me", response_model=UserResponse)
async def me(token: str):
    """Return the current user from a Bearer token (query param for simplicity)."""
    payload = _decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserResponse(**user)


# ── Health ─────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
