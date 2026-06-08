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

    logger.info(
        "chat.response_sent",
        user_id=current_user["id"],
        iterations=result["iterations"],
        tools_used=[e["tool"] for e in result.get("used_tools", [])],
        answer_length=len(filtered_answer),
        filtered=was_filtered,
    )

    return ChatResponse(
        final_answer=filtered_answer,
        trace=trace,
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
