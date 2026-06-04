"""FastAPI route handlers — chat, memories, auth, health.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.models.schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    LoginRequest,
    MemoryItem,
    MemoryListResponse,
    RegisterRequest,
    UserResponse,
)
from backend.agent.graph import run_agent
from backend.memory.store import _get_connection, get_all_memories, delete_memory
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

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Send a message to the agent and get a response."""
    result = run_agent(request.message, user_id=current_user["id"], history=request.history)

    trace = [msg["content"] for msg in result.get("messages", [])]

    return ChatResponse(
        final_answer=result.get("final_answer", ""),
        trace=trace,
    )


# ── Memory ─────────────────────────────────────────────

@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(current_user: dict = Depends(get_current_user)):
    """List all stored memories for the current user."""
    raw = get_all_memories(user_id=current_user["id"])
    memories = [
        MemoryItem(id=m["id"], content=m["content"], created_at=m["created_at"])
        for m in raw
    ]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.delete("/memories/{memory_id}")
async def remove_memory(
    memory_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Delete a memory by ID (only if it belongs to the current user)."""
    # Filter by user_id so users cannot delete each other's memories
    deleted = delete_memory(memory_id, user_id=current_user["id"])

    if not deleted:
        return {"error": f"Memory {memory_id} not found"}
    return {"status": "deleted", "id": memory_id}


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
