"""Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel, EmailStr


# ── Chat ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Previous messages: [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    final_answer: str
    trace: list[str]


# ── Memory ────────────────────────────────────────────

class MemoryItem(BaseModel):
    id: int
    content: str
    created_at: str
    updated_at: str | None = None
    expires_at: str | None = None
    importance: float = 0.5
    last_accessed: str | None = None


class MemoryListResponse(BaseModel):
    memories: list[MemoryItem]
    total: int


class MemoryStatsResponse(BaseModel):
    total: int
    expired: int
    active: int
    avg_importance: float = 0.0


# ── Auth ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    name: str