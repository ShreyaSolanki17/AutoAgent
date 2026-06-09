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
    message_id: str = ""           # Correlation ID for feedback submission
    evaluation: "EvaluationResponse | None" = None  # Inline judge result (if available)


# ── Evaluation / Feedback ────────────────────────────

class EvaluationScores(BaseModel):
    relevance: float | None = None
    accuracy: float | None = None
    helpfulness: float | None = None
    safety: float | None = None
    overall: float | None = None


class SubmitFeedbackRequest(BaseModel):
    message_id: str
    feedback: int                    # 1 = thumbs up, -1 = thumbs down
    comment: str | None = None


class EvaluationResponse(BaseModel):
    message_id: str
    user_feedback: int | None = None
    scores: EvaluationScores | None = None
    judge_reasoning: str | None = None
    evaluation_type: str = "auto"
    created_at: str = ""
    updated_at: str = ""


class EvaluationStatsResponse(BaseModel):
    period_days: int = 7
    total_evaluations: int = 0
    user_feedback_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    avg_relevance: float = 0.0
    avg_accuracy: float = 0.0
    avg_helpfulness: float = 0.0
    avg_safety: float = 0.0
    avg_overall: float = 0.0
    daily_breakdown: list[dict] = []
    score_distribution: list[dict] = []


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