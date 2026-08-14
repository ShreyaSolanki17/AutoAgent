"""Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel, EmailStr


# ── Chat ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Previous messages: [{"role": "user"|"assistant", "content": "..."}]
    debate: bool = False  # Multi-agent debate mode (needs all 3 provider keys, else falls back)


class ChatResponse(BaseModel):
    final_answer: str
    trace: list[str]
    message_id: str = ""           # Correlation ID for feedback submission
    evaluation: "EvaluationResponse | None" = None  # Inline judge result (if available)
    ab_variant: str | None = None  # 'A' or 'B' if an A/B test is active


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


# ── Prompt Versioning ────────────────────────────────────

class PromptVersionCreate(BaseModel):
    prompt_key: str
    content: str
    change_summary: str = ""


class PromptVersionResponse(BaseModel):
    id: int
    prompt_key: str
    version: int
    content: str
    status: str
    change_summary: str | None
    created_by: str
    created_at: str = ""
    promoted_at: str | None = None


class PromptPromoteRequest(BaseModel):
    target_status: str  # "staging" or "production"


class PromptListResponse(BaseModel):
    prompts: list[PromptVersionResponse]
    total: int


# ── A/B Testing ─────────────────────────────────────────

class ABCreateRequest(BaseModel):
    test_name: str
    prompt_key: str
    variant_a_version: int
    variant_b_version: int
    description: str = ""


class ABTestResponse(BaseModel):
    id: int
    test_name: str
    prompt_key: str
    variant_a_version: int
    variant_b_version: int
    status: str
    description: str | None = None
    created_by: str = ""
    created_at: str = ""
    ended_at: str | None = None


class ABTestListResponse(BaseModel):
    tests: list[ABTestResponse]
    total: int