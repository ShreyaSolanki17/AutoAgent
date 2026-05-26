"""Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    final_answer: str
    trace: list[str]


class MemoryItem(BaseModel):
    id: int
    content: str
    created_at: str


class MemoryListResponse(BaseModel):
    memories: list[MemoryItem]
    total: int