"""FastAPI route handlers — chat, memories, health.
"""

from fastapi import APIRouter

from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    MemoryItem,
    MemoryListResponse,
)
from backend.agent.graph import run_agent
from backend.memory.store import get_all_memories, delete_memory

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    result = run_agent(request.message)

    trace = [msg["content"] for msg in result.get("messages", [])]

    return ChatResponse(
        final_answer=result.get("final_answer", ""),
        trace=trace,
    )


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories():
    """List all stored memories."""
    raw = get_all_memories()
    memories = [
        MemoryItem(id=m["id"], content=m["content"], created_at=m["created_at"])
        for m in raw
    ]
    return MemoryListResponse(memories=memories, total=len(memories))


@router.delete("/memories/{memory_id}")
async def remove_memory(memory_id: int):
    """Delete a memory by ID."""
    deleted = delete_memory(memory_id)
    if not deleted:
        return {"error": f"Memory {memory_id} not found"}
    return {"status": "deleted", "id": memory_id}


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
