"""FastAPI entry point — start with: uvicorn backend.main:app --reload
"""

import atexit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.logging_config import setup_logging, get_logger
from backend.api.routes import router
from backend.memory.store import start_auto_cleanup, stop_auto_cleanup
from backend.middleware.safety import SafetyMiddleware

# Initialize structured logging before anything else
setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="AutoAgent API")

# Allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Safety middleware — rate limiting, input sanitization, output filtering
app.add_middleware(SafetyMiddleware)

app.include_router(router, prefix="/api")

# Start memory auto-cleanup background scheduler (hourly purge of expired memories)
start_auto_cleanup()
atexit.register(stop_auto_cleanup)

logger.info("app started: AutoAgent API (auto_cleanup=True)")


@app.get("/")
async def root():
    return {"message": "AutoAgent API is running"}


@app.get("/api/metrics")
async def metrics():
    """Basic health + metrics endpoint for monitoring."""
    import time
    return {
        "status": "ok",
        "uptime": time.time(),
        "version": "0.1.0",
    }
