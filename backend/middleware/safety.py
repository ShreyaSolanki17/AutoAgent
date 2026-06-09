"""Safety middleware — input sanitization, output filtering, and rate limiting.

Applied to every request before it reaches the route handlers.
"""

import re
import time
import threading
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.logging_config import get_logger

logger = get_logger(__name__)

# ── Input Sanitization ─────────────────────────────────────

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"forget\s+(everything|all|what)",
    # System prompt extraction
    r"(reveal|show|print|output|display|repeat|echo|leak)\s+(your|the|my)\s+(system\s+)?(prompt|instructions?|rules?|config)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions?|rules?)",
    r"read\s+(out|out\s+loud|back)\s+(your|the)\s+(system\s+)?prompt",
    # Role hijacking
    r"you\s+are\s+now\s+(a|an|the)",
    r"act\s+as\s+(a|an|the)\s+(?!.*(?:calculator|wikipedia|search))",  # allow tool names
    r"pretend\s+(you\s+are|to\s+be)",
    r"new\s+persona",
    # Delimiter bypass
    r"<\s*/\s*system\s*>",
    r"<\s*!\s*--\s*#\s*include",
    r"```\s*system",
    r"---\s*system",
]

# Maximum allowed input lengths
_MAX_MESSAGE_LENGTH = 40096   # ~4K chars for chat messages
_MAX_HISTORY_LENGTH = 100     # max history entries in a single request

_injection_regex = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_input(text: str, max_length: int = _MAX_MESSAGE_LENGTH) -> tuple[str, list[str]]:
    """Sanitize user input text.

    Returns:
        Tuple of (sanitized_text, list_of_issues_found).
        If issues include 'blocked', the request should be rejected.
    """
    issues: list[str] = []

    # 1. Length check
    if len(text) > max_length:
        issues.append(f"too_long: {len(text)} chars (max {max_length})")
        text = text[:max_length]

    # 2. Prompt injection detection
    if _injection_regex.search(text):
        issues.append("blocked: prompt_injection_detected")

    # 3. Null byte removal
    if "\x00" in text:
        text = text.replace("\x00", "")
        issues.append("null_bytes_removed")

    # 4. Control character stripping (keep \n \r \t)
    cleaned = ""
    for ch in text:
        if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127):
            cleaned += ch
    if len(cleaned) != len(text):
        issues.append("control_chars_stripped")
    text = cleaned

    # 5. Excessive repetition detection (e.g., "aaaaaaaaaa" x 100)
    if re.search(r"(.)\1{50,}", text):
        issues.append("excessive_repetition_detected")
        text = re.sub(r"(.)\1{50,}", r"\1" * 10, text)

    return text, issues


# ── Output Filtering ───────────────────────────────────────

# Patterns that indicate leaked internal info
_LEAK_PATTERNS = [
    r"PLANNER_SYSTEM",
    r"OBSERVER_SYSTEM",
    r"my\s+system\s+prompt\s+(is|says|contains)",
    r"here\s+is\s+my\s+(system\s+)?prompt",
    r"I\s+(was\s+)?told\s+to\s+(always|never|only)",
    r"my\s+instructions?\s+(say|are|is|tell\s+me)",
    r"GROQ_API_KEY",
    r"GOOGLE_API_KEY",
    r"OPENROUTER_API_KEY",
    r"POSTGRES_PASSWORD",
    r"JWT_SECRET",
    r"SECRET_KEY",
    r"Bearer\s+eyJ",  # JWT token leak
    r"sk-[a-zA-Z0-9]{20,}",  # API key pattern
]

_leak_regex = re.compile("|".join(_LEAK_PATTERNS), re.IGNORECASE)

# Placeholder replacement message
_REDACTED = "[REDACTED — internal content filtered for safety]"


def filter_output(text: str) -> tuple[str, bool]:
    """Filter LLM output to prevent internal info leaks.

    Returns:
        Tuple of (filtered_text, was_modified).
    """
    if _leak_regex.search(text):
        # Redact the entire response — it likely contains system prompt
        logger.warning("safety.output_filtered: potential_system_leak")
        return _REDACTED, True
    return text, False


# ── Rate Limiting ──────────────────────────────────────────

class RateLimiter:
    """In-memory sliding window rate limiter.

    Thread-safe. Stores request timestamps per key (user_id or IP).
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _cleanup(self, key: str, now: float):
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def is_allowed(self, key: str) -> tuple[bool, dict]:
        """Check whether a request from `key` is allowed.

        Returns:
            Tuple of (allowed, info_dict) where info_dict has
            'remaining', 'reset_after', 'limit' keys.
        """
        now = time.time()
        with self._lock:
            self._cleanup(key, now)
            count = len(self._requests[key])

            if count >= self.max_requests:
                oldest = self._requests[key][0]
                reset_after = round(oldest + self.window_seconds - now, 1)
                return False, {
                    "remaining": 0,
                    "reset_after": max(reset_after, 1),
                    "limit": self.max_requests,
                }

            self._requests[key].append(now)
            return True, {
                "remaining": self.max_requests - count - 1,
                "reset_after": self.window_seconds,
                "limit": self.max_requests,
            }

    def cleanup_stale(self, max_keys: int = 10000):
        """Remove entries with no recent requests to bound memory."""
        now = time.time()
        with self._lock:
            stale_keys = [
                k for k, ts in self._requests.items()
                if not ts or ts[-1] < now - self.window_seconds
            ]
            for k in stale_keys[:max_keys]:
                del self._requests[k]


# Global rate limiter instance: 30 requests per 60 seconds
_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


def get_rate_limiter() -> RateLimiter:
    """Return the global rate limiter instance."""
    return _rate_limiter


# ── Middleware ──────────────────────────────────────────────

class SafetyMiddleware:
    """ASGI middleware that applies input sanitization, output filtering,
    and rate limiting to every request.

    Implemented as pure ASGI to avoid BaseHTTPMiddleware quirks with
    early-return responses and header mutation.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # ── Rate limiting ──
        client_key = self._get_client_key(scope)
        allowed, rate_info = _rate_limiter.is_allowed(client_key)

        if not allowed:
            logger.warning(
                "safety.rate_limited client=%s path=%s",
                client_key,
                scope.get("path", ""),
            )
            from starlette.responses import JSONResponse
            retry_after = int(rate_info["reset_after"])
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": retry_after,
                    "limit": rate_info["limit"],
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        # ── Process request + attach rate limit headers ──
        # We need to intercept the response to add headers
        response_started = False
        extra_headers = [
            (b"x-ratelimit-limit", str(rate_info["limit"]).encode()),
            (b"x-ratelimit-remaining", str(rate_info["remaining"]).encode()),
        ]

        async def send_with_headers(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = list(message.get("headers", []))
                headers.extend(extra_headers)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    def _get_client_key(scope: dict) -> str:
        """Extract a rate-limit key from the ASGI scope.

        Priority: JWT user_id → X-Forwarded-For → client host.
        """
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("utf-8", errors="ignore")
        if auth.lower().startswith("bearer "):
            try:
                from backend.auth import _decode_token
                payload = _decode_token(auth[7:])
                if payload and "sub" in payload:
                    return f"user:{payload['sub']}"
            except Exception:
                pass

        # Fall back to IP
        forwarded = headers.get(b"x-forwarded-for", b"").decode("utf-8", errors="ignore")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"
        return "ip:unknown"
