"""Usage analytics and cost tracking middleware.

Tracks tokens, cost, and response time per request.
Data is persisted to the usage_logs PostgreSQL table.
"""

import os
import time
import threading
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

from backend.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Provider pricing (USD per 1M tokens) ───────────────────
# Source: provider pricing pages (free-tier where applicable)
_PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "groq": {
        "llama-3.3-70b-versatile":    {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant":        {"input": 0.05, "output": 0.08},
        "mixtral-8x7b-32768":         {"input": 0.24, "output": 0.24},
        "default":                     {"input": 0.59, "output": 0.79},
    },
    "google": {
        "gemini-2.0-flash":           {"input": 0.10, "output": 0.40},
        "gemini-1.5-flash":           {"input": 0.075, "output": 0.30},
        "default":                     {"input": 0.10, "output": 0.40},
    },
    "openrouter": {
        "openai/gpt-oss-20b:free":    {"input": 0.0,  "output": 0.0},
        "default":                     {"input": 0.0, "output": 0.0},
    },
}


def get_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call in USD.

    Args:
        provider: LLM provider name (groq, google, openrouter).
        model: Model identifier string.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Cost in USD.
    """
    provider_pricing = _PROVIDER_PRICING.get(provider, {})
    model_pricing = provider_pricing.get(model, provider_pricing.get("default", {"input": 0.0, "output": 0.0}))

    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

    return round(input_cost + output_cost, 8)


# ── In-memory cache for real-time stats ────────────────────
# These are flushed to the DB periodically and on query
_stats_lock = threading.Lock()
_daily_stats: dict[str, dict] = defaultdict(lambda: {
    "total_requests": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "total_response_time_ms": 0,
})


def record_request(
    user_id: str,
    endpoint: str,
    provider: str | None,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    response_time_ms: int,
    status_code: int,
):
    """Record a request in the daily stats cache and persist to DB."""
    today = time.strftime("%Y-%m-%d")

    with _stats_lock:
        _stats = _daily_stats[today]
        _stats["total_requests"] += 1
        _stats["total_input_tokens"] += input_tokens
        _stats["total_output_tokens"] += output_tokens
        _stats["total_cost_usd"] += cost_usd
        _stats["total_response_time_ms"] += response_time_ms

    # Persist to database
    try:
        from backend.memory.store import _get_connection
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO usage_logs
                        (user_id, endpoint, provider, model, input_tokens,
                         output_tokens, total_tokens, cost_usd, response_time_ms, status_code)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id, endpoint, provider or "", model or "",
                        input_tokens, output_tokens,
                        input_tokens + output_tokens,
                        cost_usd, response_time_ms, status_code,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # Don't break the request if logging fails


def get_usage_stats(user_id: str | None = None, days: int = 7) -> dict:
    """Get aggregated usage statistics.

    Args:
        user_id: If provided, filter to this user. Otherwise global stats.
        days: Number of past days to aggregate.

    Returns:
        Dict with request counts, tokens, cost, avg latency, per-day breakdown.
    """
    try:
        from backend.memory.store import _get_connection
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                # Overall stats
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total_requests,
                            COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                            COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                            COALESCE(SUM(total_tokens), 0) AS total_tokens,
                            COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                            COALESCE(AVG(response_time_ms), 0) AS avg_response_time_ms,
                            MIN(created_at) AS first_request,
                            MAX(created_at) AS last_request
                        FROM usage_logs
                        WHERE user_id = %s
                          AND created_at >= NOW() - INTERVAL '%s days'
                        """,
                        (user_id, days),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total_requests,
                            COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                            COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                            COALESCE(SUM(total_tokens), 0) AS total_tokens,
                            COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                            COALESCE(AVG(response_time_ms), 0) AS avg_response_time_ms,
                            MIN(created_at) AS first_request,
                            MAX(created_at) AS last_request
                        FROM usage_logs
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                        """,
                        (days,),
                    )
                row = cur.fetchone()

                # Per-day breakdown
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            DATE(created_at) AS day,
                            COUNT(*) AS requests,
                            COALESCE(SUM(total_tokens), 0) AS tokens,
                            COALESCE(SUM(cost_usd), 0.0) AS cost,
                            COALESCE(AVG(response_time_ms), 0) AS avg_latency_ms
                        FROM usage_logs
                        WHERE user_id = %s
                          AND created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY DATE(created_at)
                        ORDER BY day DESC
                        """,
                        (user_id, days),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            DATE(created_at) AS day,
                            COUNT(*) AS requests,
                            COALESCE(SUM(total_tokens), 0) AS tokens,
                            COALESCE(SUM(cost_usd), 0.0) AS cost,
                            COALESCE(AVG(response_time_ms), 0) AS avg_latency_ms
                        FROM usage_logs
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY DATE(created_at)
                        ORDER BY day DESC
                        """,
                        (days,),
                    )
                daily_rows = cur.fetchall()

                # Per-provider breakdown
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            provider,
                            model,
                            COUNT(*) AS requests,
                            COALESCE(SUM(cost_usd), 0.0) AS cost
                        FROM usage_logs
                        WHERE user_id = %s
                          AND created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY provider, model
                        ORDER BY requests DESC
                        """,
                        (user_id, days),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            provider,
                            model,
                            COUNT(*) AS requests,
                            COALESCE(SUM(cost_usd), 0.0) AS cost
                        FROM usage_logs
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY provider, model
                        ORDER BY requests DESC
                        """,
                        (days,),
                    )
                provider_rows = cur.fetchall()

            return {
                "period_days": days,
                "total_requests": row[0] or 0,
                "total_input_tokens": int(row[1] or 0),
                "total_output_tokens": int(row[2] or 0),
                "total_tokens": int(row[3] or 0),
                "total_cost_usd": round(float(row[4] or 0), 6),
                "avg_response_time_ms": round(float(row[5] or 0), 1),
                "first_request": str(row[6]) if row[6] else None,
                "last_request": str(row[7]) if row[7] else None,
                "daily_breakdown": [
                    {
                        "day": str(r[0]),
                        "requests": r[1],
                        "tokens": int(r[2]),
                        "cost_usd": round(float(r[3]), 6),
                        "avg_latency_ms": round(float(r[4]), 1),
                    }
                    for r in daily_rows
                ],
                "provider_breakdown": [
                    {
                        "provider": r[0],
                        "model": r[1],
                        "requests": r[2],
                        "cost_usd": round(float(r[3]), 6),
                    }
                    for r in provider_rows
                ],
            }
        finally:
            conn.close()
    except Exception:
        # Fallback to in-memory stats if DB is unavailable
        return {
            "period_days": days,
            "note": "in-memory only (db unavailable)",
            "cached_today": dict(_daily_stats.get(time.strftime("%Y-%m-%d"), {})),
        }
