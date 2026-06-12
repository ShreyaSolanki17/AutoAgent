"""Database operations for evaluations.

Follows the same pattern as backend.memory.store and backend.middleware.usage:
- Direct psycopg2 connections via _get_connection()
- Graceful failure (never breaks the main request)
- Upsert pattern: auto-evaluate first, user feedback updates the same row
"""

from backend.memory.store import _get_connection
from backend.logging_config import get_logger

logger = get_logger(__name__)


# ── Save / Upsert ──────────────────────────────────────────

def save_evaluation(
    user_id: str,
    message_id: str,
    user_goal: str,
    final_answer: str,
    trace_hash: str | None = None,
    judge_scores: dict | None = None,
    user_feedback: int | None = None,
    user_comment: str | None = None,
    ab_variant: str | None = None,
) -> int | None:
    """Save or upsert an evaluation record.

    Upsert strategy:
    - First call (auto-evaluate): inserts with judge scores
    - Second call (user feedback): updates same row with user_feedback
    - The DB unique constraint on message_id prevents duplicates

    Args:
        user_id: User identifier.
        message_id: Correlation ID from ChatResponse.
        user_goal: Original user message.
        final_answer: Agent's response.
        trace_hash: Optional trace hash for correlation.
        judge_scores: Dict from evaluate_response() with relevance/accuracy/etc.
        user_feedback: 1 (thumbs up), -1 (thumbs down), or None.
        user_comment: Optional free-text comment.

    Returns:
        The evaluation row ID, or None on failure.
    """
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                # Determine evaluation_type
                eval_type = "auto"
                if user_feedback is not None and judge_scores:
                    eval_type = "both"
                elif user_feedback is not None:
                    eval_type = "user"

                cur.execute(
                    """
                    INSERT INTO evaluations (
                        user_id, message_id, user_goal, final_answer, trace_hash,
                        user_feedback, user_comment,
                        judge_relevance, judge_accuracy, judge_helpfulness,
                        judge_safety, judge_overall, judge_reasoning, judge_model,
                        evaluation_type, ab_variant
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s
                    )
                    ON CONFLICT (message_id) DO UPDATE SET
                        user_feedback = COALESCE(EXCLUDED.user_feedback, evaluations.user_feedback),
                        user_comment = COALESCE(EXCLUDED.user_comment, evaluations.user_comment),
                        judge_relevance = COALESCE(EXCLUDED.judge_relevance, evaluations.judge_relevance),
                        judge_accuracy = COALESCE(EXCLUDED.judge_accuracy, evaluations.judge_accuracy),
                        judge_helpfulness = COALESCE(EXCLUDED.judge_helpfulness, evaluations.judge_helpfulness),
                        judge_safety = COALESCE(EXCLUDED.judge_safety, evaluations.judge_safety),
                        judge_overall = COALESCE(EXCLUDED.judge_overall, evaluations.judge_overall),
                        judge_reasoning = COALESCE(EXCLUDED.judge_reasoning, evaluations.judge_reasoning),
                        judge_model = COALESCE(EXCLUDED.judge_model, evaluations.judge_model),
                        ab_variant = COALESCE(EXCLUDED.ab_variant, evaluations.ab_variant),
                        evaluation_type = CASE
                            WHEN evaluations.user_feedback IS NOT NULL
                                 AND EXCLUDED.judge_overall IS NOT NULL THEN 'both'
                            WHEN evaluations.user_feedback IS NOT NULL THEN 'user'
                            ELSE 'auto'
                        END,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        user_id,
                        message_id,
                        user_goal,
                        final_answer,
                        trace_hash,
                        user_feedback,
                        user_comment,
                        judge_scores.get("relevance") if judge_scores else None,
                        judge_scores.get("accuracy") if judge_scores else None,
                        judge_scores.get("helpfulness") if judge_scores else None,
                        judge_scores.get("safety") if judge_scores else None,
                        judge_scores.get("overall") if judge_scores else None,
                        judge_scores.get("reasoning") if judge_scores else None,
                        judge_scores.get("judge_model") if judge_scores else None,
                        eval_type,
                        ab_variant,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("eval.store.save_failed", error=str(e))
        return None


# ── Aggregated Stats ───────────────────────────────────────

def get_evaluation_stats(user_id: str | None = None, days: int = 7) -> dict:
    """Get aggregated evaluation statistics for the dashboard.

    Mirrors the structure of get_usage_stats() in middleware/usage.py.

    Args:
        user_id: If provided, filter to this user. Otherwise global stats.
        days: Number of past days to aggregate.

    Returns:
        Dict with totals, per-dimension averages, daily breakdown, and score distribution.
    """
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                # ── Overall aggregates ──
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total,
                            COUNT(user_feedback) AS feedback_count,
                            COUNT(*) FILTER (WHERE user_feedback = 1) AS thumbs_up,
                            COUNT(*) FILTER (WHERE user_feedback = -1) AS thumbs_down,
                            AVG(judge_relevance) FILTER (WHERE judge_relevance IS NOT NULL) AS avg_relevance,
                            AVG(judge_accuracy) FILTER (WHERE judge_accuracy IS NOT NULL) AS avg_accuracy,
                            AVG(judge_helpfulness) FILTER (WHERE judge_helpfulness IS NOT NULL) AS avg_helpfulness,
                            AVG(judge_safety) FILTER (WHERE judge_safety IS NOT NULL) AS avg_safety,
                            AVG(judge_overall) FILTER (WHERE judge_overall IS NOT NULL) AS avg_overall
                        FROM evaluations
                        WHERE user_id = %s
                          AND created_at >= NOW() - INTERVAL '%s days'
                        """,
                        (user_id, days),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total,
                            COUNT(user_feedback) AS feedback_count,
                            COUNT(*) FILTER (WHERE user_feedback = 1) AS thumbs_up,
                            COUNT(*) FILTER (WHERE user_feedback = -1) AS thumbs_down,
                            AVG(judge_relevance) FILTER (WHERE judge_relevance IS NOT NULL) AS avg_relevance,
                            AVG(judge_accuracy) FILTER (WHERE judge_accuracy IS NOT NULL) AS avg_accuracy,
                            AVG(judge_helpfulness) FILTER (WHERE judge_helpfulness IS NOT NULL) AS avg_helpfulness,
                            AVG(judge_safety) FILTER (WHERE judge_safety IS NOT NULL) AS avg_safety,
                            AVG(judge_overall) FILTER (WHERE judge_overall IS NOT NULL) AS avg_overall
                        FROM evaluations
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                        """,
                        (days,),
                    )
                row = cur.fetchone()

                # ── Per-day breakdown ──
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            DATE(created_at) AS day,
                            COUNT(*) AS total,
                            AVG(judge_overall) FILTER (WHERE judge_overall IS NOT NULL) AS avg_score,
                            COUNT(user_feedback) AS feedback_count
                        FROM evaluations
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
                            COUNT(*) AS total,
                            AVG(judge_overall) FILTER (WHERE judge_overall IS NOT NULL) AS avg_score,
                            COUNT(user_feedback) AS feedback_count
                        FROM evaluations
                        WHERE created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY DATE(created_at)
                        ORDER BY day DESC
                        """,
                        (days,),
                    )
                daily_rows = cur.fetchall()

                # ── Score distribution (histogram buckets) ──
                if user_id:
                    cur.execute(
                        """
                        SELECT
                            CASE
                                WHEN judge_overall >= 0.9 THEN '0.9-1.0'
                                WHEN judge_overall >= 0.8 THEN '0.8-0.9'
                                WHEN judge_overall >= 0.7 THEN '0.7-0.8'
                                WHEN judge_overall >= 0.6 THEN '0.6-0.7'
                                WHEN judge_overall >= 0.5 THEN '0.5-0.6'
                                WHEN judge_overall >= 0.4 THEN '0.4-0.5'
                                WHEN judge_overall >= 0.3 THEN '0.3-0.4'
                                WHEN judge_overall >= 0.2 THEN '0.2-0.3'
                                WHEN judge_overall >= 0.1 THEN '0.1-0.2'
                                ELSE '0.0-0.1'
                            END AS score_range,
                            COUNT(*) AS count
                        FROM evaluations
                        WHERE user_id = %s
                          AND judge_overall IS NOT NULL
                          AND created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY 1
                        ORDER BY 1 DESC
                        """,
                        (user_id, days),
                    )
                else:
                    cur.execute(
                        """
                        SELECT
                            CASE
                                WHEN judge_overall >= 0.9 THEN '0.9-1.0'
                                WHEN judge_overall >= 0.8 THEN '0.8-0.9'
                                WHEN judge_overall >= 0.7 THEN '0.7-0.8'
                                WHEN judge_overall >= 0.6 THEN '0.6-0.7'
                                WHEN judge_overall >= 0.5 THEN '0.5-0.6'
                                WHEN judge_overall >= 0.4 THEN '0.4-0.5'
                                WHEN judge_overall >= 0.3 THEN '0.3-0.4'
                                WHEN judge_overall >= 0.2 THEN '0.2-0.3'
                                WHEN judge_overall >= 0.1 THEN '0.1-0.2'
                                ELSE '0.0-0.1'
                            END AS score_range,
                            COUNT(*) AS count
                        FROM evaluations
                        WHERE judge_overall IS NOT NULL
                          AND created_at >= NOW() - INTERVAL '%s days'
                        GROUP BY 1
                        ORDER BY 1 DESC
                        """,
                        (days,),
                    )
                dist_rows = cur.fetchall()

            return {
                "period_days": days,
                "total_evaluations": row[0] or 0,
                "user_feedback_count": row[1] or 0,
                "thumbs_up_count": row[2] or 0,
                "thumbs_down_count": row[3] or 0,
                "avg_relevance": round(float(row[4] or 0), 3),
                "avg_accuracy": round(float(row[5] or 0), 3),
                "avg_helpfulness": round(float(row[6] or 0), 3),
                "avg_safety": round(float(row[7] or 0), 3),
                "avg_overall": round(float(row[8] or 0), 3),
                "daily_breakdown": [
                    {
                        "day": str(r[0]),
                        "total": r[1],
                        "avg_score": round(float(r[2] or 0), 3),
                        "feedback_count": r[3],
                    }
                    for r in daily_rows
                ],
                "score_distribution": [
                    {"score_range": r[0], "count": r[1]}
                    for r in dist_rows
                ],
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error("eval.store.stats_failed", error=str(e))
        return {"period_days": days, "error": str(e)}


# ── Recent Evaluations ─────────────────────────────────────

def get_recent_evaluations(user_id: str, limit: int = 20) -> list[dict]:
    """Get recent evaluations for a user (for feedback loop analysis).

    Args:
        user_id: User to filter.
        limit: Maximum number of rows.

    Returns:
        List of evaluation dicts.
    """
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT message_id, user_goal, final_answer,
                           user_feedback, user_comment,
                           judge_relevance, judge_accuracy, judge_helpfulness,
                           judge_safety, judge_overall, judge_reasoning,
                           evaluation_type, created_at
                    FROM evaluations
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()

            return [
                {
                    "message_id": r[0],
                    "user_goal": r[1],
                    "final_answer": r[2],
                    "user_feedback": r[3],
                    "user_comment": r[4],
                    "scores": {
                        "relevance": r[5],
                        "accuracy": r[6],
                        "helpfulness": r[7],
                        "safety": r[8],
                        "overall": r[9],
                    },
                    "judge_reasoning": r[10],
                    "evaluation_type": r[11],
                    "created_at": str(r[12]),
                }
                for r in rows
            ]
        finally:
            conn.close()
    except Exception as e:
        logger.error("eval.store.recent_failed", error=str(e))
        return []


# ── A/B Comparison ──────────────────────────────────────────

def get_ab_comparison_stats(
    test_name: str,
    days: int = 30,
) -> dict:
    """Compare judge scores and usage metrics between A and B variants.

    Queries the evaluations table filtered by ab_variant and joins with
    usage_logs on message_id for cost/latency comparison.

    Args:
        test_name: The A/B test name (used to filter via ab_variant presence).
        days: Number of past days to aggregate.

    Returns:
        Dict with per-variant stats and difference.
    """
    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                # ── Per-variant judge score aggregates ──
                cur.execute(
                    """
                    SELECT
                        ab_variant,
                        COUNT(*) AS total,
                        AVG(judge_relevance) FILTER (WHERE judge_relevance IS NOT NULL) AS avg_relevance,
                        AVG(judge_accuracy) FILTER (WHERE judge_accuracy IS NOT NULL) AS avg_accuracy,
                        AVG(judge_helpfulness) FILTER (WHERE judge_helpfulness IS NOT NULL) AS avg_helpfulness,
                        AVG(judge_safety) FILTER (WHERE judge_safety IS NOT NULL) AS avg_safety,
                        AVG(judge_overall) FILTER (WHERE judge_overall IS NOT NULL) AS avg_overall,
                        COUNT(*) FILTER (WHERE user_feedback = 1) AS thumbs_up,
                        COUNT(*) FILTER (WHERE user_feedback = -1) AS thumbs_down
                    FROM evaluations
                    WHERE ab_variant IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY ab_variant
                    ORDER BY ab_variant
                    """,
                    (days,),
                )
                variant_rows = cur.fetchall()

                # ── Per-variant usage metrics (joined from usage_logs) ──
                cur.execute(
                    """
                    SELECT
                        u.ab_variant,
                        COUNT(*) AS total,
                        AVG(u.cost_usd) AS avg_cost,
                        AVG(u.response_time_ms) AS avg_latency_ms
                    FROM usage_logs u
                    WHERE u.ab_variant IS NOT NULL
                      AND u.created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY u.ab_variant
                    ORDER BY u.ab_variant
                    """,
                    (days,),
                )
                usage_rows = cur.fetchall()

            # Build per-variant dicts
            def _build_variant(row, usage_map):
                variant = row[0]
                usage = usage_map.get(variant, {})
                return {
                    "variant": variant,
                    "count": row[1] or 0,
                    "avg_relevance": round(float(row[2] or 0), 3),
                    "avg_accuracy": round(float(row[3] or 0), 3),
                    "avg_helpfulness": round(float(row[4] or 0), 3),
                    "avg_safety": round(float(row[5] or 0), 3),
                    "avg_overall": round(float(row[6] or 0), 3),
                    "thumbs_up": row[7] or 0,
                    "thumbs_down": row[8] or 0,
                    "avg_cost_usd": round(float(usage.get("avg_cost") or 0), 6),
                    "avg_response_time_ms": round(float(usage.get("avg_latency_ms") or 0), 1),
                }

            usage_map = {r[0]: {"avg_cost": r[2], "avg_latency_ms": r[3]} for r in usage_rows}

            variants = [_build_variant(row, usage_map) for row in variant_rows]

            # Split into A and B
            var_a = next((v for v in variants if v["variant"] == "A"), None)
            var_b = next((v for v in variants if v["variant"] == "B"), None)

            # Compute difference (B - A) for key metrics
            diff = {}
            if var_a and var_b:
                for metric in ("avg_relevance", "avg_accuracy", "avg_helpfulness",
                               "avg_safety", "avg_overall", "avg_cost_usd",
                               "avg_response_time_ms"):
                    a_val = var_a.get(metric, 0) or 0
                    b_val = var_b.get(metric, 0) or 0
                    diff[metric] = round(b_val - a_val, 6)
                    if a_val != 0:
                        diff[f"{metric}_pct"] = round((b_val - a_val) / a_val * 100, 2)
                    else:
                        diff[f"{metric}_pct"] = 0.0

            return {
                "test_name": test_name,
                "period_days": days,
                "variant_a": var_a,
                "variant_b": var_b,
                "difference_b_minus_a": diff,
            }
        finally:
            conn.close()
    except Exception as e:
        logger.error("eval.store.ab_comparison_failed", error=str(e))
        return {"test_name": test_name, "period_days": days, "error": str(e)}
