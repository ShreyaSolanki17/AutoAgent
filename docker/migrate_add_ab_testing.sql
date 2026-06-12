-- Migration: Add A/B testing tables and columns
-- Stores A/B test definitions and tags evaluations + usage_logs with variant info
--
-- Run with:
--   docker exec -i <db-container> psql -U autoagent -d autoagent_db < docker/migrate_add_ab_testing.sql

-- ── A/B Tests table ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ab_tests (
    id                SERIAL PRIMARY KEY,
    test_name         TEXT NOT NULL UNIQUE,          -- e.g. 'planner_v2_vs_v3'
    prompt_key        TEXT NOT NULL,                 -- e.g. 'PLANNER_SYSTEM'
    variant_a_version  INTEGER NOT NULL,              -- version number for variant A
    variant_b_version  INTEGER NOT NULL,              -- version number for variant B
    status            TEXT NOT NULL DEFAULT 'active',-- 'active' | 'paused' | 'completed'
    description       TEXT,
    created_by        TEXT DEFAULT 'system',
    created_at        TIMESTAMP DEFAULT NOW(),
    ended_at          TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ab_tests_prompt_key_idx ON ab_tests (prompt_key);
CREATE INDEX IF NOT EXISTS ab_tests_status_idx ON ab_tests (status);

-- Add ab_variant to evaluations for A/B comparison ──────
ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS ab_variant TEXT DEFAULT NULL;  -- 'A' or 'B' or NULL

CREATE INDEX IF NOT EXISTS evaluations_ab_variant_idx
    ON evaluations (ab_variant)
    WHERE ab_variant IS NOT NULL;

── Add ab_variant to usage_logs for cost/latency comparison ──
ALTER TABLE usage_logs
    ADD COLUMN IF NOT EXISTS ab_variant TEXT DEFAULT NULL;

CREATE INDEX IF NOT EXISTS usage_logs_ab_variant_idx
    ON usage_logs (ab_variant)
    WHERE ab_variant IS NOT NULL;
