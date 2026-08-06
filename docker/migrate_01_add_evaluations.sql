-- Migration: Create evaluations table for response quality scoring
-- Stores both user feedback (thumbs up/down) and automated LLM-as-judge scores
--
-- Run with:
--   docker exec -i <db-container> psql -U autoagent -d autoagent_db < docker/migrate_01_add_evaluations.sql

CREATE TABLE IF NOT EXISTS evaluations (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'anonymous',
    message_id      TEXT NOT NULL UNIQUE,           -- unique correlation ID per response
    user_goal       TEXT NOT NULL DEFAULT '',       -- the original user message
    final_answer    TEXT NOT NULL DEFAULT '',       -- the agent's response
    trace_hash      TEXT,                           -- hash of trace for correlation
    -- User feedback
    user_feedback   SMALLINT,                       -- 1 = thumbs up, -1 = thumbs down, NULL = no feedback
    user_comment    TEXT,                           -- optional free-text comment
    -- LLM-as-judge scores (each 0.0 - 1.0)
    judge_relevance     REAL DEFAULT NULL,
    judge_accuracy      REAL DEFAULT NULL,
    judge_helpfulness   REAL DEFAULT NULL,
    judge_safety        REAL DEFAULT NULL,
    judge_overall       REAL DEFAULT NULL,
    judge_reasoning     TEXT,
    judge_model         TEXT,
    -- Metadata
    evaluation_type TEXT NOT NULL DEFAULT 'auto',   -- 'auto' | 'user' | 'both'
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- message_id has a UNIQUE constraint above for upsert support
-- (auto-evaluate first, user feedback later via ON CONFLICT).

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS evaluations_user_idx
    ON evaluations (user_id);

CREATE INDEX IF NOT EXISTS evaluations_created_idx
    ON evaluations (created_at);

CREATE INDEX IF NOT EXISTS evaluations_user_created_idx
    ON evaluations (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS evaluations_overall_idx
    ON evaluations (judge_overall DESC)
    WHERE judge_overall IS NOT NULL;

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_evaluations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS evaluations_updated_at_trigger ON evaluations;
CREATE TRIGGER evaluations_updated_at_trigger
    BEFORE UPDATE ON evaluations
    FOR EACH ROW EXECUTE FUNCTION update_evaluations_updated_at();
