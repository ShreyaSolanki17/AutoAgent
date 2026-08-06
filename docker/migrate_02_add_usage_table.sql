-- Migration: Create usage_logs table for analytics and cost tracking

CREATE TABLE IF NOT EXISTS usage_logs (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'anonymous',
    endpoint        TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0.0,
    response_time_ms INTEGER DEFAULT 0,
    status_code     INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS usage_logs_user_idx
    ON usage_logs (user_id);

CREATE INDEX IF NOT EXISTS usage_logs_created_idx
    ON usage_logs (created_at);

CREATE INDEX IF NOT EXISTS usage_logs_user_created_idx
    ON usage_logs (user_id, created_at);
