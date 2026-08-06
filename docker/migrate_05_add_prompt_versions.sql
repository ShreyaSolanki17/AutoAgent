-- Migration: Create prompt_versions table for prompt versioning & promotion
-- Stores versioned prompts with lifecycle: draft → staging → production → archived
--
-- Run with:
--   docker exec -i <db-container> psql -U autoagent -d autoagent_db < docker/migrate_05_add_prompt_versions.sql

CREATE TABLE IF NOT EXISTS prompt_versions (
    id              SERIAL PRIMARY KEY,
    prompt_key      TEXT NOT NULL,          -- e.g. 'PLANNER_SYSTEM', 'OBSERVER_SYSTEM'
    version         INTEGER NOT NULL DEFAULT 1,
    content         TEXT NOT NULL,          -- the actual prompt text
    status          TEXT NOT NULL DEFAULT 'draft',  -- 'draft' | 'staging' | 'production' | 'archived'
    change_summary  TEXT,                   -- human-readable description of the change
    created_by      TEXT DEFAULT 'system',  -- who created this version
    created_at      TIMESTAMP DEFAULT NOW(),
    promoted_at     TIMESTAMP,              -- when it was promoted to production
    UNIQUE(prompt_key, version)
);

-- Index for common query: "get current production prompt for key X"
CREATE INDEX IF NOT EXISTS prompt_versions_key_status_idx
    ON prompt_versions (prompt_key, status);

-- Index for listing version history
CREATE INDEX IF NOT EXISTS prompt_versions_key_version_idx
    ON prompt_versions (prompt_key, version DESC);

-- Trigger to auto-update updated_at on promotion
CREATE OR REPLACE FUNCTION update_prompt_versions_promoted_at()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'production' AND OLD.status != 'production' THEN
        NEW.promoted_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS prompt_versions_promoted_at_trigger ON prompt_versions;
CREATE TRIGGER prompt_versions_promoted_at_trigger
    BEFORE UPDATE ON prompt_versions
    FOR EACH ROW EXECUTE FUNCTION update_prompt_versions_promoted_at();

-- Seed with current hardcoded prompt values as version 1 (production)
-- These mirror the defaults in backend/agent/prompts.py
INSERT INTO prompt_versions (prompt_key, version, content, status, change_summary, created_by)
VALUES (
    'PLANNER_SYSTEM',
    1,
    E'You are a planning module. Analyze the user''s goal and decide the NEXT single action.\n\nTools: calculator, wikipedia, search, file_read, file_write, memory_save, memory_search, finish.\n\nRules:\n- Greetings/casual chat → finish immediately\n- Resolve pronouns (he/his/their/it) from conversation history. Include the resolved subject in the tool input.\n- Never call the same tool twice. If a tool fails, switch or finish.\n- When in doubt, finish with the best answer you have.\n\nFormat:\nTOOL: <tool_name>\nINPUT: <input>\nREASON: <one sentence>\n',
    'production',
    'Initial version — seeded from hardcoded default',
    'system'
)
ON CONFLICT (prompt_key, version) DO NOTHING;

INSERT INTO prompt_versions (prompt_key, version, content, status, change_summary, created_by)
VALUES (
    'OBSERVER_SYSTEM',
    1,
    E'You are an observation module. Evaluate the tool result and decide:\n\n- Goal answered? → DONE: <final answer written in a natural, conversational way>\n- Need more? → CONTINUE: <what''s still needed>\n\nRules:\n- Be very willing to say DONE. Write the final answer as if talking to a friend — natural, concise, no bullet points, no raw tool output, no metadata.\n- NEVER include tool names, memory IDs, JSON, or internal system details in your DONE response.\n- If the same tool was called more than once, ALWAYS say DONE.\n- If iterations >= 3 without a clear answer, say DONE with the best available response.\n- For greetings, reply DONE immediately with just a friendly greeting.\n',
    'production',
    'Initial version — seeded from hardcoded default',
    'system'
)
ON CONFLICT (prompt_key, version) DO NOTHING;
