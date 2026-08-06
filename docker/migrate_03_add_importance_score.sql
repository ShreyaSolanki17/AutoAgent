-- Migration: Add importance_score and last_accessed columns to memories table
-- Enables importance-based ranking and LRU-style pruning.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS importance_score REAL DEFAULT 0.5;

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP DEFAULT NOW();

-- Index for fast pruning queries (sort by importance + age)
CREATE INDEX IF NOT EXISTS memories_importance_idx
    ON memories (user_id, importance_score DESC, created_at DESC);

-- Set initial importance based on content heuristics
-- Personal-identity facts (name, preferences) start higher
UPDATE memories SET importance_score = 0.9
WHERE content ILIKE 'my name is%'
   OR content ILIKE 'i am%'
   OR content ILIKE 'i''m%';

UPDATE memories SET importance_score = 0.8
WHERE content ILIKE 'i like%'
   OR content ILIKE 'i love%'
   OR content ILIKE 'i prefer%'
   OR content ILIKE 'my favorite%'
   OR content ILIKE 'remember that%';

UPDATE memories SET importance_score = 0.7
WHERE content ILIKE 'i work%'
   OR content ILIKE 'i live%'
   OR content ILIKE 'i have%';
