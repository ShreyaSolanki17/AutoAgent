-- Migration: Enforce at most one 'production' and one 'staging' row per
-- prompt_key at the DB level.
--
-- promote_prompt() already archives the previous holder of a status before
-- promoting, but that's an app-level convention, not a guarantee — two
-- concurrent promote_prompt() calls for the same key can both pass the
-- "archive old, then set new" steps before either commits, leaving two
-- rows with the same status. These partial unique indexes make that
-- impossible: the second concurrent UPDATE fails with a unique violation
-- instead of silently succeeding.
--
-- Run with:
--   docker exec -i <db-container> psql -U autoagent -d autoagent_db < docker/migrate_07_prompt_status_uniqueness.sql

CREATE UNIQUE INDEX IF NOT EXISTS prompt_versions_one_production_per_key
    ON prompt_versions (prompt_key)
    WHERE status = 'production';

CREATE UNIQUE INDEX IF NOT EXISTS prompt_versions_one_staging_per_key
    ON prompt_versions (prompt_key)
    WHERE status = 'staging';
