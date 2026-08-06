-- Migration: Add updated_at, expires_at columns to existing memories table
-- Run this if you already have a deployed memories table without these columns

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '90 days');
