-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing agent memories
CREATE TABLE IF NOT EXISTS memories (
    id          SERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'default',
    content     TEXT NOT NULL,
    embedding   vector(384),   -- all-MiniLM-L6-v2 produces 384-dim vectors
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for filtering by user
CREATE INDEX IF NOT EXISTS memories_user_idx
    ON memories (user_id);
