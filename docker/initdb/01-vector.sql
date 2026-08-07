-- pgvector extension, required by the document_chunks.embedding vector(768) column.
-- Runs on first initialization of the PostgreSQL data dir (docker-entrypoint-initdb.d).
CREATE EXTENSION IF NOT EXISTS vector;
