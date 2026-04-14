-- ─────────────────────────────────────────────────────────────────────────────
-- Iris Knowledge Base — Supabase pgvector schema
-- Embedding: sentence-transformers/nomic-embed-text via Ollama  (768-dim, FREE, local)
-- Requires: ollama pull nomic-embed-text
-- Run this once in your Supabase project → SQL Editor
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable the pgvector extension
create extension if not exists vector;

-- ── Main knowledge chunks table ───────────────────────────────────────────────
create table if not exists iris_knowledge (
    id              bigserial primary key,
    file_name       text        not null,          -- original filename
    file_path       text        not null,          -- full path at index time
    file_type       text        not null,          -- pdf | docx | xlsx | md | txt
    chunk_index     int         not null,          -- position within file (0-based)
    chunk_text      text        not null,          -- the actual text chunk
    heading         text        default '',        -- nearest heading/section title
    page_or_sheet   text        default '',        -- page number or sheet name
    embedding       vector(768),                  -- nomic-embed-text via Ollama (free, local)
    char_count      int         not null default 0,
    indexed_at      timestamptz not null default now(),
    metadata        jsonb       default '{}'
);

-- ── Index for fast similarity search ─────────────────────────────────────────
create index if not exists iris_knowledge_embedding_idx
    on iris_knowledge
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);

-- ── Index for file-based operations (re-indexing a single file) ───────────────
create index if not exists iris_knowledge_file_idx
    on iris_knowledge (file_name);

-- ── Similarity search function ────────────────────────────────────────────────
-- Returns top-k chunks ordered by cosine similarity to query embedding
create or replace function iris_search(
    query_embedding  vector(768),
    match_threshold  float   default 0.30,
    match_count      int     default 5
)
returns table (
    id              bigint,
    file_name       text,
    file_type       text,
    heading         text,
    page_or_sheet   text,
    chunk_text      text,
    similarity      float
)
language sql stable
as $$
    select
        id,
        file_name,
        file_type,
        heading,
        page_or_sheet,
        chunk_text,
        1 - (embedding <=> query_embedding) as similarity
    from iris_knowledge
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;

-- ── Optional: view showing index status per file ──────────────────────────────
create or replace view iris_knowledge_index_status as
select
    file_name,
    file_type,
    count(*)                                as chunks,
    sum(char_count)                         as total_chars,
    max(indexed_at)                         as last_indexed,
    round(avg(char_count)::numeric, 0)      as avg_chunk_chars
from iris_knowledge
group by file_name, file_type
order by last_indexed desc;

-- ── Long-term memory table (Iris cross-session findings) ─────────────────────
create table if not exists iris_memory (
    id           bigserial primary key,
    memory_type  text        not null default 'finding',
    dataset      text        not null default '',
    folder       text        not null default '',
    title        text        not null,
    detail       text        not null,
    tags         text[]      not null default '{}',
    embedding    vector(768),
    created_at   timestamptz not null default now(),
    session_date text        not null default ''
);

create index if not exists iris_memory_embedding_idx
    on iris_memory
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);

create or replace function iris_memory_search(
    query_embedding  vector(768),
    match_threshold  float   default 0.30,
    match_count      int     default 5
)
returns table (
    id           bigint,
    memory_type  text,
    dataset      text,
    title        text,
    detail       text,
    tags         text[],
    similarity   float,
    created_at   timestamptz
)
language sql stable as $$
    select id, memory_type, dataset, title, detail, tags,
           1 - (embedding <=> query_embedding) as similarity,
           created_at
    from iris_memory
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by embedding <=> query_embedding
    limit match_count;
$$;