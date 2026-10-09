CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS openalex_works (
    work_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    publication_year INTEGER,
    cited_by_count INTEGER DEFAULT 0,
    source_id TEXT,
    source_name TEXT,
    source_type TEXT,
    doi TEXT,
    landing_page_url TEXT,
    topics JSONB DEFAULT '[]'::jsonb,
    raw JSONB NOT NULL,
    collected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS openalex_authors (
    author_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    works_count INTEGER DEFAULT 0,
    cited_by_count INTEGER DEFAULT 0,
    affiliation TEXT,
    country_code TEXT,
    raw JSONB DEFAULT '{}'::jsonb,
    collected_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS openalex_authorships (
    work_id TEXT REFERENCES openalex_works(work_id) ON DELETE CASCADE,
    author_id TEXT REFERENCES openalex_authors(author_id) ON DELETE CASCADE,
    author_position INTEGER DEFAULT 0,
    raw_affiliation TEXT,
    PRIMARY KEY (work_id, author_id)
);

CREATE TABLE IF NOT EXISTS work_embeddings (
    work_id TEXT REFERENCES openalex_works(work_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    text_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (work_id, model_name)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight DOUBLE PRECISION DEFAULT 1.0,
    evidence JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_openalex_works_year ON openalex_works(publication_year);
CREATE INDEX IF NOT EXISTS idx_openalex_works_topics ON openalex_works USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_openalex_authors_country ON openalex_authors(country_code);
CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_work_embeddings_model ON work_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_work_embeddings_vector
    ON work_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
