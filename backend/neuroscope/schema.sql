-- NeuroScope v3 Database Schema

CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY,
    task TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'gemma-2-2b-it',
    n_steps INTEGER NOT NULL,
    sae_layer INTEGER NOT NULL DEFAULT 12,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    total_elapsed_ms INTEGER,
    status TEXT DEFAULT 'running',
    correct BOOLEAN,
    progress JSONB,
    error TEXT,
    feature_timelines JSONB,
    patch_matrix JSONB,
    patch_matrix_summary JSONB
);

CREATE TABLE IF NOT EXISTS steps (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
    step_n INTEGER NOT NULL,
    prompt TEXT,
    output TEXT,
    tool_called TEXT,
    n_active_features INTEGER,
    sae_l2_norm FLOAT,
    hallucination_score FLOAT,
    entropy FLOAT,
    attn_diffusion FLOAT,
    drift_score FLOAT,
    elapsed_ms INTEGER,
    activation_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    layer_l2_norms JSONB,
    hallucination JSONB
);

CREATE TABLE IF NOT EXISTS step_features (
    id BIGSERIAL PRIMARY KEY,
    step_id UUID REFERENCES steps(id) ON DELETE CASCADE,
    feature_id INTEGER NOT NULL,
    activation FLOAT NOT NULL,
    drift_score FLOAT
);

CREATE TABLE IF NOT EXISTS patch_results (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
    source_step INTEGER NOT NULL,
    target_step INTEGER NOT NULL,
    patch_layer INTEGER NOT NULL,
    kl FLOAT NOT NULL,
    significant BOOLEAN NOT NULL,
    top_token_change JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS queries (
    id UUID PRIMARY KEY,
    run_id TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiments (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    task TEXT NOT NULL,
    n_steps INTEGER NOT NULL,
    sae_layer INTEGER NOT NULL,
    model TEXT NOT NULL,
    finding_seed TEXT,
    finding TEXT,
    total_elapsed_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB
);

CREATE TABLE IF NOT EXISTS feature_labels (
    id TEXT PRIMARY KEY,
    layer INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    neuronpedia_url TEXT
);

CREATE TABLE IF NOT EXISTS attribution_graphs (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
    step_n INTEGER NOT NULL,
    layer INTEGER NOT NULL,
    graph JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS findings_runs (
    id SERIAL PRIMARY KEY,
    question_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    final_correct BOOLEAN NOT NULL,
    steps JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


