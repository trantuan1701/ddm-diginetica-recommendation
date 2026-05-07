-- PostgreSQL-friendly baseline schema for the DDM recommendation analytics marts.
-- Grain and wording are session-centered. CTR and revenue fields are offline proxies.

CREATE TABLE IF NOT EXISTS dim_item (
    item_id BIGINT PRIMARY KEY,
    pricelog2 NUMERIC,
    price_proxy NUMERIC,
    product_name_tokens TEXT,
    primary_category_id BIGINT,
    category_count INTEGER,
    category_ids TEXT
);

CREATE TABLE IF NOT EXISTS dim_model (
    model_key TEXT PRIMARY KEY,
    model_family TEXT,
    model_role TEXT,
    top_k INTEGER,
    source_type TEXT,
    warning_text TEXT
);

CREATE TABLE IF NOT EXISTS fact_session_summary (
    session_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    first_event_date TIMESTAMP,
    last_event_date TIMESTAMP,
    view_count INTEGER,
    unique_viewed_items INTEGER,
    first_view_timeframe BIGINT,
    last_view_timeframe BIGINT,
    first_purchase_date TIMESTAMP,
    last_purchase_date TIMESTAMP,
    purchase_count INTEGER,
    unique_purchased_items INTEGER,
    order_count INTEGER,
    purchased_value_proxy NUMERIC,
    has_purchase BOOLEAN
);

CREATE TABLE IF NOT EXISTS fact_purchases (
    purchase_id BIGINT PRIMARY KEY,
    session_id BIGINT,
    user_id BIGINT,
    timeframe BIGINT,
    event_date TIMESTAMP,
    order_number BIGINT,
    item_id BIGINT,
    price_proxy NUMERIC,
    primary_category_id BIGINT
);

CREATE TABLE IF NOT EXISTS fact_recommendations (
    model_key TEXT NOT NULL,
    example_id BIGINT NOT NULL,
    session_id BIGINT,
    rank INTEGER NOT NULL,
    pred_item_id_internal BIGINT,
    pred_item_id_raw BIGINT,
    score NUMERIC,
    PRIMARY KEY (model_key, example_id, rank)
);

CREATE TABLE IF NOT EXISTS fact_test_examples (
    example_id BIGINT PRIMARY KEY,
    session_id BIGINT,
    target_item_id_internal BIGINT,
    target_item_id_raw BIGINT,
    event_date TIMESTAMP,
    target_price_proxy NUMERIC,
    is_purchase_session BOOLEAN
);

CREATE TABLE IF NOT EXISTS fact_recommendation_eval (
    model_key TEXT NOT NULL,
    example_id BIGINT NOT NULL,
    session_id BIGINT,
    target_item_id_raw BIGINT,
    hit_at_k NUMERIC,
    target_rank INTEGER,
    reciprocal_rank NUMERIC,
    target_price_proxy NUMERIC,
    captured_value_proxy NUMERIC,
    is_purchase_session BOOLEAN,
    PRIMARY KEY (model_key, example_id)
);

CREATE TABLE IF NOT EXISTS fact_metrics (
    model_key TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    k INTEGER,
    metric_scope TEXT,
    source TEXT,
    warning_text TEXT
);

CREATE TABLE IF NOT EXISTS fact_marketing_kpis (
    model_key TEXT NOT NULL,
    k INTEGER,
    kpi_name TEXT NOT NULL,
    kpi_value NUMERIC,
    kpi_scope TEXT,
    warning_text TEXT
);
