# SR-GNN Model Evaluation Dashboard (Power BI Implementation Guide)

This guide implements the one-page report **“SR-GNN Model Evaluation Dashboard”** using repo data marts and existing dashboard style conventions.

## 1. Required files

### Core marts (preferred)
- `data/mart/fact_recommendation_eval.parquet`
- `data/mart/fact_metrics.parquet`
- `data/mart/fact_marketing_kpis.parquet`
- `data/mart/dim_model.parquet`
- `data/mart/dim_item.parquet`

### Raw session-level sources
- `data/raw/diginetica/train-item-views.csv`
- `data/raw/diginetica/train-purchases.csv`

### Optional helper outputs (generated)
Run:
```bash
PYTHONPATH=src python scripts/build_srgnn_dashboard_data.py
```
This writes:
- `dashboards/powerbi_data/pbi_session_summary.csv`
- `dashboards/powerbi_data/pbi_model_metrics_summary.csv`
- `dashboards/powerbi_data/pbi_data_quality_summary.csv`

## 2. Schema and data model

### Main tables
1. `dim_model(model_key, model_label, model_type, top_k)`
2. `fact_metrics(model_key, metric_name, metric_value, k, metric_scope)`
3. `fact_recommendation_eval(example_id, session_id, model_key, item_id/target_item_id_raw, hit_at_k, rank/target_rank, reciprocal_rank, captured_value_proxy, session_length_bucket)`
4. `fact_marketing_kpis(model_key, kpi_name, kpi_value, warning_text)`
5. `dim_item(item_id, category_id, price_log2/pricelog2, price_proxy, category_name)`
6. `session_summary(session_id, timeframe, view_count, unique_items_viewed, has_purchase, purchase_count, quantity_sum, session_length_bucket, session_view_bucket_for_conversion, segment)`
7. `purchases` (from `train-purchases.csv`, normalized ids)

### Relationships
- `dim_model[model_key]` -> `fact_metrics[model_key]` (1:*).
- `dim_model[model_key]` -> `fact_recommendation_eval[model_key]` (1:*).
- `dim_model[model_key]` -> `fact_marketing_kpis[model_key]` (1:*).
- `dim_item[item_id]` -> `fact_recommendation_eval[item_id or target_item_id_raw]` (1:*).
- `dim_item[item_id]` -> `purchases[item_id]` (1:*).
- `session_summary[session_id]` -> `fact_recommendation_eval[session_id]` (1:*).
- `session_summary[session_id]` -> `purchases[session_id]` (1:*).

Use single-direction filtering from dimensions to facts.

## 3. Power Query transformations

Use template in `power_query_srgnn_dashboard.m`.

Required normalization:
- `sessionId -> session_id`
- `itemId -> item_id`
- `eventdate -> event_date`
- numeric conversion for `hit_at_k`, `rank`, `metric_value`, `kpi_value`, `price_proxy`
- remove obvious duplicates by natural keys

### Timeframe policy (locked)
1. **Primary:** derive before/after from source data split when available.
2. **Fallback:** use simulated bounce-cut logic (existing notebook approach) or helper CSV fallback.

Standardized labels:
- `Trước (gốc)`
- `Sau (có model)`

## 4. DAX measures

Use measure pack in `dax_srgnn_measures.dax`. It includes:
- `HR@20`, `MRR@20`, `Coverage@20`
- SR-GNN and Co-occurrence variants
- Uplifts vs Co-occurrence
- Conversion rate and uplift (percentage points)
- AOV proxy and uplift
- Data quality counters

Formatting:
- HR/MRR: `0.0000`
- Percentages/uplifts: `0.0%`
- Uplift PP: `0.0 pp`

## 5. One-page layout (aligned to existing pages)

Keep current report style:
- Page size: **1280x720**
- Margin: **24px**
- Gap: **16px**
- Title textbox style: same as existing pages (`24pt`, top-left)
- Theme: existing `CY26SU04`

Page name:
- **SR-GNN Model Evaluation Dashboard**

Placement (3-30-300):
1. Top row: KPI cards (HR@20, MRR@20, Coverage@20, HR uplift, MRR uplift, Conversion uplift, AOV uplift)
2. Mid-left: offline comparison table + HR/MRR/Coverage charts
3. Mid-right: HR by session length + conversion by views (before/after)
4. Bottom-left: funnel
5. Bottom-middle: session segmentation
6. Bottom-right: AOV proxy + data quality/methodology panel

Color mapping (within existing theme palette):
- Popularity: neutral gray
- Co-occurrence: blue/purple
- SR-GNN: teal/green
- Before: gray
- After: teal/green
- Warning: orange

Vietnamese labels:
- `Trước (gốc)`, `Sau (có model)`
- `Tỷ lệ chuyển đổi`
- `Số views trong session`
- `Phân nhóm session`
- `AOV Proxy`

## 6. Metric formulas

- `HR@20 = average(hit_at_k) at k=20`
- `MRR@20 = average(reciprocal_rank) for top-20`
- `Coverage@20 = distinct recommended items / total catalog items`
- `Uplift = (SR-GNN - baseline) / baseline`
- `Conversion Rate = sessions_with_purchase / total_sessions`
- `Purchase Rate Uplift PP = after - before`
- `AOV Proxy = average(price_proxy of purchased items)`
- `AOV Proxy Uplift = (after - before) / before`

If precomputed values exist in `fact_metrics` or `fact_marketing_kpis`, use those first.

## 7. Refresh steps

1. Build marts:
```bash
make marts
```
2. Build helper CSVs (optional but recommended):
```bash
PYTHONPATH=src python scripts/build_srgnn_dashboard_data.py
```
3. Open PBIP/PBIX and refresh all queries.
4. Validate key cards against reference rows in:
   - `pbi_offline_metrics.csv`
   - `pbi_hr_by_bucket.csv`
   - `pbi_cr_by_session.csv`
   - `pbi_funnel.csv`
   - `pbi_rfm_segments.csv`
   - `pbi_aov_proxy.csv`

## 8. Interpretation caveats (must show in report)

- Metrics are **offline proxy metrics**, not online causal impact.
- `CTR Proxy` uses hit-rate logic, not impression-based CTR.
- `AOV Proxy` is relative (derived from transformed item price features), not audited revenue.
- Before/after comparisons are directional unless based on randomized experimentation.
