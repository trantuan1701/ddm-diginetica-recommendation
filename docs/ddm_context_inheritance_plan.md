# DDM Context Inheritance Plan

## Purpose

The `ddm-diginetica-recommendation` repository should become a marketing analytics and reporting layer on top of the trained session-based SR-GNN recommender from `recsys-group-project`.

It should inherit model outputs, data lineage, vocabulary mappings, evaluation context, and business metadata from the backbone repo. It should not duplicate SR-GNN preprocessing, training, serving, deployment, Kubernetes, MLflow promotion, or monitoring logic.

The DDM layer should translate valid offline recommendation results into marketing-friendly KPIs while keeping claims technically correct. In particular, real CTR cannot be computed from the checked offline Diginetica-style interaction data because there are no recommendation impression logs.

## 1. Model Context To Inherit

The current selected model context in `recsys-group-project` is:

- Selection record: `metrics/best_model.json`
- Selected data version: `v1_strict_filter`
- Selected model profile: `srgnn_fc`
- Selected local artifact directory: `models/experiments/v1_strict_filter/srgnn_fc/latest`
- Model weights: `models/experiments/v1_strict_filter/srgnn_fc/latest/model.pt`
- Model metadata: `models/experiments/v1_strict_filter/srgnn_fc/latest/model.json`
- Runtime config: `models/experiments/v1_strict_filter/srgnn_fc/latest/config.json`
- Artifact metrics: `models/experiments/v1_strict_filter/srgnn_fc/latest/metrics.json`
- Pointer file: `models/experiments/v1_strict_filter/srgnn_fc/latest/pointer.txt`

Important selected model attributes:

| Field | Value |
|---|---|
| Architecture | `srgnn` |
| Variant | `srgnn-fc` |
| Model version | `0.1.0` |
| Embedding dimension | `128` |
| Hidden size | `128` |
| Propagation step | `1` |
| Max session length | `20` |
| Fallback weight | `0.05` |
| Seed | `42` |
| Known item count | `23072` |
| Evaluation/reporting K | `20` |

The serving promotion context is separate from the local selected artifact:

- Promotion record: `metrics/promotion_result.json`
- Canonical serving model: `recsys-serving`
- Promoted model version: `1`
- Promoted run ID: `e1896f8e3e024808a00652f6ca965a18`

The DDM repo should reference the inherited model by a manifest and exported predictions, not by importing training code. The backbone should produce or provide a fixed export bundle containing model metadata, predictions, metrics, mappings, and lineage. DDM should treat that bundle as read-only source data.

Recommended DDM location:

```text
ddm-diginetica-recommendation/
`-- data/
    `-- inherited/
        `-- recsys/
            `-- v1_strict_filter_srgnn_fc_top20/
```

## 2. Data Context To Inherit

The selected data version is `v1_strict_filter`.

Backbone source paths:

- Raw interactions: `data/raw/train-item-views.csv`
- Raw DVC metadata: `data/raw.dvc`
- Version config: `configs/data_versions/v1_strict_filter.yaml`
- Interim interactions: `data/versions/v1_strict_filter/interim`
- Processed examples: `data/versions/v1_strict_filter/processed`
- Data stats: `data/versions/v1_strict_filter/processed/data_stats.json`
- Validation report: `data/versions/v1_strict_filter/validation_report.json`

Important raw data facts:

- Raw file rows: about `1,235,380`
- Raw columns present locally: `sessionId`, `userId`, `itemId`, `timeframe`, `eventdate`
- No recommendation impression column exists.
- No price column exists in the checked raw interaction file.

Selected preprocessing rules:

| Rule | Value |
|---|---|
| Split strategy | `diginetica_legacy` |
| Test window | last 7 days |
| Validation window | 7 days before test |
| Minimum session length | `3` |
| Minimum item frequency | `10` |
| Maximum session length | `20` |
| Duplicate item handling | allowed |
| Training example format | graph |
| Vocab order | first seen |
| Example order | reverse |
| Minimum prefix length | `1` |

Selected processed data stats:

| Split | Interactions | Sessions | Items | Examples |
|---|---:|---:|---:|---:|
| Train | `618718` | `110030` | `23072` | `508688` |
| Validation | `40411` | `7248` | `11988` | `33095` |
| Test | `56144` | `10028` | `13600` | `45910` |

The DDM repo should reuse the existing processed test examples from `data/versions/v1_strict_filter/processed/test_examples.parquet`. It should not re-run splitting or create a new test set, because that would break comparability with the trained model and could introduce leakage.

## 3. Vocabulary And Mapping Context

The selected item vocabulary is:

```text
recsys-group-project/data/versions/v1_strict_filter/processed/item_vocab.json
```

Vocabulary schema:

- `item2id`: raw business `itemId` as string key to internal model ID.
- `id2item`: internal model ID as string key to raw business `itemId`.
- `size`: `23072`.
- `start_id`: `1`.

Mapping rules:

- Internal ID `0` is reserved for padding/unknown handling.
- Valid selected-model item IDs start at `1`.
- The vocabulary is built from the training split only.
- Validation and test examples drop unknown items during example generation.
- Prediction rows exported to DDM must include both internal model IDs and raw business item IDs.

OOV policy for DDM:

- For offline inherited test examples, DDM should not invent OOV replacements.
- If an exported prediction cannot map through `id2item`, mark it as unmapped and exclude it from business item-level reporting until fixed.
- For any future live-session analytics, unknown raw item IDs should be counted as OOV and reported separately, matching the backbone serving behavior.

## 4. Evaluation Context

The backbone evaluator measures session-based next-item prediction over processed graph examples.

Native metrics:

- `Hit Rate@K`
- `MRR@K`
- Optional additions for DDM reporting: `Recall@K`, `NDCG@K`

Selected model offline test metrics:

| Metric | Value |
|---|---:|
| HR@20 | `0.5416249183184492` from `metrics/best_model.json` selection |
| MRR@20 | `0.18315839816184917` from `metrics/best_model.json` selection |
| Artifact HR@20 | `0.5473334340534824` from selected artifact `metrics.json` |
| Artifact MRR@20 | `0.1840223540313709` from selected artifact `metrics.json` |

The small mismatch between selection and artifact metrics should be preserved in the model card until reconciled by a formal export manifest. Use `metrics/best_model.json` as the selection authority.

Test example format:

- Each row represents `session prefix -> next-click target item`.
- Graph columns: `x`, `edge_index`, `alias_inputs`, `item_seq_len`, `pos_items`, `session_id`, `eventdate`.
- `pos_items` is the encoded next-click target.
- `x` and `alias_inputs` encode the session prefix graph.

Baseline context:

- The backbone contains model popularity metadata in `model.json`, but no standalone marketing baseline export exists yet.
- DDM should request or generate leakage-safe baseline prediction exports from the backbone context.
- Popularity baseline should rank items by frequency in the selected training split only.
- Co-occurrence/MBA baseline should use only selected training split transitions or co-occurrence pairs.
- Baselines must not use validation/test target frequency, test sessions after the prefix target, or global item counts computed over the full raw dataset.

## 5. Marketing KPI Context

Real CTR cannot be computed from the offline Diginetica-style data currently checked into the backbone because the data contains click/view interaction records, not recommendation impressions. There is no denominator of displayed recommendations, no exposure position log, and no user assignment to a recommendation policy.

Valid offline interpretation:

- `Hit Rate@K` can be labeled as `CTR Proxy@K` only if defined as "the fraction of next-click targets captured in the model top-K list."
- This is not real CTR and should not be compared directly with online ad/recommender CTR.

Revenue-proxy KPIs should be prioritized when item price is available:

### Captured GMV@K

For each test example:

```text
hit@K = 1 if target_item is in top-K recommendations else 0
captured_gmv@K = price(target_item) * hit@K
Captured GMV@K = sum(captured_gmv@K across examples)
```

This estimates how much target-item value was captured by the top-K list. It is not realized revenue.

### Revenue-weighted Hit Rate@K

```text
Revenue-weighted HR@K =
  sum(price(target_item) * hit@K) / sum(price(target_item))
```

This answers: "What share of observed target-item value was captured by recommendations?"

### GMV Uplift Versus Baselines

```text
GMV uplift vs baseline =
  (model_captured_gmv@K - baseline_captured_gmv@K)
  / baseline_captured_gmv@K
```

Compute this against:

- Popularity baseline.
- Co-occurrence/MBA baseline.

If `baseline_captured_gmv@K` is zero, report absolute difference and mark percentage uplift as undefined.

Claims to allow:

- "The model captures X% of next-click targets at K."
- "The model captures X amount of price-weighted target value at K on offline test examples."
- "The model outperforms the popularity baseline by X points of HR@K / revenue-weighted HR@K."

Claims to avoid:

- "The model achieved real CTR."
- "The model increased revenue by X%."
- "The model caused GMV uplift."
- "The model improved ROAS/conversion rate."
- "The model performed better for exposed users."

Those claims require impression logs, recommendation exposure records, attribution windows, and preferably an online experiment.

## 6. Export Interface Between Repositories

The backbone repo should export an immutable context bundle. The DDM repo should consume the bundle, validate it, and build analytics tables from it.

Recommended bundle path in DDM:

```text
data/inherited/recsys/v1_strict_filter_srgnn_fc_top20/
```

Recommended files:

| File | Purpose |
|---|---|
| `manifest.json` | Source repo, export timestamp, selected data/model IDs, top-K, source paths, hashes, schema versions |
| `model_card.md` or `model_card.json` | Human-readable model context, metrics, limitations, selection and promotion metadata |
| `test_examples.parquet` | Analytics-friendly test examples with raw and internal item IDs |
| `predictions.parquet` | SR-GNN top-K predictions for each test example |
| `item_vocab.json` | Original vocabulary artifact |
| `item_mapping.parquet` | Flattened raw-to-internal and internal-to-raw item mapping |
| `metrics.json` | Native offline metrics such as HR@20 and MRR@20 |
| `baselines.parquet` | Baseline top-K predictions by example |
| `baseline_metrics.json` | HR/MRR/Recall/NDCG and revenue-proxy metrics for baselines |
| `item_metadata.parquet` | Optional item attributes: item ID, category, name, price |
| `marketing_kpis.parquet` or `marketing_kpis.json` | CTR proxy, captured GMV, revenue-weighted HR, uplift vs baselines |

Suggested schemas:

### `manifest.json`

```json
{
  "export_id": "v1_strict_filter_srgnn_fc_top20",
  "source_repo": "recsys-group-project",
  "data_version": "v1_strict_filter",
  "model_profile": "srgnn_fc",
  "model_variant": "srgnn-fc",
  "top_k": 20,
  "selection_source": "metrics/best_model.json",
  "artifact_path": "models/experiments/v1_strict_filter/srgnn_fc/latest",
  "promotion": {
    "model_name": "recsys-serving",
    "model_version": "1",
    "run_id": "e1896f8e3e024808a00652f6ca965a18"
  }
}
```

### `test_examples.parquet`

| Column | Type | Notes |
|---|---|---|
| `example_id` | string/int | Stable row identifier |
| `session_id` | string/int | Original session identifier |
| `eventdate` | timestamp/date | Target item event date |
| `prefix_item_ids_internal` | list[int] | Internal encoded prefix |
| `prefix_item_ids_raw` | list[int] | Business item IDs |
| `target_item_id_internal` | int | Encoded next-click target |
| `target_item_id_raw` | int | Business next-click target |

### `predictions.parquet`

| Column | Type | Notes |
|---|---|---|
| `example_id` | string/int | Joins to test examples |
| `session_id` | string/int/null | Original session identifier when available |
| `rank` | int | 1-based prediction rank |
| `pred_item_id_internal` | int | Encoded prediction |
| `pred_item_id_raw` | int | Business item ID |
| `score` | float/null | Optional if exported by backbone |

### `item_metadata.parquet`

| Column | Type | Notes |
|---|---|---|
| `item_id_raw` | int | Business item ID |
| `category_id` | int/null | Optional |
| `name` | string/null | Optional |
| `price` | float/null | Required for GMV KPIs |
| `price_source` | string/null | Catalog/source identifier |

### `marketing_kpis.json`

```json
{
  "k": 20,
  "hit_rate_at_k": 0.5416249183184492,
  "ctr_proxy_at_k": 0.5416249183184492,
  "captured_gmv_at_k": null,
  "revenue_weighted_hit_rate_at_k": null,
  "gmv_uplift_vs_popularity": null,
  "gmv_uplift_vs_cooccurrence": null,
  "price_coverage": 0.0,
  "limitations": [
    "CTR proxy is Hit Rate@K, not real CTR.",
    "GMV metrics require item price metadata."
  ]
}
```

## 7. DDM Repo Next Steps

Suggested folder structure:

```text
ddm-diginetica-recommendation/
|-- data/
|   |-- inherited/recsys/
|   |-- curated/
|   `-- mart/
|-- docs/
|-- notebooks/
|-- reports/
|-- sql/
`-- src/ddm_metrics/
```

Power BI database tables:

- `dim_model`: model profile, data version, artifact ID, run ID, top-K, metric source.
- `dim_item`: item ID, category, name, price, price coverage flag.
- `fact_test_examples`: one row per offline next-click example.
- `fact_recommendations`: one row per example and recommended item rank.
- `fact_baseline_recommendations`: one row per example, baseline, and rank.
- `fact_offline_metrics`: HR@K, MRR@K, Recall@K, NDCG@K by model/baseline.
- `fact_marketing_kpis`: CTR proxy, captured GMV, revenue-weighted HR, baseline uplift.

Notebook/report workflow:

1. Validate the inherited manifest and source file checksums.
2. Validate mapping completeness from internal IDs to raw item IDs.
3. Load test examples and SR-GNN predictions.
4. Compute native offline metrics and compare with inherited metrics.
5. Compute popularity and co-occurrence baseline metrics from train-only context.
6. Join item metadata and calculate price coverage.
7. Compute revenue-proxy KPIs only when price coverage is sufficient.
8. Publish curated tables for Power BI.

Implementation priority:

1. Create the inherited export contract and model card.
2. Export or ingest test examples, vocabulary, predictions, and native metrics.
3. Add validation checks for row counts, mappings, top-K, and metric reproducibility.
4. Add leakage-safe popularity and co-occurrence baselines.
5. Add item metadata and price coverage.
6. Compute Captured GMV@K, Revenue-weighted HR@K, and GMV uplift versus baselines.
7. Build database tables and Power BI dashboard.

## Acceptance Criteria

The DDM repo is ready for analytics implementation when:

- It can identify exactly which SR-GNN model, data version, vocabulary, and test split were inherited.
- It can map all exported predictions and targets back to raw business item IDs.
- It reports HR@20 and MRR@20 as native offline metrics.
- It treats `CTR Proxy@20` only as a business interpretation of HR@20.
- It does not claim real CTR, real revenue lift, or causal GMV impact.
- It computes revenue-proxy KPIs only when item price metadata exists.
- It compares SR-GNN against leakage-safe popularity and co-occurrence baselines.
- It does not duplicate backbone training, serving, or deployment code.
