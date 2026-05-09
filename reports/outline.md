# Final DDM Report Outline

## 1. Introduction

- Present the project as a session-based recommendation analytics layer for the Diginetica e-commerce dataset.
- State the main question: can session behavior improve next-product recommendation, and how can that be translated into marketing-safe offline KPIs?

## 2. Problem Context: Anonymous Session-Based E-Commerce Behavior

- Explain that the session is the main analytical unit.
- Use the EDA to show that the dataset is interaction-heavy: item views dominate the raw tables, while purchases are much smaller.
- Show that `userId` is heavily missing in `item_views` and `purchases`, so many behavioral journeys are anonymous or weakly identified.
- Traditional user-level methods such as RFM, CLV, and churn modeling are less suitable as the main story under high user anonymity and sparse user identity.
- Therefore, the project shifts the unit of analysis from user to session and asks how well session behavior can support offline next-click capture.

## 3. Data Description and Cleaning

- Core tables:
  - `train-item-views.csv`: session behavior and next-click context.
  - `products.csv`: `pricelog2` price proxy.
  - `product-categories.csv`: item/category enrichment.
  - `train-purchases.csv`: purchase/session value proxy.
- Optional supporting tables:
  - `train-queries.csv`
  - `train-clicks.csv`
- Cleaned outputs:
  - `clean_item_views`
  - `clean_purchases`
  - `dim_item`
  - `fact_session_summary`
- Cleaning validation points:
  - Parse event dates and standardize IDs.
  - Convert `pricelog2` into `price_proxy = 2^pricelog2 - 1`.
  - Treat non-positive `price_proxy` values as missing, not as real value.
  - Add `session_length_bucket` to support session-centered EDA and metric slicing.
  - Add simple item popularity buckets for recommendation-oriented drilldowns.

## 4. Visual EDA Story

- Dataset overview:
  - Row counts by raw table show that the data is primarily interaction behavior.
  - Duplicate row counts are checked before cleaning.
  - Missing `userId` rates explain why the session is the practical analysis unit.
- Session behavior:
  - Session length, unique items per session, and repeat ratio describe short sequential journeys.
  - Views over time show event volume stability and date coverage.
  - Top viewed items and categories show concentration in the interaction stream.
- Purchase and value proxy:
  - Purchase session rate is sparse and should be treated as a purchase/session proxy.
  - Purchase rate by session length bucket connects session depth to purchase propensity without claiming causality.
  - `pricelog2` and `price_proxy` distributions explain value-proxy coverage.
  - Purchase value proxy by category supports value-oriented offline analysis without claiming audited revenue.
- Recommendation-oriented EDA:
  - HR@20, MRR@20, and Catalog Coverage@20 compare offline next-click behavior across models.
  - Target rank buckets show where the held-out next item appears when captured.
  - HR@20 by session length bucket tests whether sequence depth changes offline next-click capture.
  - Revenue-weighted HR@20 or Captured GMV Proxy@20 may be used only as value proxies.

## 5. Methodology: Inherited Recommender Artifacts

- Use a configured trained recommender artifact from DagsHub/MLflow.
- The default is the canonical `recsys-serving` model with alias `Production`, but a pinned model version can be used for reproducibility or intermediate analysis.
- Download the trained `registered_model` artifact into `data/inherited/recsys/`.
- Inherit compatible intermediate context artifacts from `recsys-group-project`, especially `test_examples.parquet` and `item_vocab.json`.
- Run final offline top-20 inference in this DDM repo on those compatible test examples.
- Do not retrain SR-GNN in the DDM repo.
- Do not re-split model evaluation data.
- Validate the manifest, model card, test examples, prediction rows, item vocabulary, and metrics before computing marts.
- Explain why SR-GNN is reasonable: it models item transitions inside session graphs, which matches the observed session-centered data structure.

## 6. Offline Evaluation Metrics

- HR@20: whether the held-out next item appears in the top-20 list.
- MRR@20: reciprocal rank of the held-out next item.
- Catalog Coverage@20: unique recommended items divided by the known catalog size.
- Report HR@20 and MRR@20 from the DDM-generated top-20 prediction rows using the inherited trained model.
- Compare against DDM-computed classic baselines: train-split popularity and one-step co-occurrence/MBA.
- Use safe wording: offline next-click capture, not real CTR or recommendation-caused conversion.

## 7. Marketing Proxy KPIs

- Recommendation Success Rate@20 = HR@20.
- CTR Proxy@20 = HR@20 expressed in business language; this is not real CTR.
- Purchase Session Rate = sessions with purchase / total sessions.
- Hit Rate@20 among purchase sessions.
- Captured GMV Proxy@20 = sum of `price_proxy(target_item) * hit@20`.
- Revenue-weighted HR@20 = price-weighted hit rate over target items with price coverage.
- Captured Purchase Value Proxy@20 = captured target value where the target item is also purchased in the same session.
- All purchase and value metrics are offline proxies. Do not claim real revenue, causal uplift, ROAS, or conversion caused by recommendation.

## 8. Results

- Show SR-GNN inherited prediction-row metrics:
  - HR@20
  - MRR@20
  - Catalog Coverage@20
- Compare model HR@20, MRR@20, and coverage against popularity and co-occurrence classic baselines.
- Discuss catalog coverage tradeoffs.
- Discuss marketing proxy KPIs with safe language:
  - offline next-click capture
  - purchase proxy
  - value proxy
  - not real CTR
  - not causal revenue or conversion uplift

## 9. Power BI Dashboard Design

- Page 1: Session overview and purchase proxy summary.
- Page 2: Recommendation model comparison with HR@20, MRR@20, and coverage.
- Page 3: Marketing proxy KPIs and value-proxy capture.
- Page 4: Item/category drilldown using `dim_item` and recommendation rows.

## 10. Limitations

- No recommendation impression logs, so real CTR cannot be computed.
- No online experiment or exposure assignment, so causal conversion or revenue uplift cannot be claimed.
- `price_proxy` is derived from `pricelog2`; it is not audited price or real revenue.
- SR-GNN top-20 rows are offline predictions, not observed recommendation impressions.
- Baselines are offline classic-method benchmarks, not production systems.
- Recommendation metrics are offline proxies and should not be interpreted as ROAS or marketing incrementality.

## 11. Conclusion

- Reaffirm that the session is the correct practical unit for this dataset.
- Summarize whether session-based recommendation improves offline next-click capture versus simple classic baselines.
- State what additional data would be needed for real marketing measurement: impressions, recommendation exposure, clicks, purchases after exposure, and online experiment assignment.
