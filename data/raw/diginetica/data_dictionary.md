# Diginetica Raw Data Dictionary

Expected source files from the Kaggle Diginetica dataset:

- `product-categories.csv`
- `products.csv`
- `train-clicks.csv`
- `train-item-views.csv`
- `train-purchases.csv`
- `train-queries.csv`

Local row counts after download:

| File | Rows |
|---|---:|
| `product-categories.csv` | 184047 |
| `products.csv` | 184047 |
| `train-clicks.csv` | 1127764 |
| `train-item-views.csv` | 1235380 |
| `train-purchases.csv` | 18025 |
| `train-queries.csv` | 923127 |

## `train-item-views.csv`

| Column | Type | Description |
|---|---|---|
| `sessionId` | integer/string | Session identifier. |
| `userId` | integer/string/null | User identifier when available. May be missing. |
| `itemId` | integer | Raw business item identifier. |
| `timeframe` | integer | Within-session ordering signal. |
| `eventdate` | date/string | Interaction date. |

## `products.csv`

| Column | Type | Description |
|---|---|---|
| `itemId` | integer | Raw business item identifier. |
| `pricelog2` | numeric | Encoded product price. Use `price = 2^pricelog2 - 1` as a price proxy. |
| `product.name.tokens` | string | Tokenized product name text. |

## `product-categories.csv`

| Column | Type | Description |
|---|---|---|
| `itemId` | integer | Raw business item identifier. |
| `categoryId` | integer | Product category identifier. |

## `train-purchases.csv`

| Column | Type | Description |
|---|---|---|
| `sessionId` | integer/string | Session identifier. |
| `userId` | integer/string/null | User identifier when available. |
| `timeframe` | integer | Within-session ordering signal. |
| `eventdate` | date/string | Purchase date. |
| `ordernumber` | integer/string | Order identifier. |
| `itemId` | integer | Purchased item ID. |

## `train-queries.csv`

| Column | Type | Description |
|---|---|---|
| `queryId` | integer/string | Search query identifier. |
| `sessionId` | integer/string | Session identifier. |
| `userId` | integer/string/null | User identifier when available. |
| `timeframe` | integer | Within-session ordering signal. |
| `duration` | integer | Query duration signal. |
| `eventdate` | date/string | Query date. |
| `searchstring.tokens` | string | Tokenized search query text. |
| `categoryId` | integer | Query category context when available. |
| `items` | string | Candidate/result item IDs shown for the query. |
| `is.test` | boolean/string | Dataset test flag from the original source. |

## `train-clicks.csv`

| Column | Type | Description |
|---|---|---|
| `queryId` | integer/string | Search query identifier. |
| `timeframe` | integer | Click timing signal. |
| `itemId` | integer | Clicked item ID. |

## Missing Fields And Impact

| Missing field | Impact |
|---|---|
| Recommendation impression ID/list | Real CTR cannot be computed. |
| Recommendation exposure position | Position-based recommender CTR cannot be computed. |
| Campaign exposure | Campaign lift and attribution cannot be computed. |
| Online experiment assignment | Causal model impact cannot be computed. |

Recommended metric framing:

- Use `HR@20` and `MRR@20` as native offline recommendation metrics.
- Use `Recommendation Success Rate@20` as a business-friendly label for `HR@20`.
- Use `CTR Proxy@20` only when explicitly defined as `HR@20`, not real CTR.
- Compute revenue proxies by joining predictions/targets to `products.pricelog2` and purchases, then label them as offline proxies.

