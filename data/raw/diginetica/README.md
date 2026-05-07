# Raw Diginetica Data

This directory is the main location for raw analytics data owned by the DDM course project.

Expected files:

```text
product-categories.csv
products.csv
train-clicks.csv
train-item-views.csv
train-purchases.csv
train-queries.csv
```

This raw data contains product catalog rows, product categories, item views, purchases, search queries, and search-result clicks.

Downloaded source:

```text
https://www.kaggle.com/datasets/profalbusdumbledore/diginetica-dataset
```

Download command used locally:

```bash
curl -L -o /tmp/diginetica-kaggle.zip \
  https://www.kaggle.com/api/v1/datasets/download/profalbusdumbledore/diginetica-dataset
unzip -o /tmp/diginetica-kaggle.zip -d data/raw/diginetica
```

Marketing-relevant fields:

- `products.pricelog2` can be converted to a price proxy with `price = 2^pricelog2 - 1`.
- `train-purchases.csv` provides purchased items and order numbers for purchase/revenue proxy analysis.
- `product-categories.csv` enables category-level merchandising views.
- `train-queries.csv` and `train-clicks.csv` support search funnel analysis.
- `train-item-views.csv` supports session behavior and inherited SR-GNN next-item evaluation context.

Known limitations:

- No recommendation impression logs.
- No campaign exposure data.
- No online A/B test assignment.
- Product prices are encoded as `pricelog2`; derived revenue is a proxy.
- Purchases are observed transactions, but recommendation exposure is not observed.

Because impression logs are unavailable, the project cannot compute real CTR. `CTR Proxy@20` may only be used as a business interpretation of `Hit Rate@20`.

Revenue proxy KPIs such as `Captured GMV@20`, `Revenue-weighted HR@20`, and purchase-value capture can use `products.pricelog2` only as offline proxy evidence. Do not claim real revenue lift or causal impact.

Large raw files are ignored by Git. Keep this README and `data_dictionary.md` tracked.
