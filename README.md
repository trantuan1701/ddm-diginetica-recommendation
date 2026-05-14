# DDM Diginetica Session Recommendation Analytics

This repo is the Data-Driven Marketing analytics layer for a Diginetica session-based recommendation project.

The main analytical unit is the **session**. In this dataset, `userId` is heavily missing or weak, so user-level RFM, CLV, churn prediction, and customer segmentation are less suitable as the main project story. The practical question is:

> Can session behavior recommend the next product more effectively, and can that improvement be translated into marketing-safe offline KPIs?

## Project Scope

This repo owns the Diginetica analytics, evaluation, and reporting layer. It
does not train, serve, deploy, or monitor the recommender. Those backend and
MLOps responsibilities live in the upstream SR-GNN repository:
[`tlong-ds/recsys-group-project`](https://github.com/tlong-ds/recsys-group-project).

It does:

- clean item-view and purchase data
- build `dim_item` and session/purchase marts
- inherit the trained recommender artifact from the DagsHub/MLflow model registry
- run final offline top-20 inference for the course reporting tables
- compute offline recommendation metrics
- compute session-centered marketing proxy KPIs
- prepare parquet mart tables for PostgreSQL and Power BI
- support the final report/dashboard narrative

It does not:

- retrain SR-GNN
- re-split model evaluation data
- own training, experiment, serving, Kubernetes, or monitoring code
- train or promote models
- claim real CTR, causal conversion lift, ROAS, or real revenue uplift

## Upstream SR-GNN Backend and MLOps Repo

The model backend is maintained in
[`tlong-ds/recsys-group-project`](https://github.com/tlong-ds/recsys-group-project).
Use that repository for:

- SR-GNN, GGNN, TAGNN, and baseline recommender implementation details
- data processing, training, evaluation, and experiment pipelines
- model selection, promotion, and DagsHub/MLflow registry publishing
- FastAPI serving and production prediction behavior
- Docker, Kubernetes, Terraform, CI/CD, and monitoring assets

This DDM repo treats that repo as the source of recommender truth. It consumes
the registered model artifact and compatible evaluation context artifacts such
as test examples, train examples, and item vocabulary. This repo then produces
offline metrics, marketing-safe KPI framing, parquet marts, Power BI assets,
and report-ready analysis.

## Core Data

Generated data is intentionally not committed. A fresh clone needs local raw
data and, for recommendation metrics, DagsHub/MLflow credentials that can read
the configured registry model artifact. The default is the promoted model, but
the same contract can pin a specific version.

## Fresh Clone Setup

From a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Diginetica CSV files under `data/raw/diginetica/`. The pipeline needs
at least `product-categories.csv`, `products.csv`, `train-item-views.csv`, and
`train-purchases.csv`. Add `train-clicks.csv` and `train-queries.csv` for the
raw validation notebook and supporting search-funnel analysis. More
source/download notes live in `data/raw/diginetica/README.md`.

For `make inherit`, set local credentials in `.env`:

```text
DAGSHUB_USERNAME=
DAGSHUB_USER_TOKEN=
# Optional explicit MLflow basic-auth overrides:
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

Registry settings live in `configs/project_config.yaml`:

- `registry.model_name: recsys-serving`
- `registry.model_alias: Production`
- `registry.model_version: ""` for alias mode, or a concrete version for pinned reproducibility
- `registry.artifact_path: registered_model`
- `inheritance.top_k: 20`
- `inheritance.context_repo_path: ../recsys-group-project`
- `inheritance.test_examples_path` and `inheritance.item_vocab_path`: optional overrides for intermediate recsys artifacts. If blank, DDM uses the compatible paths saved in the trained model artifact config.

Then run:

```bash
make validate
make inherit
make metrics
make marts
```

`make validate` only requires the Diginetica raw data. `make inherit` downloads
the configured trained model artifact, inherits compatible intermediate context
artifacts such as test examples and item vocabulary, runs final offline
inference, and validates the local DDM bundle. `make metrics` and `make marts`
fail closed if that inherited bundle is missing or invalid.

Raw data lives under `data/raw/diginetica/`:

```text
product-categories.csv
products.csv
train-clicks.csv
train-item-views.csv
train-purchases.csv
train-queries.csv
```

The main project uses:

- `train-item-views.csv` for session behavior
- `products.csv` for `pricelog2` and `price_proxy = 2^pricelog2 - 1`
- `product-categories.csv` for item/category enrichment
- `train-purchases.csv` for offline purchase/session value proxies

`train-queries.csv` and `train-clicks.csv` are optional supporting data.

## Inherited Recommender Context

Expected inherited artifact folder:

```text
data/inherited/recsys/recsys-serving_Production_top20/
```

Required files created by `make inherit`:

- `manifest.json`
- `model_card.json` or `model_card.md`
- `test_examples.parquet`
- `predictions.parquet`
- `item_vocab.json`
- `metrics.json`
- `model_artifact/` containing the inherited trained model files

Optional files:

- `item_mapping.parquet`
- `baseline_metrics.json`
- `item_metadata.parquet`

Classic baselines are computed in DDM from the compatible inherited train split
when `train_examples.parquet` is available through the model artifact config or
`inheritance.train_examples_path`. These baselines are offline reporting
benchmarks, not production systems.

`manifest.json` records model name, alias or version, run ID, artifact URI,
data version, model profile, top K, export timestamp, source repo, the
intermediate context artifact paths, and the fact that inference came from
`registered_model`. The DDM repo validates this bundle before computing any
marts.

## Metrics and KPI Framing

Offline recommendation metrics:

- `HR@20`
- `MRR@20`
- `Catalog Coverage@20`

Marketing-safe proxy KPIs:

- `Recommendation Success Rate@20 = HR@20`
- `CTR Proxy@20 = HR@20`, not real CTR
- `Purchase Session Rate`
- `Hit Rate@20 among purchase sessions`
- `Captured GMV Proxy@20`
- `Revenue-weighted HR@20`
- `Captured Purchase Value Proxy@20`

Revenue and GMV fields use `price_proxy` from `pricelog2`. They are offline proxies only.

## Workflow

```bash
make validate
make inherit
make metrics
make marts
make report
```

Equivalent direct commands:

```bash
PYTHONPATH=src python -m ddm.pipeline validate
PYTHONPATH=src python -m ddm.pipeline inherit
PYTHONPATH=src python -m ddm.pipeline metrics
PYTHONPATH=src python -m ddm.pipeline marts
PYTHONPATH=src python -m ddm.pipeline report
```

Main outputs:

```text
data/processed/clean_item_views.parquet
data/processed/clean_purchases.parquet
data/mart/dim_item.parquet
data/mart/dim_model.parquet
data/mart/fact_session_summary.parquet
data/mart/fact_purchases.parquet
data/mart/fact_test_examples.parquet
data/mart/fact_recommendation_eval.parquet
data/mart/fact_metrics.parquet
data/mart/fact_marketing_kpis.parquet
data/mart/fact_recommendations.parquet
```

## Notebooks

- `notebooks/01_validate_data_and_context.ipynb`: raw validation and cleaned first marts
- `notebooks/02_compute_metrics.ipynb`: inherited model context, metrics, final inference rows, and KPIs
- `notebooks/03_prepare_powerbi_tables.ipynb`: final mart tables and Power BI notes

## SR-GNN Power BI Implementation Assets

For the one-page **SR-GNN Model Evaluation Dashboard** implementation package
(required files, model relationships, Power Query template, DAX measures,
refresh workflow, and interpretation caveats), see:

- `dashboards/powerbi_data/README.md`
- `dashboards/powerbi_data/power_query_srgnn_dashboard.m`
- `dashboards/powerbi_data/dax_srgnn_measures.dax`
- `scripts/build_srgnn_dashboard_data.py`

## Safe Claim Boundary

This project can discuss offline next-click capture and proxy value capture. It cannot claim real CTR, real conversion uplift, causal revenue uplift, ROAS, or actual business impact because recommendation impressions, exposure assignment, and online experiment data are not available.
