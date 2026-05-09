"""Course-facing runners for the DDM analytics/reporting workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ddm.baselines import build_cooccurrence_baseline, build_popularity_baseline, validate_no_leakage
from ddm.cleaning import (
    add_item_popularity_features,
    build_clean_item_views,
    build_clean_purchases,
    build_dim_item,
    build_session_summary,
)
from ddm.io import load_config, load_raw_tables, read_json, save_parquet
from ddm.kpis import compute_model_proxy_kpis, enrich_scored_examples_for_value
from ddm.metrics import evaluate_topk_predictions, score_topk_predictions
from ddm.registry import (
    bundle_directory,
    bundle_paths,
    download_registry_bundle,
    validate_inherited_bundle,
)

POPULARITY_MODEL_KEY = "popularity_top20"
COOCCURRENCE_MODEL_KEY = "cooccurrence_top20"


def _project_root(path: str | Path = ".") -> Path:
    root = Path(path).resolve()
    if (root / "configs/project_config.yaml").exists():
        return root
    if (root.parent / "configs/project_config.yaml").exists():
        return root.parent
    raise FileNotFoundError("Could not locate configs/project_config.yaml")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _top_k(config: dict[str, Any]) -> int:
    return int(config.get("inheritance", {}).get("top_k", 20))


def build_clean_layer(project_root: str | Path = ".") -> dict[str, Path]:
    """Build cleaned session/item/purchase marts from raw Diginetica tables."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    required_tables = ["item_views", "purchases", "products", "product_categories"]
    tables = load_raw_tables(config, project_root=root, table_names=required_tables)

    clean_item_views = build_clean_item_views(tables["item_views"])
    clean_purchases = build_clean_purchases(tables["purchases"])
    dim_item = build_dim_item(tables["products"], tables["product_categories"])
    dim_item = add_item_popularity_features(dim_item, clean_item_views)
    fact_session_summary = build_session_summary(clean_item_views, clean_purchases, dim_item)

    processed_root = _resolve(root, config["outputs"]["processed_root"])
    mart_root = _resolve(root, config["outputs"]["mart_root"])
    outputs = {
        "clean_item_views": processed_root / "clean_item_views.parquet",
        "clean_purchases": processed_root / "clean_purchases.parquet",
        "dim_item": mart_root / "dim_item.parquet",
        "fact_session_summary": mart_root / "fact_session_summary.parquet",
    }
    save_parquet(clean_item_views, outputs["clean_item_views"])
    save_parquet(clean_purchases, outputs["clean_purchases"])
    save_parquet(dim_item, outputs["dim_item"])
    save_parquet(fact_session_summary, outputs["fact_session_summary"])
    return outputs


def inherit_recsys_context(project_root: str | Path = ".") -> dict[str, Path]:
    """Download model artifact, inherit compatible context, and validate bundle."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    bundle = download_registry_bundle(root, config)
    paths = bundle_paths(root, config)
    return {name: path for name, path in paths.items() if path.exists()}


def validate_local_inherited_context(project_root: str | Path = ".") -> dict[str, Path]:
    """Validate the already downloaded inherited bundle."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    bundle = bundle_directory(root, config)
    validate_inherited_bundle(bundle, config)
    paths = bundle_paths(root, config)
    return {name: path for name, path in paths.items() if path.exists()}


def compute_metrics_and_kpis(project_root: str | Path = ".") -> dict[str, Path]:
    """Compute offline metrics, recommendation eval rows, and proxy KPI marts."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    _ensure_clean_layer(root, config)
    inherited = validate_local_inherited_context(root)

    top_k = _top_k(config)
    mart_root = _resolve(root, config["outputs"]["mart_root"])
    processed_root = _resolve(root, config["outputs"]["processed_root"])

    test_examples = _normalise_test_examples(pd.read_parquet(inherited["test_examples"]))
    predictions = _normalise_predictions(pd.read_parquet(inherited["predictions"]), config)
    item_vocab = read_json(inherited["item_vocab"])
    catalog_size = int(item_vocab.get("size") or len(item_vocab.get("item2id", {})) or len(item_vocab.get("id2item", {})))
    dim_item = pd.read_parquet(mart_root / "dim_item.parquet")
    session_summary = pd.read_parquet(mart_root / "fact_session_summary.parquet")
    purchases = pd.read_parquet(processed_root / "clean_purchases.parquet")

    fact_recommendations = predictions[
        [
            "model_key",
            "example_id",
            "session_id",
            "rank",
            "pred_item_id_internal",
            "pred_item_id_raw",
            "score",
        ]
    ].copy()
    manifest = read_json(inherited["manifest"])
    train_interactions = _load_baseline_train_interactions(root, config, inherited, manifest)
    if train_interactions is not None:
        if not validate_no_leakage(train_interactions, test_examples):
            raise ValueError("Baseline train examples overlap test examples by date; refusing to compute baselines.")
        baseline_predictions = pd.concat(
            [
                build_popularity_baseline(
                    train_interactions,
                    test_examples,
                    item_vocab=item_vocab,
                    k=top_k,
                    model_key=POPULARITY_MODEL_KEY,
                ),
                build_cooccurrence_baseline(
                    train_interactions,
                    test_examples,
                    item_vocab=item_vocab,
                    k=top_k,
                    model_key=COOCCURRENCE_MODEL_KEY,
                ),
            ],
            ignore_index=True,
        )
        fact_recommendations = pd.concat([fact_recommendations, baseline_predictions], ignore_index=True)

    metric_rows: list[dict[str, object]] = []
    eval_frames: list[pd.DataFrame] = []
    kpi_frames: list[pd.DataFrame] = []
    scored_by_model: dict[str, pd.DataFrame] = {}

    for model_key, model_predictions in fact_recommendations.groupby("model_key", sort=False):
        metric_source = (
            "train_split_classic_baseline"
            if model_key in {POPULARITY_MODEL_KEY, COOCCURRENCE_MODEL_KEY}
            else "inherited_model_prediction_rows"
        )
        metric_warning = (
            "Offline classic-method benchmark; not a production model or causal business impact."
            if model_key in {POPULARITY_MODEL_KEY, COOCCURRENCE_MODEL_KEY}
            else "Offline next-click metric; not real CTR or causal business impact."
        )
        metrics = evaluate_topk_predictions(
            model_predictions,
            test_examples,
            k=top_k,
            catalog_size=catalog_size,
            prediction_item_col=_prediction_item_column(model_predictions),
            target_item_col=_target_item_column(test_examples),
        )
        scored = score_topk_predictions(
            model_predictions,
            test_examples,
            k=top_k,
            prediction_item_col=_prediction_item_column(model_predictions),
            target_item_col=_target_item_column(test_examples),
        )
        scored = enrich_scored_examples_for_value(scored, test_examples, dim_item, purchases)
        scored_by_model[model_key] = scored
        eval_frames.append(_build_fact_recommendation_eval(model_key, scored, session_summary))
        kpi_frames.append(compute_model_proxy_kpis(model_key, scored, k=top_k, session_summary=session_summary))
        for metric_name in [f"HR@{top_k}", f"MRR@{top_k}", f"Catalog Coverage@{top_k}"]:
            metric_rows.append(
                {
                    "model_key": model_key,
                    "metric_name": metric_name,
                    "metric_value": metrics[metric_name],
                    "k": top_k,
                    "metric_scope": "offline_test",
                    "source": metric_source,
                    "warning_text": metric_warning,
                }
            )

    metric_rows.extend(_optional_baseline_metric_rows(inherited.get("baseline_metrics"), top_k))
    fact_metrics = pd.DataFrame(metric_rows)
    fact_marketing_kpis = pd.concat(kpi_frames, ignore_index=True) if kpi_frames else pd.DataFrame()
    fact_test_examples = _build_fact_test_examples(test_examples, dim_item, session_summary)
    fact_recommendation_eval = (
        pd.concat(eval_frames, ignore_index=True) if eval_frames else _empty_recommendation_eval()
    )
    dim_model = _build_dim_model(inherited["manifest"], fact_recommendations["model_key"].unique().tolist(), top_k)

    outputs = {
        "fact_metrics": mart_root / "fact_metrics.parquet",
        "fact_marketing_kpis": mart_root / "fact_marketing_kpis.parquet",
        "fact_recommendations": mart_root / "fact_recommendations.parquet",
        "fact_test_examples": mart_root / "fact_test_examples.parquet",
        "fact_recommendation_eval": mart_root / "fact_recommendation_eval.parquet",
        "dim_model": mart_root / "dim_model.parquet",
    }
    save_parquet(fact_metrics, outputs["fact_metrics"])
    save_parquet(fact_marketing_kpis, outputs["fact_marketing_kpis"])
    save_parquet(fact_recommendations, outputs["fact_recommendations"])
    save_parquet(fact_test_examples, outputs["fact_test_examples"])
    save_parquet(fact_recommendation_eval, outputs["fact_recommendation_eval"])
    save_parquet(dim_model, outputs["dim_model"])
    return outputs


def prepare_powerbi_marts(project_root: str | Path = ".") -> dict[str, Path]:
    """Prepare PostgreSQL/Power BI-ready mart parquet tables."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    _ensure_clean_layer(root, config)
    metric_outputs = compute_metrics_and_kpis(root)

    mart_root = _resolve(root, config["outputs"]["mart_root"])
    purchases = pd.read_parquet(_resolve(root, config["outputs"]["processed_root"]) / "clean_purchases.parquet")
    dim_item = pd.read_parquet(mart_root / "dim_item.parquet")

    fact_purchases = purchases.merge(
        dim_item[["item_id", "price_proxy", "primary_category_id"]],
        on="item_id",
        how="left",
    ).copy()
    fact_purchases.insert(0, "purchase_id", range(1, len(fact_purchases) + 1))

    outputs = {
        "dim_item": mart_root / "dim_item.parquet",
        "fact_session_summary": mart_root / "fact_session_summary.parquet",
        "fact_purchases": mart_root / "fact_purchases.parquet",
        **metric_outputs,
    }
    save_parquet(fact_purchases, outputs["fact_purchases"])
    _write_powerbi_notes(mart_root)
    return outputs


def check_report_inventory(project_root: str | Path = ".") -> dict[str, Path]:
    """Check that dashboard/report tables and figure inventory exist."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    mart_root = _resolve(root, config["outputs"]["mart_root"])
    required_marts = [
        "dim_item.parquet",
        "dim_model.parquet",
        "fact_session_summary.parquet",
        "fact_purchases.parquet",
        "fact_test_examples.parquet",
        "fact_recommendations.parquet",
        "fact_recommendation_eval.parquet",
        "fact_metrics.parquet",
        "fact_marketing_kpis.parquet",
    ]
    missing = [name for name in required_marts if not (mart_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing report mart tables: {missing}. Run `make marts` first.")
    figures_dir = root / "reports" / "figures"
    if not figures_dir.exists() or not list(figures_dir.glob("*.png")):
        raise FileNotFoundError("No report figures found under reports/figures.")
    return {"mart_root": mart_root, "figures_dir": figures_dir}


def _normalise_test_examples(examples: pd.DataFrame) -> pd.DataFrame:
    out = examples.copy()
    if "target_item_id_internal" not in out.columns and "pos_items" in out.columns:
        out["target_item_id_internal"] = out["pos_items"]
    if "target_item_id_raw" not in out.columns and "target_item_id_internal" in out.columns:
        out["target_item_id_raw"] = pd.NA
    if "session_id" not in out.columns:
        out["session_id"] = pd.NA
    return out


def _normalise_predictions(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = predictions.copy()
    model_name = str(config.get("registry", {}).get("model_name", "recsys-serving"))
    if "model_key" not in out.columns:
        out["model_key"] = model_name
    if "session_id" not in out.columns:
        out["session_id"] = pd.NA
    if "pred_item_id_internal" not in out.columns:
        out["pred_item_id_internal"] = pd.NA
    if "pred_item_id_raw" not in out.columns:
        out["pred_item_id_raw"] = pd.NA
    if "score" not in out.columns:
        out["score"] = pd.NA
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    return out


def _prediction_item_column(predictions: pd.DataFrame) -> str:
    return "pred_item_id_internal" if predictions["pred_item_id_internal"].notna().any() else "pred_item_id_raw"


def _target_item_column(test_examples: pd.DataFrame) -> str:
    return (
        "target_item_id_internal"
        if test_examples["target_item_id_internal"].notna().any()
        else "target_item_id_raw"
    )


def _optional_baseline_metric_rows(path: Path | None, top_k: int) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    payload = read_json(path)
    rows: list[dict[str, object]] = []
    for model_key, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        for source_name, target_name in [
            ("hr@k", f"HR@{top_k}"),
            ("mrr@k", f"MRR@{top_k}"),
            ("coverage@k", f"Catalog Coverage@{top_k}"),
        ]:
            if source_name not in metrics:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "metric_name": target_name,
                    "metric_value": float(metrics[source_name]),
                    "k": top_k,
                    "metric_scope": "offline_test",
                    "source": "inherited_baseline_metrics",
                    "warning_text": "Train-safe baseline metric inherited from registry export.",
                }
            )
    return rows


def _load_baseline_train_interactions(
    root: Path,
    config: dict[str, Any],
    inherited: dict[str, Path],
    manifest: dict[str, Any],
) -> pd.DataFrame | None:
    """Load compatible train examples for classic-method offline baselines."""
    candidates: list[Path] = []
    inheritance = config.get("inheritance", {})
    source_repo = _context_repo_path(root, config)

    explicit = str(inheritance.get("train_examples_path") or "").strip()
    if explicit:
        candidates.append(_source_repo_path(source_repo, explicit))

    context_artifacts = manifest.get("context_artifacts", {})
    if isinstance(context_artifacts, dict):
        train_path = str(context_artifacts.get("train_examples_path") or "").strip()
        if train_path:
            candidates.append(Path(train_path))

    model_artifact_dir = inherited.get("model_artifact", root / "__missing__")
    bundle_train_examples = model_artifact_dir / "train_examples.parquet"
    if bundle_train_examples.exists():
        candidates.append(bundle_train_examples)

    model_config_path = model_artifact_dir / "config.json"
    if model_config_path.exists():
        model_config = read_json(model_config_path)
        data_cfg = model_config.get("data", {})
        if isinstance(data_cfg, dict) and data_cfg.get("train_examples_path"):
            candidates.append(_source_repo_path(source_repo, data_cfg["train_examples_path"]))

    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    return None


def _context_repo_path(root: Path, config: dict[str, Any]) -> Path:
    path_str = str(config.get("inheritance", {}).get("context_repo_path") or "").strip()
    path = Path(path_str) if path_str else Path(".")
    return path if path.is_absolute() else (root / path).resolve()


def _source_repo_path(source_repo: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else source_repo / path


def _build_fact_test_examples(
    test_examples: pd.DataFrame,
    dim_item: pd.DataFrame,
    session_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["example_id", "session_id", "target_item_id_internal", "target_item_id_raw"]
    out = test_examples[[column for column in columns if column in test_examples.columns]].copy()

    if "eventdate" in test_examples.columns:
        out["event_date"] = pd.to_datetime(test_examples["eventdate"], errors="coerce")
    elif "event_date" in test_examples.columns:
        out["event_date"] = pd.to_datetime(test_examples["event_date"], errors="coerce")
    else:
        out["event_date"] = pd.NaT

    item_columns = [
        column
        for column in [
            "item_id",
            "price_proxy",
            "primary_category_id",
            "item_view_count",
            "item_popularity_bucket",
        ]
        if column in dim_item.columns
    ]
    if "item_id" in item_columns and "target_item_id_raw" in out.columns:
        item_attrs = dim_item[item_columns].rename(
            columns={
                "item_id": "target_item_id_raw",
                "price_proxy": "target_price_proxy",
                "primary_category_id": "target_primary_category_id",
                "item_view_count": "target_item_view_count",
                "item_popularity_bucket": "target_item_popularity_bucket",
            }
        )
        out = out.merge(item_attrs, on="target_item_id_raw", how="left")

    for column in [
        "target_price_proxy",
        "target_primary_category_id",
        "target_item_view_count",
        "target_item_popularity_bucket",
    ]:
        if column not in out.columns:
            out[column] = pd.NA

    if {"session_id", "has_purchase"}.issubset(session_summary.columns) and "session_id" in out.columns:
        session_columns = [
            column for column in ["session_id", "has_purchase", "session_length_bucket"] if column in session_summary.columns
        ]
        session_attrs = session_summary[session_columns].rename(columns={"has_purchase": "is_purchase_session"})
        out = out.merge(session_attrs, on="session_id", how="left")
        out["is_purchase_session"] = out["is_purchase_session"].fillna(False)
    else:
        out["is_purchase_session"] = False
    if "session_length_bucket" not in out.columns:
        out["session_length_bucket"] = pd.NA

    return out[
        [
            "example_id",
            "session_id",
            "target_item_id_internal",
            "target_item_id_raw",
            "event_date",
            "target_price_proxy",
            "target_primary_category_id",
            "target_item_view_count",
            "target_item_popularity_bucket",
            "is_purchase_session",
            "session_length_bucket",
        ]
    ]


def _empty_recommendation_eval() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_key",
            "example_id",
            "session_id",
            "target_item_id_raw",
            "hit_at_k",
            "target_rank",
            "reciprocal_rank",
            "target_price_proxy",
            "target_primary_category_id",
            "target_item_view_count",
            "target_item_popularity_bucket",
            "captured_value_proxy",
            "is_purchase_session",
            "session_length_bucket",
        ]
    )


def _build_dim_model(manifest_path: Path, model_keys: list[str], top_k: int) -> pd.DataFrame:
    manifest = read_json(manifest_path)
    rows = []
    for model_key in model_keys:
        if model_key == POPULARITY_MODEL_KEY:
            model_family = "classic_popularity"
            model_role = "offline_classic_benchmark"
            source_type = "train_split_popularity_baseline"
            warning = "Offline classic-method benchmark; not a production model or causal marketing measurement."
        elif model_key == COOCCURRENCE_MODEL_KEY:
            model_family = "classic_cooccurrence"
            model_role = "offline_classic_benchmark"
            source_type = "train_split_transition_baseline"
            warning = "Offline classic-method benchmark; not a production model or causal marketing measurement."
        else:
            model_family = str(manifest.get("model_profile", "registered_recommender"))
            model_role = "inherited_configured_artifact"
            source_type = "mlflow_artifact_with_intermediate_context_inference"
            warning = "Offline next-click model; not a causal marketing measurement."
        rows.append(
            {
                "model_key": model_key,
                "model_family": model_family,
                "model_role": model_role,
                "top_k": top_k,
                "source_type": source_type,
                "warning_text": warning,
            }
        )
    return pd.DataFrame(rows)


def _build_fact_recommendation_eval(
    model_key: str,
    scored_examples: pd.DataFrame,
    session_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = scored_examples.copy()
    out["model_key"] = model_key

    if "target_item_id_raw" not in out.columns:
        out["target_item_id_raw"] = pd.NA
    if "target_rank" not in out.columns:
        out["target_rank"] = pd.NA
    if "reciprocal_rank" not in out.columns:
        out["reciprocal_rank"] = out.get("mrr_at_k", 0.0)
    if "target_price_proxy" not in out.columns:
        out["target_price_proxy"] = pd.NA
    for column in ["target_primary_category_id", "target_item_view_count", "target_item_popularity_bucket"]:
        if column not in out.columns:
            out[column] = pd.NA

    out["captured_value_proxy"] = pd.NA
    priced = out["target_price_proxy"].notna()
    out.loc[priced, "captured_value_proxy"] = (
        out.loc[priced, "target_price_proxy"].astype(float) * out.loc[priced, "hit_at_k"].astype(float)
    )

    if {"session_id", "has_purchase"}.issubset(session_summary.columns) and "session_id" in out.columns:
        session_columns = [
            column for column in ["session_id", "has_purchase", "session_length_bucket"] if column in session_summary.columns
        ]
        session_attrs = session_summary[session_columns].rename(columns={"has_purchase": "is_purchase_session"})
        drop_columns = [column for column in ["is_purchase_session", "session_length_bucket"] if column in out.columns]
        out = out.drop(columns=drop_columns)
        out = out.merge(session_attrs, on="session_id", how="left")
        out["is_purchase_session"] = out["is_purchase_session"].fillna(False)
    else:
        out["is_purchase_session"] = False
    if "session_length_bucket" not in out.columns:
        out["session_length_bucket"] = pd.NA

    out["target_rank"] = out["target_rank"].astype("Int64")
    return out[
        [
            "model_key",
            "example_id",
            "session_id",
            "target_item_id_raw",
            "hit_at_k",
            "target_rank",
            "reciprocal_rank",
            "target_price_proxy",
            "target_primary_category_id",
            "target_item_view_count",
            "target_item_popularity_bucket",
            "captured_value_proxy",
            "is_purchase_session",
            "session_length_bucket",
        ]
    ]


def _ensure_clean_layer(root: Path, config: dict[str, Any]) -> None:
    required = [
        _resolve(root, config["outputs"]["processed_root"]) / "clean_item_views.parquet",
        _resolve(root, config["outputs"]["processed_root"]) / "clean_purchases.parquet",
        _resolve(root, config["outputs"]["mart_root"]) / "dim_item.parquet",
        _resolve(root, config["outputs"]["mart_root"]) / "fact_session_summary.parquet",
    ]
    if not all(path.exists() for path in required):
        build_clean_layer(root)


def _write_powerbi_notes(mart_root: Path) -> None:
    notes = mart_root / "powerbi_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# Power BI Notes",
                "",
                "- Session is the central unit of analysis.",
                "- `userId` is too sparse for user-level RFM, CLV, churn, or segmentation as the main story.",
                "- `fact_recommendation_eval` is one row per model/test example for offline hit, rank, and value-proxy analysis.",
                "- CTR fields are offline Hit Rate proxies, not real CTR.",
                "- Revenue and GMV fields use `pricelog2` price proxies and are not audited revenue.",
                "- Relative-delta fields are offline benchmark deltas, not causal business impact.",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "metrics"
    root = _project_root(args[1] if len(args) > 1 else ".")

    if command == "validate":
        outputs = build_clean_layer(root)
    elif command == "inherit":
        outputs = inherit_recsys_context(root)
    elif command == "metrics":
        outputs = compute_metrics_and_kpis(root)
    elif command == "marts":
        outputs = prepare_powerbi_marts(root)
    elif command == "report":
        outputs = check_report_inventory(root)
    else:
        raise SystemExit(f"Unknown command {command!r}. Use validate, inherit, metrics, marts, or report.")

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
