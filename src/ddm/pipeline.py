"""Minimal project runners for the DDM course workflow.

This module keeps the Makefile and notebooks thin. It does not train SR-GNN;
it only consumes inherited artifacts and computes analytics-layer outputs.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ddm.baselines import build_cooccurrence_baseline, build_popularity_baseline
from ddm.cleaning import (
    add_item_popularity_features,
    build_clean_item_views,
    build_clean_purchases,
    build_dim_item,
    build_session_summary,
)
from ddm.io import load_config, load_raw_tables, read_json, save_parquet
from ddm.kpis import (
    compute_model_proxy_kpis,
    enrich_scored_examples_for_value,
    native_metric_proxy_kpis,
    revenue_weighted_hit_rate,
)
from ddm.metrics import evaluate_topk_predictions, score_topk_predictions

SRGNN_MODEL_KEY = "srgnn_fc_v1_strict_filter_top20"
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


def _backbone_root(root: Path, config: dict[str, Any]) -> Path:
    return _resolve(root, config["paths"]["backbone_repo_path"]).resolve()


def _inheritance_root(root: Path, config: dict[str, Any]) -> Path:
    return _resolve(root, config["inheritance"]["inherited_root"])


def _load_vocab(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _id2item(vocab: dict[str, Any]) -> dict[int, int]:
    if "id2item" in vocab:
        return {int(k): int(v) for k, v in vocab["id2item"].items()}
    return {int(v): int(k) for k, v in vocab.get("item2id", {}).items()}


def _prefix_items(row: pd.Series) -> list[int]:
    x = list(row["x"])
    alias_inputs = list(row["alias_inputs"])
    return [int(x[int(alias)]) for alias in alias_inputs]


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


def inherit_recsys_context(project_root: str | Path = ".") -> dict[str, Path | None]:
    """Copy/prepare inherited SR-GNN context without retraining the model."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    backbone = _backbone_root(root, config)
    inherited_root = _inheritance_root(root, config)
    inherited_root.mkdir(parents=True, exist_ok=True)

    data_version = config["inheritance"]["data_version"]
    model_profile = config["inheritance"]["model_profile"]
    top_k = int(config["inheritance"]["top_k"])

    source_processed = backbone / "data" / "versions" / data_version / "processed"
    source_model = backbone / "models" / "experiments" / data_version / model_profile / "latest"

    source_vocab = source_processed / "item_vocab.json"
    source_metrics = _find_source_metrics(backbone, data_version, model_profile, source_model)
    source_test = source_processed / "test_examples.parquet"

    if not source_vocab.exists():
        raise FileNotFoundError(f"Missing inherited item vocab: {source_vocab}")
    if not source_test.exists():
        raise FileNotFoundError(f"Missing inherited test examples: {source_test}")

    vocab_path = inherited_root / "item_vocab.json"
    shutil.copy2(source_vocab, vocab_path)
    if source_metrics is not None:
        metrics_payload = _normalise_metrics_payload(read_json(source_metrics))
        (inherited_root / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )

    vocab = _load_vocab(vocab_path)
    internal_to_raw = _id2item(vocab)
    source_examples = pd.read_parquet(source_test).reset_index(drop=True)
    examples = source_examples.copy()
    examples.insert(0, "example_id", range(1, len(examples) + 1))
    examples["target_item_id_internal"] = examples["pos_items"].astype("int64")
    examples["target_item_id_raw"] = examples["target_item_id_internal"].map(internal_to_raw).astype("Int64")
    examples["prefix_item_ids_internal"] = examples.apply(_prefix_items, axis=1)
    examples["last_item_id_internal"] = examples["prefix_item_ids_internal"].apply(
        lambda items: int(items[-1]) if items else pd.NA
    )
    examples["last_item_id_raw"] = examples["last_item_id_internal"].map(internal_to_raw).astype("Int64")

    keep_columns = [
        "example_id",
        "session_id",
        "target_item_id_internal",
        "target_item_id_raw",
        "eventdate",
        "item_seq_len",
        "prefix_item_ids_internal",
        "last_item_id_internal",
        "last_item_id_raw",
        "x",
        "edge_index",
        "alias_inputs",
    ]
    test_examples_path = inherited_root / "test_examples.parquet"
    save_parquet(examples[keep_columns], test_examples_path)

    predictions_path = inherited_root / "predictions.parquet"
    source_predictions = _find_source_predictions(backbone, data_version, model_profile)
    if source_predictions is not None:
        predictions = pd.read_parquet(source_predictions)
        predictions = _normalise_prediction_export(predictions, examples, internal_to_raw)
        save_parquet(predictions, predictions_path)
        todo_path = None
    elif predictions_path.exists():
        todo = inherited_root / "PREDICTIONS_EXPORT_TODO.md"
        if todo.exists():
            todo.unlink()
        todo_path = None
    else:
        todo_path = _write_prediction_todo(
            inherited_root=inherited_root,
            backbone=backbone,
            source_model=source_model,
            source_test=source_test,
            top_k=top_k,
        )
        _try_export_predictions_if_torch_available(
            predictions_path=predictions_path,
            source_model=source_model,
            source_examples=source_examples,
            examples=examples,
            internal_to_raw=internal_to_raw,
            top_k=top_k,
            backbone=backbone,
        )
        if predictions_path.exists():
            if todo_path is not None and todo_path.exists():
                todo_path.unlink()
            todo_path = None

    return {
        "test_examples": test_examples_path,
        "predictions": predictions_path if predictions_path.exists() else None,
        "item_vocab": vocab_path,
        "metrics": inherited_root / "metrics.json" if (inherited_root / "metrics.json").exists() else None,
        "prediction_todo": todo_path,
    }


def _find_source_predictions(backbone: Path, data_version: str, model_profile: str) -> Path | None:
    candidates = [
        backbone / "data" / "versions" / data_version / "processed" / "predictions.parquet",
        backbone / "models" / "experiments" / data_version / model_profile / "latest" / "predictions.parquet",
        backbone / "metrics" / "experiments" / data_version / model_profile / "predictions.parquet",
    ]
    return next((path for path in candidates if path.exists()), None)


def _find_source_metrics(
    backbone: Path,
    data_version: str,
    model_profile: str,
    source_model: Path,
) -> Path | None:
    """Prefer the selected-model metrics when they point at this artifact."""
    best_model_path = backbone / "metrics" / "best_model.json"
    if best_model_path.exists():
        try:
            payload = read_json(best_model_path)
            best = payload.get("best_model", {})
            if (
                best.get("data_version") == data_version
                and best.get("model_profile") == model_profile
                and str(best.get("source", "")).endswith(str(source_model.relative_to(backbone)))
            ):
                return best_model_path
        except (KeyError, ValueError):
            pass

    metrics_path = source_model / "metrics.json"
    return metrics_path if metrics_path.exists() else None


def _normalise_metrics_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Return the flat HR/MRR payload used by DDM marts."""
    if "best_model" in payload:
        metrics = payload.get("best_model", {}).get("metrics", {}).get("test_metrics", {})
    elif "test_metrics" in payload:
        metrics = payload.get("test_metrics", {})
    else:
        metrics = payload
    return {key: float(value) for key, value in metrics.items() if key in {"hr@k", "mrr@k"}}


def _normalise_prediction_export(
    predictions: pd.DataFrame,
    examples: pd.DataFrame,
    internal_to_raw: dict[int, int],
) -> pd.DataFrame:
    out = predictions.copy()
    if "example_id" not in out.columns:
        if len(out) % len(examples) != 0:
            raise ValueError("Prediction export lacks example_id and cannot be aligned.")
        top_k = len(out) // len(examples)
        out.insert(0, "example_id", examples["example_id"].repeat(top_k).to_numpy())
    if "session_id" not in out.columns:
        out = out.merge(examples[["example_id", "session_id"]], on="example_id", how="left")
    if "pred_item_id_internal" not in out.columns:
        for candidate in ["item_id_internal", "item_id", "prediction"]:
            if candidate in out.columns:
                out["pred_item_id_internal"] = out[candidate]
                break
    if "pred_item_id_raw" not in out.columns and "pred_item_id_internal" in out.columns:
        out["pred_item_id_raw"] = out["pred_item_id_internal"].map(internal_to_raw).astype("Int64")
    if "rank" not in out.columns:
        out["rank"] = out.groupby("example_id").cumcount() + 1
    if "score" not in out.columns:
        out["score"] = pd.NA
    out["model_key"] = SRGNN_MODEL_KEY
    columns = [
        "model_key",
        "example_id",
        "session_id",
        "rank",
        "pred_item_id_internal",
        "pred_item_id_raw",
        "score",
    ]
    return out[columns]


def _write_prediction_todo(
    inherited_root: Path,
    backbone: Path,
    source_model: Path,
    source_test: Path,
    top_k: int,
) -> Path:
    todo_path = inherited_root / "PREDICTIONS_EXPORT_TODO.md"
    torch_state = "available" if importlib.util.find_spec("torch") else "not installed"
    todo_path.write_text(
        "\n".join(
            [
                "# SR-GNN Prediction Export TODO",
                "",
                "`predictions.parquet` was not found in the backbone repo.",
                f"Torch status in this DDM environment: `{torch_state}`.",
                "",
                "No SR-GNN retraining is needed. To export predictions, use the already trained model:",
                "",
                f"- Model directory: `{source_model}`",
                f"- Test examples: `{source_test}`",
                f"- Backbone repo: `{backbone}`",
                f"- Top K: `{top_k}`",
                "",
                "Expected output schema:",
                "",
                "- `example_id`",
                "- `session_id`",
                "- `rank`",
                "- `pred_item_id_internal`",
                "- `pred_item_id_raw`",
                "- `score` when available",
                "",
                "The DDM downstream code assumes this schema and will use native aggregate HR/MRR plus train-only baselines until the export exists.",
            ]
        ),
        encoding="utf-8",
    )
    return todo_path


def _try_export_predictions_if_torch_available(
    predictions_path: Path,
    source_model: Path,
    source_examples: pd.DataFrame,
    examples: pd.DataFrame,
    internal_to_raw: dict[int, int],
    top_k: int,
    backbone: Path,
) -> None:
    if importlib.util.find_spec("torch") is None:
        return
    sys.path.insert(0, str(backbone / "src"))
    from recsys.models.srgnn import SRGNNRecommender  # type: ignore

    model = SRGNNRecommender.load(source_model)
    rows: list[dict[str, object]] = []
    for idx, row in source_examples.reset_index(drop=True).iterrows():
        example = examples.iloc[idx]
        recs = model.recommend_from_graph(row["x"], row["edge_index"], row["alias_inputs"], top_k=top_k)
        for rank, internal_id in enumerate(recs, start=1):
            rows.append(
                {
                    "model_key": SRGNN_MODEL_KEY,
                    "example_id": int(example["example_id"]),
                    "session_id": int(example["session_id"]),
                    "rank": rank,
                    "pred_item_id_internal": int(internal_id),
                    "pred_item_id_raw": internal_to_raw.get(int(internal_id)),
                    "score": pd.NA,
                }
            )
    save_parquet(pd.DataFrame(rows), predictions_path)


def compute_metrics_and_kpis(project_root: str | Path = ".") -> dict[str, Path]:
    """Compute offline metrics, train-only baselines, recommendations, and KPI marts."""
    root = _project_root(project_root)
    config = load_config(root / "configs/project_config.yaml")
    inheritance = inherit_recsys_context(root)
    _ensure_clean_layer(root, config)

    inherited_root = _inheritance_root(root, config)
    mart_root = _resolve(root, config["outputs"]["mart_root"])
    backbone = _backbone_root(root, config)
    top_k = int(config["inheritance"]["top_k"])

    test_examples = pd.read_parquet(inheritance["test_examples"])
    item_vocab = read_json(inheritance["item_vocab"])
    catalog_size = int(item_vocab.get("size") or len(item_vocab.get("item2id", {})))
    dim_item = pd.read_parquet(mart_root / "dim_item.parquet")
    session_summary = pd.read_parquet(mart_root / "fact_session_summary.parquet")
    purchases = pd.read_parquet(_resolve(root, config["outputs"]["processed_root"]) / "clean_purchases.parquet")
    train_interactions = pd.read_parquet(
        backbone
        / "data"
        / "versions"
        / config["inheritance"]["data_version"]
        / "interim"
        / "train_interactions.parquet"
    )

    prediction_frames: list[pd.DataFrame] = []
    srgnn_predictions_path = inherited_root / "predictions.parquet"
    srgnn_predictions_available = srgnn_predictions_path.exists()
    if srgnn_predictions_available:
        srgnn_predictions = pd.read_parquet(srgnn_predictions_path)
        if "model_key" not in srgnn_predictions.columns:
            srgnn_predictions["model_key"] = SRGNN_MODEL_KEY
        prediction_frames.append(srgnn_predictions)

    popularity_predictions = build_popularity_baseline(
        train_interactions, test_examples, item_vocab=item_vocab, k=top_k, model_key=POPULARITY_MODEL_KEY
    )
    cooccurrence_predictions = build_cooccurrence_baseline(
        train_interactions, test_examples, item_vocab=item_vocab, k=top_k, model_key=COOCCURRENCE_MODEL_KEY
    )
    prediction_frames.extend([popularity_predictions, cooccurrence_predictions])
    fact_recommendations = pd.concat(prediction_frames, ignore_index=True)
    fact_recommendations = fact_recommendations[
        [
            "model_key",
            "example_id",
            "session_id",
            "rank",
            "pred_item_id_internal",
            "pred_item_id_raw",
            "score",
        ]
    ]

    metric_rows: list[dict[str, object]] = []
    scored_by_model: dict[str, pd.DataFrame] = {}

    native_metrics = _load_native_metrics(inheritance.get("metrics"))
    if native_metrics and not srgnn_predictions_available:
        for metric_name, value in [
            (f"HR@{top_k}", native_metrics.get("hr@k")),
            (f"MRR@{top_k}", native_metrics.get("mrr@k")),
        ]:
            metric_rows.append(
                {
                    "model_key": SRGNN_MODEL_KEY,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "k": top_k,
                    "metric_scope": "offline_test",
                    "source": "inherited_native_metrics",
                    "warning_text": "Offline next-click metric; not real CTR or causal business impact.",
                }
            )
        metric_rows.append(
            {
                "model_key": SRGNN_MODEL_KEY,
                "metric_name": f"Catalog Coverage@{top_k}",
                "metric_value": pd.NA,
                "k": top_k,
                "metric_scope": "offline_test",
                "source": "missing_prediction_export",
                "warning_text": "Requires SR-GNN top-k prediction rows; see inherited prediction export TODO.",
            }
        )

    for model_key, predictions in fact_recommendations.groupby("model_key", sort=False):
        if model_key == SRGNN_MODEL_KEY:
            source = "inherited_prediction_rows"
        else:
            source = "computed_train_only_baseline"
        metrics = evaluate_topk_predictions(
            predictions,
            test_examples,
            k=top_k,
            catalog_size=catalog_size,
            prediction_item_col="pred_item_id_internal",
            target_item_col="target_item_id_internal",
        )
        scored = score_topk_predictions(
            predictions,
            test_examples,
            k=top_k,
            prediction_item_col="pred_item_id_internal",
            target_item_col="target_item_id_internal",
        )
        scored = enrich_scored_examples_for_value(scored, test_examples, dim_item, purchases)
        scored_by_model[model_key] = scored
        for metric_name in [f"HR@{top_k}", f"MRR@{top_k}", f"Catalog Coverage@{top_k}"]:
            metric_rows.append(
                {
                    "model_key": model_key,
                    "metric_name": metric_name,
                    "metric_value": metrics[metric_name],
                    "k": top_k,
                    "metric_scope": "offline_test",
                    "source": source,
                    "warning_text": "Offline next-click metric; not real CTR or causal business impact.",
                }
            )

    fact_metrics = pd.DataFrame(metric_rows)
    fact_test_examples = _build_fact_test_examples(test_examples, dim_item, session_summary)
    eval_frames = [
        _build_fact_recommendation_eval(model_key, scored, session_summary)
        for model_key, scored in scored_by_model.items()
    ]
    fact_recommendation_eval = (
        pd.concat(eval_frames, ignore_index=True) if eval_frames else _empty_recommendation_eval()
    )
    dim_model = _build_dim_model(top_k)
    baseline_scored = scored_by_model.get(POPULARITY_MODEL_KEY)
    baseline_values = None
    if baseline_scored is not None:
        baseline_values = {
            "hr": float(baseline_scored["hit_at_k"].mean()),
            "revenue_weighted_hr": revenue_weighted_hit_rate(baseline_scored),
        }

    kpi_frames: list[pd.DataFrame] = []
    if native_metrics and SRGNN_MODEL_KEY not in scored_by_model:
        kpi_frames.append(
            native_metric_proxy_kpis(
                SRGNN_MODEL_KEY,
                float(native_metrics["hr@k"]),
                k=top_k,
                baseline_hr=baseline_values["hr"] if baseline_values else None,
            )
        )
    for model_key, scored in scored_by_model.items():
        compare_to_popularity = baseline_values if model_key != POPULARITY_MODEL_KEY else None
        kpi_frames.append(
            compute_model_proxy_kpis(
                model_key,
                scored,
                k=top_k,
                session_summary=session_summary,
                baseline_values=compare_to_popularity,
            )
        )
    fact_marketing_kpis = pd.concat(kpi_frames, ignore_index=True) if kpi_frames else pd.DataFrame()

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


def _load_native_metrics(path: Path | str | None) -> dict[str, float]:
    if not path:
        return {}
    payload = read_json(path)
    return {key: float(value) for key, value in payload.items() if key in {"hr@k", "mrr@k"}}


def _build_fact_test_examples(
    test_examples: pd.DataFrame,
    dim_item: pd.DataFrame,
    session_summary: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "example_id",
        "session_id",
        "target_item_id_internal",
        "target_item_id_raw",
    ]
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
            column
            for column in ["session_id", "has_purchase", "session_length_bucket"]
            if column in session_summary.columns
        ]
        session_attrs = session_summary[session_columns].rename(
            columns={"has_purchase": "is_purchase_session"}
        )
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


def _build_dim_model(top_k: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_key": SRGNN_MODEL_KEY,
                "model_family": "SR-GNN",
                "model_role": "selected_backbone",
                "top_k": top_k,
                "source_type": "inherited_trained_artifact",
                "warning_text": "Offline next-click model; not a causal marketing measurement.",
            },
            {
                "model_key": POPULARITY_MODEL_KEY,
                "model_family": "Popularity",
                "model_role": "train_only_baseline",
                "top_k": top_k,
                "source_type": "computed_in_ddm_from_train_interactions",
                "warning_text": "Offline train-only baseline; not a production recommender.",
            },
            {
                "model_key": COOCCURRENCE_MODEL_KEY,
                "model_family": "Co-occurrence",
                "model_role": "train_only_baseline",
                "top_k": top_k,
                "source_type": "computed_in_ddm_from_train_interactions",
                "warning_text": "Offline train-only baseline; not a production recommender.",
            },
        ]
    )


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
            column
            for column in ["session_id", "has_purchase", "session_length_bucket"]
            if column in session_summary.columns
        ]
        session_attrs = session_summary[session_columns].rename(
            columns={"has_purchase": "is_purchase_session"}
        )
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
    else:
        raise SystemExit(f"Unknown command {command!r}. Use validate, inherit, metrics, or marts.")

    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
