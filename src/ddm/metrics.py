"""Offline recommendation metrics for next-item analytics.

The helpers in this module operate on long top-k prediction tables with one row
per recommended item. They intentionally avoid any model-training dependency so
the DDM repo can evaluate final inference rows from the inherited trained model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


def _normalise_item(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _prediction_item_column(predictions: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in predictions.columns:
        return preferred
    for column in ["pred_item_id_internal", "pred_item_id_raw", "item_id_internal", "item_id_raw"]:
        if column in predictions.columns:
            return column
    raise ValueError("Could not find a prediction item column.")


def _target_item_column(examples: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in examples.columns:
        return preferred
    for column in ["target_item_id_internal", "target_item_id_raw", "pos_items", "item_id"]:
        if column in examples.columns:
            return column
    raise ValueError("Could not find a target item column.")


def hit_rate_at_k(recommendations: Sequence[int], target_item: int, k: int = 20) -> float:
    """Return 1.0 when the target item appears in the top-k list, else 0.0."""
    target = _normalise_item(target_item)
    if target is None:
        return 0.0
    top_k_items = [_normalise_item(item) for item in recommendations[:k]]
    return float(target in top_k_items)


def mrr_at_k(recommendations: Sequence[int], target_item: int, k: int = 20) -> float:
    """Return reciprocal rank of the target item within top-k, else 0.0."""
    target = _normalise_item(target_item)
    if target is None:
        return 0.0
    for rank, item in enumerate(recommendations[:k], start=1):
        if _normalise_item(item) == target:
            return 1.0 / rank
    return 0.0


def target_rank_at_k(recommendations: Sequence[int], target_item: int, k: int = 20) -> int | None:
    """Return 1-based rank of the target item within top-k, else None."""
    target = _normalise_item(target_item)
    if target is None:
        return None
    for rank, item in enumerate(recommendations[:k], start=1):
        if _normalise_item(item) == target:
            return rank
    return None


def catalog_coverage_at_k(
    recommendations: pd.DataFrame | Iterable[Sequence[int]],
    catalog_size: int | None = None,
    k: int = 20,
    item_col: str | None = None,
) -> float:
    """Return unique recommended items in top-k divided by catalog size.

    When `catalog_size` is not provided, the denominator is the number of unique
    recommended items, which makes the result 1.0. Pass a known vocab/catalog
    size for a useful coverage value.
    """
    if isinstance(recommendations, pd.DataFrame):
        if recommendations.empty:
            return 0.0
        item_column = _prediction_item_column(recommendations, item_col)
        topk = recommendations.copy()
        if "rank" in topk.columns:
            topk = topk[pd.to_numeric(topk["rank"], errors="coerce").le(k)]
        unique_items = topk[item_column].dropna().astype("int64").nunique()
    else:
        items: set[int] = set()
        for row in recommendations:
            items.update(int(item) for item in row[:k] if not pd.isna(item))
        unique_items = len(items)

    denominator = int(catalog_size or unique_items or 1)
    return float(unique_items / denominator)


def score_topk_predictions(
    predictions: pd.DataFrame,
    test_examples: pd.DataFrame,
    k: int = 20,
    prediction_item_col: str | None = None,
    target_item_col: str | None = None,
) -> pd.DataFrame:
    """Return one scored row per example with hit and reciprocal-rank fields."""
    if predictions.empty:
        raise ValueError("Predictions are empty.")
    if test_examples.empty:
        raise ValueError("Test examples are empty.")
    if "example_id" not in predictions.columns or "example_id" not in test_examples.columns:
        raise ValueError("Both predictions and test_examples must contain example_id.")

    pred_col = _prediction_item_column(predictions, prediction_item_col)
    target_col = _target_item_column(test_examples, target_item_col)

    topk = predictions.copy()
    topk["rank"] = pd.to_numeric(topk["rank"], errors="coerce")
    topk = topk[topk["rank"].between(1, k, inclusive="both")]
    topk = topk.sort_values(["example_id", "rank"])

    grouped = topk.groupby("example_id", sort=False)[pred_col].apply(list).rename("recommendations")
    scored = test_examples[["example_id", target_col]].merge(
        grouped.reset_index(), on="example_id", how="left"
    )
    scored["recommendations"] = scored["recommendations"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    scored["target_rank"] = [
        target_rank_at_k(recs, target, k)
        for recs, target in zip(scored["recommendations"], scored[target_col])
    ]
    scored["target_rank"] = scored["target_rank"].astype("Int64")
    scored["hit_at_k"] = scored["target_rank"].notna().astype(float)
    scored["reciprocal_rank"] = scored["target_rank"].apply(
        lambda rank: 1.0 / int(rank) if pd.notna(rank) else 0.0
    )
    scored["mrr_at_k"] = scored["reciprocal_rank"]
    scored = scored.drop(columns=["recommendations"]).rename(columns={target_col: "target_item_id"})
    return scored


def evaluate_topk_predictions(
    predictions: pd.DataFrame,
    test_examples: pd.DataFrame,
    k: int = 20,
    catalog_size: int | None = None,
    prediction_item_col: str | None = None,
    target_item_col: str | None = None,
) -> dict[str, float]:
    """Compute HR@K, MRR@K, and Catalog Coverage@K from long predictions."""
    scored = score_topk_predictions(
        predictions,
        test_examples,
        k=k,
        prediction_item_col=prediction_item_col,
        target_item_col=target_item_col,
    )
    pred_col = _prediction_item_column(predictions, prediction_item_col)
    return {
        f"HR@{k}": float(scored["hit_at_k"].mean()) if not scored.empty else 0.0,
        f"MRR@{k}": float(scored["mrr_at_k"].mean()) if not scored.empty else 0.0,
        f"Catalog Coverage@{k}": catalog_coverage_at_k(
            predictions, catalog_size=catalog_size, k=k, item_col=pred_col
        ),
        "n_examples": float(len(test_examples)),
        "n_prediction_rows": float(len(predictions)),
    }


def recommendation_success_rate_at_k(
    recommendations: Sequence[int], target_item: int, k: int = 20
) -> float:
    """Business-friendly alias for Hit Rate@K, not real CTR."""
    return hit_rate_at_k(recommendations, target_item, k)


def summarize_offline_metrics(
    predictions: pd.DataFrame,
    test_examples: pd.DataFrame,
    k: int = 20,
    catalog_size: int | None = None,
) -> dict[str, float]:
    """Backward-compatible wrapper around `evaluate_topk_predictions`."""
    return evaluate_topk_predictions(
        predictions=predictions,
        test_examples=test_examples,
        k=k,
        catalog_size=catalog_size,
    )
