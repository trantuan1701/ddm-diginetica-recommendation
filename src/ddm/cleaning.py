"""Lightweight Diginetica cleaning helpers for the first DDM data layer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


RENAME_COLUMNS = {
    "sessionId": "session_id",
    "userId": "user_id",
    "itemId": "item_id",
    "categoryId": "category_id",
    "queryId": "query_id",
    "ordernumber": "order_number",
    "product.name.tokens": "product_name_tokens",
    "searchstring.tokens": "searchstring_tokens",
    "is.test": "is_test",
}

ID_COLUMNS = [
    "session_id",
    "user_id",
    "item_id",
    "category_id",
    "query_id",
    "order_number",
]

INTEGER_COLUMNS = [*ID_COLUMNS, "timeframe", "duration"]


def _first_valid(values: pd.Series) -> object:
    """Return the first non-null value in a Series, or NA when none exists."""
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def _join_sorted_unique(values: Iterable[object]) -> str | None:
    """Return sorted unique values as a comma-delimited string."""
    cleaned = pd.Series(values).dropna()
    if cleaned.empty:
        return None
    unique_values = sorted({int(value) for value in cleaned})
    return ",".join(str(value) for value in unique_values)


def standardize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common raw ID columns to snake_case and use nullable integers."""
    out = df.rename(columns={k: v for k, v in RENAME_COLUMNS.items() if k in df.columns}).copy()

    for column in INTEGER_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    if "pricelog2" in out.columns:
        out["pricelog2"] = pd.to_numeric(out["pricelog2"], errors="coerce").astype("Float64")

    if "is_test" in out.columns:
        out["is_test"] = out["is_test"].map(
            {True: True, False: False, "TRUE": True, "FALSE": False, "true": True, "false": False}
        )

    return out


def parse_event_dates(
    df: pd.DataFrame,
    source_column: str = "eventdate",
    output_column: str = "event_date",
    drop_source: bool = True,
) -> pd.DataFrame:
    """Parse the raw Diginetica event date column into a datetime column."""
    out = df.copy()
    if source_column not in out.columns:
        return out
    out[output_column] = pd.to_datetime(out[source_column], errors="coerce")
    if drop_source and source_column != output_column:
        out = out.drop(columns=[source_column])
    return out


def build_dim_item(products: pd.DataFrame, product_categories: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-item dimension table with category and price proxies."""
    products_clean = standardize_id_columns(products)
    categories_clean = standardize_id_columns(product_categories)

    dim = products_clean.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"]).copy()
    price_input = pd.to_numeric(dim["pricelog2"], errors="coerce")
    price_proxy = pd.Series(np.power(2.0, price_input.astype("float64")) - 1.0, index=dim.index)
    dim["price_proxy"] = price_proxy.where(np.isfinite(price_proxy) & price_proxy.gt(0)).astype("Float64")

    category_summary = (
        categories_clean.dropna(subset=["item_id", "category_id"])
        .groupby("item_id", as_index=False)
        .agg(
            primary_category_id=("category_id", "min"),
            category_count=("category_id", "nunique"),
            category_ids=("category_id", _join_sorted_unique),
        )
    )

    dim = dim.merge(category_summary, on="item_id", how="left")
    dim["primary_category_id"] = dim["primary_category_id"].astype("Int64")
    dim["category_count"] = dim["category_count"].fillna(0).astype("Int64")

    columns = [
        "item_id",
        "pricelog2",
        "price_proxy",
        "product_name_tokens",
        "primary_category_id",
        "category_count",
        "category_ids",
    ]
    return dim[[column for column in columns if column in dim.columns]].sort_values("item_id").reset_index(drop=True)


def add_item_popularity_features(dim_item: pd.DataFrame, clean_item_views: pd.DataFrame) -> pd.DataFrame:
    """Add simple train-data item popularity fields to the item dimension."""
    out = dim_item.copy()
    if clean_item_views.empty or "item_id" not in clean_item_views.columns:
        out["item_view_count"] = 0
    else:
        view_counts = clean_item_views.groupby("item_id").size().rename("item_view_count").reset_index()
        out = out.merge(view_counts, on="item_id", how="left")
        out["item_view_count"] = out["item_view_count"].fillna(0)

    out["item_view_count"] = pd.to_numeric(out["item_view_count"], errors="coerce").fillna(0).astype("Int64")
    out["item_popularity_bucket"] = (
        pd.cut(
            out["item_view_count"].astype("float64"),
            bins=[-1, 0, 1, 5, 20, 100, np.inf],
            labels=["unviewed", "1", "2-5", "6-20", "21-100", "101+"],
        )
        .astype("string")
        .fillna("unviewed")
    )
    return out


def build_clean_item_views(item_views: pd.DataFrame) -> pd.DataFrame:
    """Build the cleaned session item-view fact table."""
    out = parse_event_dates(standardize_id_columns(item_views))
    out = (
        out.dropna(subset=['session_id', 'item_id', 'event_date'])
           .drop_duplicates(
               subset=['session_id', 'item_id', 'event_date'],
               keep='first'
           )
    )
    columns = ['session_id', 'user_id', 'item_id', 'timeframe', 'event_date']
    sort_columns = ['session_id', 'event_date', 'timeframe', 'item_id']
    return (
        out[[c for c in columns if c in out.columns]]
        .sort_values([c for c in sort_columns if c in out.columns])
        .reset_index(drop=True)
    )


def build_clean_purchases(purchases: pd.DataFrame) -> pd.DataFrame:
    """Build the cleaned purchase fact table."""
    out = parse_event_dates(standardize_id_columns(purchases))
    out = (
        out.dropna(subset=['session_id', 'item_id', 'event_date'])
           .drop_duplicates(
               subset=['session_id', 'item_id', 'event_date'],
               keep='first'
           )
    )
    columns = ['session_id', 'user_id', 'timeframe', 'event_date', 'order_number', 'item_id']
    sort_columns = ['session_id', 'event_date', 'timeframe', 'order_number', 'item_id']
    return (
        out[[c for c in columns if c in out.columns]]
        .sort_values([c for c in sort_columns if c in out.columns])
        .reset_index(drop=True)
    )


def build_session_summary(
    clean_item_views: pd.DataFrame,
    clean_purchases: pd.DataFrame | None = None,
    dim_item: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact one-row-per-session mart table from views and purchases."""
    views = clean_item_views.copy()
    purchases = pd.DataFrame() if clean_purchases is None else clean_purchases.copy()

    view_summary = (
        views.groupby("session_id", as_index=False)
        .agg(
            user_id=("user_id", _first_valid),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
            view_count=("item_id", "size"),
            unique_viewed_items=("item_id", "nunique"),
            first_view_timeframe=("timeframe", "min"),
            last_view_timeframe=("timeframe", "max"),
        )
    )

    if purchases.empty:
        summary = view_summary
        summary["purchase_count"] = 0
        summary["unique_purchased_items"] = 0
        summary["order_count"] = 0
        summary["purchased_value_proxy"] = np.nan
        summary["has_purchase"] = False
    else:
        purchases_for_summary = purchases.copy()
        if dim_item is not None and {"item_id", "price_proxy"}.issubset(dim_item.columns):
            purchases_for_summary = purchases_for_summary.merge(
                dim_item[["item_id", "price_proxy"]],
                on="item_id",
                how="left",
            )
        else:
            purchases_for_summary["price_proxy"] = np.nan

        purchase_summary = (
            purchases_for_summary.groupby("session_id", as_index=False)
            .agg(
                purchase_user_id=("user_id", _first_valid),
                first_purchase_date=("event_date", "min"),
                last_purchase_date=("event_date", "max"),
                purchase_count=("item_id", "size"),
                unique_purchased_items=("item_id", "nunique"),
                order_count=("order_number", "nunique"),
                purchased_value_proxy=("price_proxy", lambda values: values.sum(min_count=1)),
            )
        )

        summary = view_summary.merge(purchase_summary, on="session_id", how="outer")
        summary["user_id"]          = summary["user_id"].combine_first(summary["purchase_user_id"])
        summary["first_event_date"] = summary["first_event_date"].combine_first(summary["first_purchase_date"])
        summary["last_event_date"]  = summary["last_event_date"].combine_first(summary["last_purchase_date"])
        summary["has_purchase"]     = summary["purchase_count"].fillna(0).gt(0)

        # Drop all intermediate columns
        summary = summary.drop(columns=[
            "purchase_user_id",
            "first_purchase_date",
            "last_purchase_date",
        ])

    count_columns = [
        "view_count",
        "unique_viewed_items",
        "purchase_count",
        "unique_purchased_items",
        "order_count",
    ]
    for column in count_columns:
        if column in summary.columns:
            summary[column] = summary[column].fillna(0).astype("Int64")

    summary["purchased_value_proxy"] = summary["purchased_value_proxy"].fillna(0.0)

    for column in ["user_id", "first_view_timeframe", "last_view_timeframe"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").astype("Int64")

    summary["session_length_bucket"] = (
        pd.cut(
            pd.to_numeric(summary["view_count"], errors="coerce").fillna(0).astype("float64"),
            bins=[-np.inf, 0, 1, 2, 5, 10, 20, np.inf],
            labels=["0", "1", "2", "3-5", "6-10", "11-20", "21+"],
        )
        .astype("string")
        .fillna("0")
    )

    return summary.sort_values("session_id").reset_index(drop=True)
