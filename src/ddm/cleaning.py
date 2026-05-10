"""Stage ? Data Cleaning: Standardize, parse, and deduplicate raw Diginetica data.

This module handles only the cleaning phase (Raw ? Cleaned):
- Column standardization (rename, type conversion)
- Date parsing
- Deduplication of fact tables

For Stage ? (Cleaned ? Transformed dimensions/aggregates), see transformations.py
"""

from __future__ import annotations

from collections.abc import Iterable

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


def build_clean_item_views(item_views: pd.DataFrame) -> pd.DataFrame:
    """Build the cleaned session item-view fact table.
    
    Removes duplicates and standardizes columns.
    Grain: One row per (session_id, item_id, event_date) combination
    """
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
