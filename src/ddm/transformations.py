"""Stage ② Data Transformation: Build dimensions, aggregates, and features from cleaned data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ddm.cleaning import standardize_id_columns


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


def _add_price_features(
    dim: pd.DataFrame,
    num_price_quantiles: int = 5,
) -> pd.DataFrame:
    """
    Add price-related features to item dimension table.
    
    Combines methods 2+4: quantile-based price bucketing with missing price flags.
    - is_price_known: Flag indicating whether pricelog2 value is available (not null/0/NaN)
    - pricelog2_known: Same as is_price_known (redundant but explicit per user request)
    - price_bucket: Quantile-based buckets (0-25%, 25-50%, 50-75%, 75-100%, or 'unknown')
    
    Args:
        dim: Item dimension table with pricelog2 column
        num_price_quantiles: Number of quantiles for price bucketing (default 5 creates quintiles)
    
    Returns:
        dim table with added price features
    """
    out = dim.copy()
    
    # Handle pricelog2: convert to numeric, treat 0 as missing
    price_input = pd.to_numeric(out["pricelog2"], errors="coerce")
    
    # Create price_known flags: True if price exists and is not zero
    has_valid_price = (price_input.notna()) & (price_input != 0.0)
    out["is_price_known"] = has_valid_price
    out["pricelog2_known"] = has_valid_price
    
    # Calculate price proxy for known prices
    price_proxy = pd.Series(
        np.power(2.0, price_input.astype("float64")) - 1.0,
        index=out.index
    )
    out["price_proxy"] = price_proxy.where(
        np.isfinite(price_proxy) & price_proxy.gt(0)
    ).astype("Float64")
    
    # Create quantile-based price buckets
    # For items with known prices, assign to quantile buckets
    # For items with unknown prices, assign to 'unknown' bucket
    out["price_bucket"] = "unknown"
    
    if has_valid_price.sum() > 0:
        known_prices = price_input[has_valid_price]
        
        # Use qcut to create quantile-based buckets
        try:
            bucket_labels = [f"q{i+1}" for i in range(num_price_quantiles)]
            price_quantiles = pd.qcut(
                known_prices,
                q=num_price_quantiles,
                labels=bucket_labels,
                duplicates="drop",  # Handle cases where quantiles can't be created
            )
            
            # Assign quantile buckets to items with known prices
            quantile_mapping = pd.Series(price_quantiles.values, index=known_prices.index)
            out.loc[has_valid_price, "price_bucket"] = quantile_mapping.astype("string")
        except Exception:
            # Fallback: if qcut fails, use simple bucket assignment
            price_proxy_known = out.loc[has_valid_price, "price_proxy"]
            out.loc[has_valid_price, "price_bucket"] = pd.cut(
                price_proxy_known,
                bins=num_price_quantiles,
                labels=[f"q{i+1}" for i in range(num_price_quantiles)],
                include_lowest=True,
            ).astype("string")
    
    out["price_bucket"] = out["price_bucket"].astype("string")
    
    return out


def build_dim_item(
    products: pd.DataFrame,
    product_categories: pd.DataFrame,
    clean_item_views: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a one-row-per-item dimension table with category, price, and popularity features.
    
    Processes:
    1. Price information through quantile-based bucketing with missing value handling:
       - Items with known prices (pricelog2 != 0) are assigned to quantile buckets (q1-q5)
       - Items with missing prices are labeled as 'unknown' price_bucket
       - Flags (is_price_known, pricelog2_known) indicate price availability
    
    2. Popularity metrics from item views:
       - item_view_count: Total views during training
       - item_popularity_bucket: Categorized popularity tier
    
    3. Category information from category mappings
    """
    products_clean = standardize_id_columns(products)
    categories_clean = standardize_id_columns(product_categories)

    dim = products_clean.dropna(subset=["item_id"]).drop_duplicates(subset=["item_id"]).copy()
    
    # Step 1: Add price features using quantile-based bucketing
    dim = _add_price_features(dim, num_price_quantiles=5)

    # Step 2: Add category information
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

    # Step 3: Add popularity features (integrated here instead of separate function)
    if clean_item_views is not None and not clean_item_views.empty and "item_id" in clean_item_views.columns:
        view_counts = clean_item_views.groupby("item_id").size().rename("item_view_count").reset_index()
        dim = dim.merge(view_counts, on="item_id", how="left")
        dim["item_view_count"] = dim["item_view_count"].fillna(0)
    else:
        dim["item_view_count"] = 0

    dim["item_view_count"] = pd.to_numeric(dim["item_view_count"], errors="coerce").fillna(0).astype("Int64")
    dim["item_popularity_bucket"] = (
        pd.cut(
            dim["item_view_count"].astype("float64"),
            bins=[-1, 0, 1, 5, 20, 100, np.inf],
            labels=["unviewed", "1", "2-5", "6-20", "21-100", "101+"],
        )
        .astype("string")
        .fillna("unviewed")
    )

    columns = [
        "item_id",
        "pricelog2",
        "is_price_known",
        "pricelog2_known",
        "price_bucket",
        "price_proxy",
        "product_name_tokens",
        "primary_category_id",
        "category_count",
        "category_ids",
        "item_view_count",
        "item_popularity_bucket",
    ]
    return dim[[column for column in columns if column in dim.columns]].sort_values("item_id").reset_index(drop=True)


def build_dim_user(clean_item_views: pd.DataFrame, clean_purchases: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build user dimension table from views and purchases.
    
    Includes:
    - user_id (PK, sentinel -1 for anonymous users)
    - is_anonymous: Flag for anonymous vs logged-in users
    - first_session_date: First interaction date
    - session_count: Total unique sessions
    - total_views: Total item views
    - total_purchases: Total purchase events
    """
    views = clean_item_views.copy() if not clean_item_views.empty else pd.DataFrame()
    purchases = pd.DataFrame() if clean_purchases is None else clean_purchases.copy()
    
    # Get unique user_ids from views
    user_ids_set = set()
    user_stats = {}
    
    if not views.empty and "user_id" in views.columns:
        user_view_stats = (
            views.groupby("user_id", as_index=False)
            .agg(
                first_session_date=("event_date", "min"),
                session_count=("session_id", "nunique"),
                total_views=("item_id", "size"),
            )
        )
        user_stats = {row["user_id"]: row for _, row in user_view_stats.iterrows()}
        user_ids_set.update(views["user_id"].dropna().unique())
    
    if not purchases.empty and "user_id" in purchases.columns:
        user_ids_set.update(purchases["user_id"].dropna().unique())
    
    if not user_ids_set:
        # No users found, return empty table
        return pd.DataFrame(columns=["user_id", "is_anonymous", "first_session_date", "session_count", "total_views", "total_purchases"])
    
    # Build dimension
    dim_user = pd.DataFrame({"user_id": sorted(user_ids_set)})
    dim_user["user_id"] = dim_user["user_id"].astype("Int64")
    
    # Fill in statistics from views
    dim_user = dim_user.merge(
        pd.DataFrame(user_stats).T.reset_index(drop=True),
        on="user_id",
        how="left"
    )
    
    # Add purchase statistics
    if not purchases.empty:
        user_purchase_stats = (
            purchases.groupby("user_id", as_index=False)
            .agg(total_purchases=("item_id", "size"))
        )
        dim_user = dim_user.merge(user_purchase_stats, on="user_id", how="left")
    
    # Fill defaults for missing data
    dim_user["total_views"] = pd.to_numeric(dim_user["total_views"], errors="coerce").fillna(0).astype("Int64")
    dim_user["total_purchases"] = pd.to_numeric(dim_user["total_purchases"], errors="coerce").fillna(0).astype("Int64")
    dim_user["session_count"] = pd.to_numeric(dim_user["session_count"], errors="coerce").fillna(1).astype("Int64")
    dim_user["first_session_date"] = pd.to_datetime(dim_user["first_session_date"], errors="coerce")
    dim_user["is_anonymous"] = dim_user["user_id"].eq(-1)
    
    columns = ["user_id", "is_anonymous", "first_session_date", "session_count", "total_views", "total_purchases"]
    return dim_user[[c for c in columns if c in dim_user.columns]].sort_values("user_id").reset_index(drop=True)


def build_dim_date(start_date: str = "2013-01-01", end_date: str = "2015-12-31") -> pd.DataFrame:
    """Build calendar dimension table for temporal joins.
    
    Includes:
    - date_key: Date (PK)
    - year, month, day, quarter, week
    - month_name, day_name
    - is_weekend: Boolean flag
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    dim_date = pd.DataFrame({
        "date_key": dates,
        "year": dates.year,
        "month": dates.month,
        "month_name": dates.strftime("%B"),
        "day": dates.day,
        "day_name": dates.strftime("%A"),
        "quarter": dates.quarter,
        "week": dates.isocalendar().week,
        "day_of_week": dates.dayofweek + 1,  # 1-7 (Monday-Sunday)
        "is_weekend": dates.dayofweek.isin([5, 6]),  # Saturday, Sunday
    })
    
    return dim_date.reset_index(drop=True)


def build_dim_query(queries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build search query dimension table.
    
    Includes:
    - query_id: Query identifier (PK)
    - searchstring_tokens: Tokenized search string
    
    Args:
        queries: Raw query data (optional)
    
    Returns:
        Query dimension table
    """
    if queries is None or queries.empty:
        return pd.DataFrame(columns=["query_id", "searchstring_tokens"])
    
    queries_clean = standardize_id_columns(queries)
    
    dim_query = (
        queries_clean.dropna(subset=["query_id"])
        .drop_duplicates(subset=["query_id"])
        .copy()
    )
    
    columns = ["query_id", "searchstring_tokens"]
    return dim_query[[c for c in columns if c in dim_query.columns]].sort_values("query_id").reset_index(drop=True)


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
