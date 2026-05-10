"""Marketing-safe KPI helpers for offline recommendation analytics."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

PROXY_WARNING = (
    "Offline proxy only: not real CTR, not causal conversion, and not audited revenue."
)


def price_from_pricelog2(pricelog2: float | int | None) -> float | None:
    """Convert Diginetica `pricelog2` into a price proxy.

    DDM uses `2^pricelog2 - 1` as an offline price proxy, not audited revenue.
    """
    if pricelog2 is None or pd.isna(pricelog2):
        return None
    value = (2.0 ** float(pricelog2)) - 1.0
    return value if value > 0 else None


def ctr_proxy_from_hit_rate(hit_rate_at_k: float) -> float:
    """Return CTR Proxy@K as HR@K with explicit limitations.

    This is not real CTR because the offline data has no recommendation
    impression logs.
    """
    return float(hit_rate_at_k)


def captured_gmv_at_k(price: float | None, hit_at_k: float) -> float | None:
    """Return price-weighted captured value for one example when price exists."""
    if price is None:
        return None
    return float(price) * float(hit_at_k)


def captured_purchase_value_at_k(
    purchase_price: float | None, recommended_before_purchase: float
) -> float | None:
    """Return captured purchase value proxy when a purchased item is recommended.

    This is useful for marketing scenarios, but remains offline proxy evidence
    because recommendation impressions are not observed in the raw data.
    """
    if purchase_price is None:
        return None
    return float(purchase_price) * float(recommended_before_purchase)


def revenue_weighted_hit_rate(
    scored_examples: pd.DataFrame,
    price_col: str = "target_price_proxy",
    hit_col: str = "hit_at_k",
) -> float | None:
    """Return price-weighted HR@K from per-example hit flags."""
    if scored_examples.empty or price_col not in scored_examples.columns or hit_col not in scored_examples.columns:
        return None
    frame = scored_examples[[price_col, hit_col]].dropna()
    denominator = float(frame[price_col].sum())
    if denominator == 0:
        return None
    return float((frame[price_col] * frame[hit_col]).sum() / denominator)


def gmv_uplift_vs_baseline(model_gmv: float, baseline_gmv: float) -> float | None:
    """Return relative GMV proxy uplift, or None when baseline GMV is zero."""
    if baseline_gmv == 0:
        return None
    return (float(model_gmv) - float(baseline_gmv)) / float(baseline_gmv)


def _kpi_row(
    model_key: str,
    k: int | None,
    kpi_name: str,
    kpi_value: float | None,
    scope: str,
    warning_text: str = PROXY_WARNING,
) -> dict[str, object]:
    return {
        "model_key": model_key,
        "k": k,
        "kpi_name": kpi_name,
        "kpi_value": kpi_value,
        "kpi_scope": scope,
        "warning_text": warning_text,
    }


def purchase_session_rate(session_summary: pd.DataFrame) -> float | None:
    """Return sessions with purchase divided by total sessions."""
    if session_summary.empty or "has_purchase" not in session_summary.columns:
        return None
    return float(session_summary["has_purchase"].fillna(False).mean())


def enrich_scored_examples_for_value(
    scored_examples: pd.DataFrame,
    test_examples: pd.DataFrame,
    dim_item: pd.DataFrame,
    purchases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join target price and same-session purchase flags onto scored examples."""
    if scored_examples.empty:
        return scored_examples.copy()

    examples_cols = [
        column
        for column in ["example_id", "session_id", "target_item_id_raw", "target_item_id_internal"]
        if column in test_examples.columns
    ]
    out = scored_examples.merge(test_examples[examples_cols], on="example_id", how="left")

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

    if purchases is not None and not purchases.empty and {"session_id", "item_id"}.issubset(purchases.columns):
        purchase_pairs = purchases[["session_id", "item_id"]].drop_duplicates().rename(
            columns={"item_id": "target_item_id_raw"}
        )
        purchase_pairs["target_purchased_in_session"] = True
        out = out.merge(purchase_pairs, on=["session_id", "target_item_id_raw"], how="left")
        out["target_purchased_in_session"] = out["target_purchased_in_session"].fillna(False)
    else:
        out["target_purchased_in_session"] = False

    return out


def compute_model_proxy_kpis(
    model_key: str,
    scored_examples: pd.DataFrame,
    k: int = 20,
    session_summary: pd.DataFrame | None = None,
    baseline_values: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Compute session-centered offline proxy KPIs for one model/baseline."""
    rows: list[dict[str, object]] = []
    scope = "offline_test_examples"
    hit_rate = float(scored_examples["hit_at_k"].mean()) if "hit_at_k" in scored_examples.columns else None
    rows.append(_kpi_row(model_key, k, f"Recommendation Success Rate@{k}", hit_rate, scope))
    rows.append(
        _kpi_row(
            model_key,
            k,
            f"CTR Proxy@{k}",
            ctr_proxy_from_hit_rate(hit_rate) if hit_rate is not None else None,
            scope,
            "CTR Proxy@K equals offline Hit Rate@K; it is not real CTR.",
        )
    )

    if session_summary is not None:
        rows.append(
            _kpi_row(
                model_key,
                None,
                "Purchase Session Rate",
                purchase_session_rate(session_summary),
                "all_sessions",
                "Offline purchase/session proxy; not causal conversion after recommendation.",
            )
        )

    if {"session_id", "hit_at_k"}.issubset(scored_examples.columns) and session_summary is not None:
        purchase_sessions = session_summary[["session_id", "has_purchase"]].copy()
        purchase_scored = scored_examples.merge(purchase_sessions, on="session_id", how="left")
        purchase_scored = purchase_scored[purchase_scored["has_purchase"].fillna(False)]
        value = float(purchase_scored["hit_at_k"].mean()) if not purchase_scored.empty else None
        rows.append(_kpi_row(model_key, k, f"Hit Rate@{k} among purchase sessions", value, scope))

    if {"target_price_proxy", "hit_at_k"}.issubset(scored_examples.columns):
        priced = scored_examples.dropna(subset=["target_price_proxy"]).copy()
        captured_gmv = float((priced["target_price_proxy"] * priced["hit_at_k"]).sum()) if not priced.empty else None
        rows.append(_kpi_row(model_key, k, f"Captured GMV Proxy@{k}", captured_gmv, scope))
        rows.append(
            _kpi_row(
                model_key,
                k,
                f"Revenue-weighted HR@{k}",
                revenue_weighted_hit_rate(priced),
                scope,
            )
        )
        if "target_purchased_in_session" in priced.columns:
            captured_purchase = priced[priced["target_purchased_in_session"].fillna(False)]
        else:
            captured_purchase = priced.iloc[0:0]
        purchase_value = (
            float((captured_purchase["target_price_proxy"] * captured_purchase["hit_at_k"]).sum())
            if not captured_purchase.empty
            else 0.0
        )
        rows.append(
            _kpi_row(
                model_key,
                k,
                f"Captured Purchase Value Proxy@{k}",
                purchase_value,
                scope,
                "Only counts target items also purchased in the same session; not causal conversion.",
            )
        )

    if baseline_values:
        if hit_rate is not None and baseline_values.get("hr") not in (None, 0):
            baseline_hr = float(baseline_values["hr"])
            rows.append(
                _kpi_row(
                    model_key,
                    k,
                    f"Relative delta vs popularity baseline HR@{k}",
                    (hit_rate - baseline_hr) / baseline_hr,
                    scope,
                )
            )
        rw_hr = revenue_weighted_hit_rate(scored_examples)
        baseline_rw_hr = baseline_values.get("revenue_weighted_hr")
        if rw_hr is not None and baseline_rw_hr not in (None, 0):
            rows.append(
                _kpi_row(
                    model_key,
                    k,
                    f"Relative delta vs popularity baseline Revenue-weighted HR@{k}",
                    (rw_hr - float(baseline_rw_hr)) / float(baseline_rw_hr),
                    scope,
                )
            )

    return pd.DataFrame(rows)


def native_metric_proxy_kpis(
    model_key: str,
    hr_at_k: float,
    k: int = 20,
    baseline_hr: float | None = None,
) -> pd.DataFrame:
    """Build KPI rows available when only aggregate native HR@K is inherited."""
    rows = [
        _kpi_row(model_key, k, f"Recommendation Success Rate@{k}", float(hr_at_k), "offline_test_examples"),
        _kpi_row(
            model_key,
            k,
            f"CTR Proxy@{k}",
            ctr_proxy_from_hit_rate(hr_at_k),
            "offline_test_examples",
            "CTR Proxy@K equals offline Hit Rate@K; it is not real CTR.",
        ),
    ]
    if baseline_hr not in (None, 0):
        rows.append(
            _kpi_row(
                model_key,
                k,
                f"Relative delta vs popularity baseline HR@{k}",
                (float(hr_at_k) - float(baseline_hr)) / float(baseline_hr),
                "offline_test_examples",
            )
        )
    return pd.DataFrame(rows)
