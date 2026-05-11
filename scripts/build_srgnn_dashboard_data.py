"""Build helper datasets for the SR-GNN Model Evaluation Dashboard.

Outputs are written to dashboards/powerbi_data/ and can be loaded directly in
Power BI as report-local tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MART_ROOT = REPO_ROOT / "data" / "mart"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "diginetica"
OUT_ROOT = REPO_ROOT / "dashboards" / "powerbi_data"


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=";")


def _norm_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {"sessionId": "session_id", "itemId": "item_id", "eventdate": "event_date"}
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def _session_length_bucket(view_count: pd.Series) -> pd.Series:
    val = pd.to_numeric(view_count, errors="coerce").fillna(0)
    bins = [-np.inf, 0, 3, 5, 9, 20, np.inf]
    labels = ["0", "1–3", "4–5", "6–9", "10–20", "20+"]
    return pd.cut(val, bins=bins, labels=labels).astype("string").fillna("0")


def _session_view_bucket_for_conversion(view_count: pd.Series) -> pd.Series:
    val = pd.to_numeric(view_count, errors="coerce").fillna(0).astype("int64")
    out = pd.Series("16–20", index=val.index, dtype="string")
    out[val <= 11] = val[val <= 11].astype(str)
    out[(val > 11) & (val <= 15)] = "11–15"
    out[(val > 15) & (val <= 20)] = "16–20"
    out[val <= 0] = "0"
    return out


def _segment(view_count: pd.Series, has_purchase: pd.Series) -> pd.Series:
    vc = pd.to_numeric(view_count, errors="coerce").fillna(0)
    hp = has_purchase.fillna(False).astype(bool)
    seg = pd.Series("Bouncer", index=vc.index, dtype="string")
    seg[(vc >= 5) & hp] = "High-Intent Buyer"
    seg[(vc >= 5) & (~hp)] = "Browser"
    seg[(vc < 5) & hp] = "Quick Buyer"
    return seg


def _source_timeframe_label(views: pd.DataFrame, cutoff: int | None) -> pd.Series | None:
    if "timeframe" not in views.columns:
        return None
    tf = pd.to_numeric(views["timeframe"], errors="coerce")
    if tf.isna().all():
        return None
    threshold = cutoff if cutoff is not None else int(tf.median())
    lbl = np.where(tf <= threshold, "Trước (gốc)", "Sau (có model)")
    return pd.Series(lbl, index=views.index, dtype="string")


def _fallback_timeframe_label_from_eval(
    views: pd.DataFrame, eval_df: pd.DataFrame
) -> pd.Series:
    """Fallback: sessions in recommendation eval become 'after', else 'before'."""
    if views.empty or eval_df.empty or "session_id" not in views.columns:
        return pd.Series("Trước (gốc)", index=views.index, dtype="string")
    eval_sessions = set(pd.to_numeric(eval_df.get("session_id"), errors="coerce").dropna().astype("int64"))
    sess = pd.to_numeric(views["session_id"], errors="coerce")
    lbl = np.where(sess.isin(eval_sessions), "Sau (có model)", "Trước (gốc)")
    return pd.Series(lbl, index=views.index, dtype="string")


def build_session_summary(
    item_views: pd.DataFrame,
    purchases: pd.DataFrame,
    eval_df: pd.DataFrame,
    timeframe_cutoff: int | None,
) -> pd.DataFrame:
    if item_views.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "timeframe",
                "view_count",
                "unique_items_viewed",
                "has_purchase",
                "purchase_count",
                "quantity_sum",
                "session_length_bucket",
                "session_view_bucket_for_conversion",
                "segment",
            ]
        )

    views = _norm_ids(item_views)
    buys = _norm_ids(purchases)
    views["session_id"] = pd.to_numeric(views["session_id"], errors="coerce").astype("Int64")
    views["item_id"] = pd.to_numeric(views["item_id"], errors="coerce").astype("Int64")
    views = views.dropna(subset=["session_id", "item_id"]).copy()

    tf_label = _source_timeframe_label(views, timeframe_cutoff)
    if tf_label is None:
        tf_label = _fallback_timeframe_label_from_eval(views, eval_df)
    views["timeframe_label"] = tf_label

    session_views = (
        views.groupby("session_id", as_index=False)
        .agg(
            timeframe=("timeframe_label", lambda s: s.mode().iloc[0] if not s.mode().empty else "Trước (gốc)"),
            view_count=("item_id", "size"),
            unique_items_viewed=("item_id", "nunique"),
        )
        .copy()
    )

    if buys.empty:
        session_buys = pd.DataFrame(
            {
                "session_id": session_views["session_id"],
                "has_purchase": False,
                "purchase_count": 0,
                "quantity_sum": 0,
            }
        )
    else:
        buys["session_id"] = pd.to_numeric(buys["session_id"], errors="coerce").astype("Int64")
        buys["item_id"] = pd.to_numeric(buys["item_id"], errors="coerce").astype("Int64")
        if "quantity" in buys.columns:
            buys["quantity"] = pd.to_numeric(buys["quantity"], errors="coerce").fillna(1.0)
        else:
            buys["quantity"] = 1.0
        buys = buys.dropna(subset=["session_id", "item_id"]).copy()
        session_buys = (
            buys.groupby("session_id", as_index=False)
            .agg(purchase_count=("item_id", "size"), quantity_sum=("quantity", "sum"))
            .copy()
        )
        session_buys["has_purchase"] = session_buys["purchase_count"] > 0

    out = session_views.merge(session_buys, on="session_id", how="left")
    out["has_purchase"] = out["has_purchase"].fillna(False).astype(bool)
    out["purchase_count"] = pd.to_numeric(out["purchase_count"], errors="coerce").fillna(0).astype("int64")
    out["quantity_sum"] = pd.to_numeric(out["quantity_sum"], errors="coerce").fillna(0.0)
    out["session_length_bucket"] = _session_length_bucket(out["view_count"])
    out["session_view_bucket_for_conversion"] = _session_view_bucket_for_conversion(out["view_count"])
    out["segment"] = _segment(out["view_count"], out["has_purchase"])
    return out.sort_values("session_id").reset_index(drop=True)


def build_model_metrics_summary(
    dim_model: pd.DataFrame,
    fact_metrics: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    if fact_metrics.empty and eval_df.empty:
        return pd.DataFrame(columns=["model_key", "model_label", "HR@20", "MRR@20", "Coverage@20"])

    mm = fact_metrics.copy()
    if not mm.empty:
        mm = mm.rename(columns={"metric_name": "metric_name_raw"})
        mm["metric_name"] = (
            mm["metric_name_raw"]
            .astype(str)
            .str.replace("Catalog Coverage@20", "Coverage@20", regex=False)
        )
        mm = mm[mm["metric_name"].isin(["HR@20", "MRR@20", "Coverage@20"])]
        pvt = (
            mm.groupby(["model_key", "metric_name"], as_index=False)["metric_value"]
            .mean()
            .pivot(index="model_key", columns="metric_name", values="metric_value")
            .reset_index()
        )
    else:
        pvt = pd.DataFrame(columns=["model_key", "HR@20", "MRR@20", "Coverage@20"])

    if not eval_df.empty:
        ev = eval_df.copy()
        rec_item_col = "target_item_id_raw" if "target_item_id_raw" in ev.columns else "item_id"
        grp = ev.groupby("model_key", as_index=False).agg(
            hr=("hit_at_k", "mean"),
            mrr=("reciprocal_rank", "mean"),
            rec_items=(rec_item_col, "nunique"),
        )
        denom = float(pd.to_numeric(ev.get(rec_item_col), errors="coerce").dropna().nunique() or 1.0)
        grp["Coverage@20_derived"] = grp["rec_items"] / denom
        grp = grp.rename(columns={"hr": "HR@20_derived", "mrr": "MRR@20_derived"})
        pvt = pvt.merge(
            grp[["model_key", "HR@20_derived", "MRR@20_derived", "Coverage@20_derived"]],
            on="model_key",
            how="outer",
        )

    for m in ["HR@20", "MRR@20", "Coverage@20"]:
        derived = f"{m}_derived"
        if derived in pvt.columns:
            pvt[m] = pvt[m].combine_first(pvt[derived]) if m in pvt.columns else pvt[derived]

    labels = dim_model.copy()
    if "model_label" not in labels.columns:
        labels["model_label"] = labels.get("model_key", "").astype(str)
    out = pvt.merge(labels[["model_key", "model_label"]], on="model_key", how="left")
    for col in ["HR@20", "MRR@20", "Coverage@20"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[["model_key", "model_label", "HR@20", "MRR@20", "Coverage@20"]].sort_values("model_key")


def build_data_quality_summary(
    eval_df: pd.DataFrame, dim_item: pd.DataFrame, item_views: pd.DataFrame, purchases: pd.DataFrame
) -> pd.DataFrame:
    iv = _norm_ids(item_views)
    pr = _norm_ids(purchases)
    min_date = pd.to_datetime(iv.get("event_date"), errors="coerce").min()
    max_date = pd.to_datetime(iv.get("event_date"), errors="coerce").max()

    rec_item_col = "target_item_id_raw" if "target_item_id_raw" in eval_df.columns else "item_id"
    rows = [
        ("total_sessions", int(pd.to_numeric(iv.get("session_id"), errors="coerce").dropna().nunique())),
        ("total_items", int(pd.to_numeric(dim_item.get("item_id"), errors="coerce").dropna().nunique())),
        ("total_purchases", int(len(pr))),
        ("date_range_start", "" if pd.isna(min_date) else str(min_date.date())),
        ("date_range_end", "" if pd.isna(max_date) else str(max_date.date())),
        ("number_of_models", int(eval_df["model_key"].dropna().astype(str).nunique() if "model_key" in eval_df.columns else 0)),
        ("missing_session_id_count", int(eval_df["session_id"].isna().sum() if "session_id" in eval_df.columns else 0)),
        ("missing_item_id_count", int(eval_df[rec_item_col].isna().sum() if rec_item_col in eval_df.columns else 0)),
        ("missing_model_key_count", int(eval_df["model_key"].isna().sum() if "model_key" in eval_df.columns else 0)),
        ("missing_price_proxy_count", int(dim_item["price_proxy"].isna().sum() if "price_proxy" in dim_item.columns else 0)),
    ]
    return pd.DataFrame(rows, columns=["metric_name", "metric_value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SR-GNN dashboard helper datasets.")
    parser.add_argument(
        "--timeframe-cutoff",
        type=int,
        default=None,
        help="Optional numeric cutoff for raw timeframe split (<= cutoff: before, > cutoff: after).",
    )
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    eval_df = _read_parquet(MART_ROOT / "fact_recommendation_eval.parquet")
    fact_metrics = _read_parquet(MART_ROOT / "fact_metrics.parquet")
    dim_model = _read_parquet(MART_ROOT / "dim_model.parquet")
    dim_item = _read_parquet(MART_ROOT / "dim_item.parquet")
    item_views = _read_csv(RAW_ROOT / "train-item-views.csv")
    purchases = _read_csv(RAW_ROOT / "train-purchases.csv")

    session_summary = build_session_summary(item_views, purchases, eval_df, args.timeframe_cutoff)
    model_metrics_summary = build_model_metrics_summary(dim_model, fact_metrics, eval_df)
    data_quality_summary = build_data_quality_summary(eval_df, dim_item, item_views, purchases)

    session_summary.to_csv(OUT_ROOT / "pbi_session_summary.csv", index=False)
    model_metrics_summary.to_csv(OUT_ROOT / "pbi_model_metrics_summary.csv", index=False)
    data_quality_summary.to_csv(OUT_ROOT / "pbi_data_quality_summary.csv", index=False)

    print(f"Wrote: {OUT_ROOT / 'pbi_session_summary.csv'}")
    print(f"Wrote: {OUT_ROOT / 'pbi_model_metrics_summary.csv'}")
    print(f"Wrote: {OUT_ROOT / 'pbi_data_quality_summary.csv'}")


if __name__ == "__main__":
    main()
