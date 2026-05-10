"""Simple train-only baselines for offline comparison.

Baselines must be computed from train-side context only. Do not use test
targets or full raw-data popularity when comparing against SR-GNN predictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import pandas as pd


def _item_mappings(item_vocab: Mapping[str, Any] | None) -> tuple[dict[int, int], dict[int, int]]:
    if not item_vocab:
        return {}, {}
    raw_to_internal = {int(k): int(v) for k, v in item_vocab.get("item2id", {}).items()}
    if "id2item" in item_vocab:
        internal_to_raw = {int(k): int(v) for k, v in item_vocab["id2item"].items()}
    else:
        internal_to_raw = {internal: raw for raw, internal in raw_to_internal.items()}
    return raw_to_internal, internal_to_raw


def _first_existing(columns: pd.Index, candidates: list[str]) -> str:
    for column in candidates:
        if column in columns:
            return column
    raise ValueError(f"Expected one of these columns: {candidates}")


def _with_example_columns(test_examples: pd.DataFrame) -> pd.DataFrame:
    if "example_id" not in test_examples.columns:
        raise ValueError("test_examples must contain example_id.")
    columns = ["example_id"]
    if "session_id" in test_examples.columns:
        columns.append("session_id")
    return test_examples[columns].copy()


def _expand_recommendation_lists(
    examples: pd.DataFrame,
    recommendation_lists: list[list[dict[str, float | int | None]]],
    model_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    has_session = "session_id" in examples.columns
    for example, recs in zip(examples.itertuples(index=False), recommendation_lists, strict=False):
        example_id = getattr(example, "example_id")
        session_id = getattr(example, "session_id", None) if has_session else None
        for rank, rec in enumerate(recs, start=1):
            rows.append(
                {
                    "model_key": model_key,
                    "example_id": example_id,
                    "session_id": session_id,
                    "rank": rank,
                    "pred_item_id_internal": rec.get("pred_item_id_internal"),
                    "pred_item_id_raw": rec.get("pred_item_id_raw"),
                    "score": rec.get("score"),
                }
            )
    return pd.DataFrame(rows)


def _top_popularity_items(
    train_interactions: pd.DataFrame,
    item_vocab: Mapping[str, Any] | None = None,
    k: int = 20,
) -> list[dict[str, float | int | None]]:
    item_col = _first_existing(train_interactions.columns, ["item_id", "itemId", "pos_items"])
    raw_to_internal, internal_to_raw = _item_mappings(item_vocab)
    ids_are_internal = item_col == "pos_items"

    counts = train_interactions[item_col].dropna().astype("int64").value_counts()
    rows: list[dict[str, float | int | None]] = []
    for item_id, count in counts.items():
        item = int(item_id)
        if ids_are_internal:
            internal_id = item
            raw_id = internal_to_raw.get(internal_id, item) if internal_to_raw else item
        elif raw_to_internal:
            raw_id = item
            internal_id = raw_to_internal.get(raw_id)
            if internal_id is None:
                continue
        else:
            internal_id = item
            raw_id = internal_to_raw.get(internal_id, item) if internal_to_raw else item
        rows.append(
            {
                "pred_item_id_internal": int(internal_id),
                "pred_item_id_raw": int(raw_id) if raw_id is not None else None,
                "score": float(count),
            }
        )
        if len(rows) >= k:
            break
    return rows


def build_popularity_baseline(
    train_interactions: pd.DataFrame,
    test_examples: pd.DataFrame,
    item_vocab: Mapping[str, Any] | None = None,
    k: int = 20,
    model_key: str = "popularity_top20",
) -> pd.DataFrame:
    """Build same-for-every-example top-k popularity predictions from train data."""
    examples = _with_example_columns(test_examples)
    top_items = _top_popularity_items(train_interactions, item_vocab=item_vocab, k=k)
    recommendation_lists = [top_items] * len(examples)
    return _expand_recommendation_lists(examples, recommendation_lists, model_key=model_key)


def _session_sequences(train_interactions: pd.DataFrame) -> pd.Series:
    if {"x", "alias_inputs", "pos_items"}.issubset(train_interactions.columns):
        return train_interactions.apply(_graph_example_sequence, axis=1)
    session_col = _first_existing(train_interactions.columns, ["session_id", "sessionId"])
    item_col = _first_existing(train_interactions.columns, ["item_id", "itemId", "pos_items"])
    sort_cols = [session_col]
    for candidate in ["event_date", "eventdate", "timeframe"]:
        if candidate in train_interactions.columns:
            sort_cols.append(candidate)
    ordered = train_interactions.sort_values(sort_cols)
    return ordered.groupby(session_col, sort=False)[item_col].apply(list)


def _graph_example_sequence(row: pd.Series) -> list[int]:
    """Return an internal-ID prefix + target sequence from a graph example row."""
    x = list(row["x"])
    alias_inputs = list(row["alias_inputs"])
    sequence = [int(x[int(alias)]) for alias in alias_inputs]
    if "pos_items" in row and pd.notna(row["pos_items"]):
        sequence.append(int(row["pos_items"]))
    return sequence


def _last_internal_item(row: pd.Series) -> int | None:
    if "last_item_id_internal" in row and pd.notna(row["last_item_id_internal"]):
        return int(row["last_item_id_internal"])
    if "prefix_item_ids_internal" in row and isinstance(row["prefix_item_ids_internal"], list):
        return int(row["prefix_item_ids_internal"][-1]) if row["prefix_item_ids_internal"] else None
    if {"x", "alias_inputs"}.issubset(row.index):
        x = list(row["x"])
        alias_inputs = list(row["alias_inputs"])
        if x and alias_inputs:
            return int(x[int(alias_inputs[-1])])
    return None


def build_cooccurrence_baseline(
    train_interactions: pd.DataFrame,
    test_examples: pd.DataFrame,
    item_vocab: Mapping[str, Any] | None = None,
    k: int = 20,
    model_key: str = "cooccurrence_top20",
) -> pd.DataFrame:
    """Build next-item transition baseline using train-side session sequences."""
    examples = _with_example_columns(test_examples)
    raw_to_internal, internal_to_raw = _item_mappings(item_vocab)
    popularity = _top_popularity_items(train_interactions, item_vocab=item_vocab, k=max(k, 100))

    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    ids_are_internal = "pos_items" in train_interactions.columns
    for sequence in _session_sequences(train_interactions):
        internal_sequence: list[int] = []
        for item in sequence:
            if pd.isna(item):
                continue
            item_int = int(item)
            internal = item_int if ids_are_internal else raw_to_internal.get(item_int, item_int)
            internal_sequence.append(int(internal))
        for current, nxt in zip(internal_sequence, internal_sequence[1:], strict=False):
            transition_counts[current][nxt] += 1

    top_by_last: dict[int, list[dict[str, float | int | None]]] = {}
    for last_item, counts in transition_counts.items():
        recs: list[dict[str, float | int | None]] = []
        seen: set[int] = set()
        for candidate, count in counts.most_common():
            if candidate == last_item or candidate in seen:
                continue
            seen.add(candidate)
            recs.append(
                {
                    "pred_item_id_internal": int(candidate),
                    "pred_item_id_raw": int(internal_to_raw.get(candidate, candidate)),
                    "score": float(count),
                }
            )
            if len(recs) >= k:
                break
        for fallback in popularity:
            candidate = int(fallback["pred_item_id_internal"])
            if candidate == last_item or candidate in seen:
                continue
            seen.add(candidate)
            recs.append(fallback)
            if len(recs) >= k:
                break
        top_by_last[last_item] = recs[:k]

    recommendation_lists: list[list[dict[str, float | int | None]]] = []
    for _, row in test_examples.iterrows():
        last_item = _last_internal_item(row)
        recommendation_lists.append(top_by_last.get(last_item, popularity[:k]))

    return _expand_recommendation_lists(examples, recommendation_lists, model_key=model_key)


def validate_no_leakage(
    train_interactions: pd.DataFrame,
    test_examples: pd.DataFrame,
    train_date_col: str = "eventdate",
    test_date_col: str = "eventdate",
) -> bool:
    """Return True when train rows end before test examples begin, if dates exist."""
    if train_date_col not in train_interactions.columns or test_date_col not in test_examples.columns:
        return True
    train_max = pd.to_datetime(train_interactions[train_date_col], errors="coerce").max()
    test_min = pd.to_datetime(test_examples[test_date_col], errors="coerce").min()
    if pd.isna(train_max) or pd.isna(test_min):
        return True
    return bool(train_max <= test_min)
