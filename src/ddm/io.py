"""Input/output helpers for local DDM analytics files.

These helpers are intentionally small. The DDM repo should consume exported
tables and metadata from the SR-GNN backbone, not import backbone training or
serving code.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "configs/project_config.yaml") -> dict[str, Any]:
    """Load the project YAML config as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def load_project_config(path: str | Path = "configs/project_config.yaml") -> dict[str, Any]:
    """Backward-compatible alias for `load_config`."""
    return load_config(path)


def _detect_csv_separator(path: Path) -> str:
    """Pick the likely CSV separator from the header line."""
    with path.open("r", encoding="utf-8") as file:
        header = file.readline()
    return ";" if header.count(";") > header.count(",") else ","


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV, parquet, or JSON records file into a DataFrame."""
    source = Path(path)
    if source.suffix == ".csv":
        return pd.read_csv(
            source,
            sep=_detect_csv_separator(source),
            na_values=["NA"],
            keep_default_na=True,
            low_memory=False,
        )
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".json":
        return pd.read_json(source)
    raise ValueError(f"Unsupported table format: {source.suffix}")


def load_raw_tables(
    config: Mapping[str, Any] | str | Path = "configs/project_config.yaml",
    project_root: str | Path = ".",
    table_names: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load raw tables declared in `raw_tables` from the project config."""
    payload = load_config(config) if isinstance(config, str | Path) else dict(config)
    raw_tables = payload.get("raw_tables", {})
    if not isinstance(raw_tables, Mapping):
        raise ValueError("Expected `raw_tables` mapping in project config")

    root = Path(project_root)
    tables: dict[str, pd.DataFrame] = {}
    selected_names = list(table_names) if table_names is not None else list(raw_tables)
    for table_name in selected_names:
        if table_name not in raw_tables:
            raise KeyError(f"Raw table `{table_name}` is not declared in project config")
        table_config = raw_tables[table_name]
        if not isinstance(table_config, Mapping) or "path" not in table_config:
            raise ValueError(f"Raw table `{table_name}` must define a path")
        table_path = Path(str(table_config["path"]))
        if not table_path.is_absolute():
            table_path = root / table_path
        tables[str(table_name)] = read_table(table_path)
    return tables


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save a DataFrame to parquet, creating parent directories when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False, **kwargs)
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def missing_paths(paths: dict[str, str]) -> list[str]:
    """Return config keys whose file paths do not exist yet."""
    return [key for key, value in paths.items() if value and not Path(value).exists()]
