"""Artifact inheritance for the DDM reporting repo.

The DDM project consumes a fixed trained recommender from MLflow/DagsHub and may
also inherit compatible intermediate evaluation artifacts from
``recsys-group-project``. It does not train, select, promote, deploy, or monitor
models.
"""

from __future__ import annotations

import os
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ddm.io import read_json


REQUIRED_BUNDLE_FILES = {
    "manifest.json",
    "test_examples.parquet",
    "predictions.parquet",
    "item_vocab.json",
    "metrics.json",
}
MODEL_CARD_CANDIDATES = ("model_card.json", "model_card.md")
OPTIONAL_BUNDLE_FILES = {
    "item_mapping.parquet",
    "baseline_metrics.json",
    "item_metadata.parquet",
}
REQUIRED_MANIFEST_FIELDS = {
    "model_name",
    "run_id",
    "artifact_uri",
    "data_version",
    "model_profile",
    "top_k",
    "export_timestamp",
    "source_repo",
}


@dataclass(frozen=True)
class RegistryReference:
    """Resolved MLflow registry reference without secret-bearing fields."""

    model_name: str
    model_version: str
    model_alias: str
    run_id: str
    artifact_uri: str | None = None


def registry_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the registry config mapping with defaults."""
    loaded = config.get("registry", {})
    if not isinstance(loaded, dict):
        raise ValueError("`registry` config must be a mapping.")
    return {
        "tracking_uri": "",
        "dagshub_repo_owner": "",
        "dagshub_repo_name": "",
        "model_name": "recsys-serving",
        "model_alias": "Production",
        "model_version": "",
        "artifact_path": "registered_model",
        **loaded,
    }


def inheritance_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return inheritance config with defaults."""
    loaded = config.get("inheritance", {})
    if not isinstance(loaded, dict):
        raise ValueError("`inheritance` config must be a mapping.")
    return {
        "top_k": 20,
        "inherited_root": "data/inherited/recsys",
        "context_repo_path": "",
        "context_artifact_source": "",
        "test_examples_path": "",
        "train_examples_path": "",
        "item_vocab_path": "",
        **loaded,
    }


def tracking_uri_from_config(config: dict[str, Any]) -> str:
    """Resolve the MLflow tracking URI without reading or exposing secrets."""
    cfg = registry_config(config)
    if cfg.get("tracking_uri"):
        return str(cfg["tracking_uri"])
    owner = str(cfg.get("dagshub_repo_owner") or "").strip()
    repo = str(cfg.get("dagshub_repo_name") or "").strip()
    if not owner or not repo:
        raise ValueError(
            "Set registry.tracking_uri or both registry.dagshub_repo_owner "
            "and registry.dagshub_repo_name."
        )
    return f"https://dagshub.com/{owner}/{repo}.mlflow"


def bundle_directory(root: Path, config: dict[str, Any]) -> Path:
    """Return the local inherited bundle directory for this registry reference."""
    registry = registry_config(config)
    inheritance = inheritance_config(config)
    name = _slug(str(registry["model_name"]))
    version_or_alias = str(registry.get("model_version") or registry.get("model_alias") or "unversioned")
    version_or_alias = _slug(version_or_alias)
    top_k = int(inheritance["top_k"])
    base = Path(str(inheritance["inherited_root"]))
    if not base.is_absolute():
        base = root / base
    return base / f"{name}_{version_or_alias}_top{top_k}"


def resolve_registry_reference(config: dict[str, Any]) -> RegistryReference:
    """Resolve the configured MLflow registered model alias or version."""
    cfg = registry_config(config)
    mlflow = _configure_tracking(config)
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    model_name = str(cfg["model_name"])
    pinned_version = str(cfg.get("model_version") or "").strip()
    alias = str(cfg.get("model_alias") or "").strip()
    try:
        if pinned_version:
            version = client.get_model_version(model_name, pinned_version)
            resolved_alias = alias
        else:
            if not alias:
                raise ValueError("Set registry.model_alias or registry.model_version.")
            version = client.get_model_version_by_alias(model_name, alias)
            resolved_alias = alias
    except Exception as exc:
        message = str(exc)
        if "401" in message or "authentication" in message.lower():
            raise RuntimeError(
                "DagsHub/MLflow authentication failed while resolving the registered model. "
                "Check that `.env` contains a valid DAGSHUB_USERNAME and DAGSHUB_USER_TOKEN "
                "with access to the configured repo, or set MLFLOW_TRACKING_USERNAME and "
                "MLFLOW_TRACKING_PASSWORD explicitly."
            ) from exc
        raise

    return RegistryReference(
        model_name=model_name,
        model_version=str(version.version),
        model_alias=resolved_alias,
        run_id=str(version.run_id),
        artifact_uri=getattr(version, "source", None),
    )


def download_registry_bundle(root: Path, config: dict[str, Any]) -> Path:
    """Download the trained model artifact, run final inference, and validate."""
    cfg = registry_config(config)
    reference = resolve_registry_reference(config)
    destination = bundle_directory(root, config)
    tmp_destination = destination.with_name(f".{destination.name}.tmp")
    if tmp_destination.exists():
        shutil.rmtree(tmp_destination)
    tmp_destination.mkdir(parents=True, exist_ok=True)

    mlflow = _configure_tracking(config)
    try:
        downloaded = Path(mlflow.artifacts.download_artifacts(
            run_id=reference.run_id,
            artifact_path=str(cfg["artifact_path"]),
            dst_path=str(tmp_destination),
        ))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download MLflow artifact path `{cfg['artifact_path']}` "
            f"for {reference.model_name} version {reference.model_version}."
        ) from exc

    model_artifact = _single_downloaded_directory(downloaded, str(cfg["artifact_path"]))
    prepared_root = tmp_destination / "ddm_bundle"
    _prepare_inference_bundle(
        root=root,
        config=config,
        reference=reference,
        model_artifact=model_artifact,
        destination=prepared_root,
    )
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(prepared_root), str(destination))
    if tmp_destination.exists():
        shutil.rmtree(tmp_destination)

    validate_inherited_bundle(destination, config)
    return destination


def validate_inherited_bundle(bundle_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the inherited bundle and return its manifest."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inherited registry bundle not found at {path}. Run `make inherit` "
            "after configuring DagsHub/MLflow credentials."
        )

    missing = sorted(file for file in REQUIRED_BUNDLE_FILES if not (path / file).exists())
    if not any((path / file).exists() for file in MODEL_CARD_CANDIDATES):
        missing.append("model_card.json or model_card.md")
    if missing:
        raise FileNotFoundError(f"Inherited registry bundle is missing required files: {missing}")

    manifest = read_json(path / "manifest.json")
    missing_fields = sorted(field for field in REQUIRED_MANIFEST_FIELDS if not manifest.get(field))
    registry = registry_config(config)
    if not (manifest.get("model_alias") or manifest.get("model_version")):
        missing_fields.append("model_alias or model_version")
    if missing_fields:
        raise ValueError(f"manifest.json is missing required fields: {missing_fields}")

    expected_model_name = str(registry["model_name"])
    if str(manifest["model_name"]) != expected_model_name:
        raise ValueError(
            f"manifest model_name `{manifest['model_name']}` does not match config `{expected_model_name}`."
        )

    expected_top_k = int(inheritance_config(config)["top_k"])
    if int(manifest["top_k"]) != expected_top_k:
        raise ValueError(f"manifest top_k {manifest['top_k']} does not match config top_k {expected_top_k}.")

    examples = pd.read_parquet(path / "test_examples.parquet")
    predictions = pd.read_parquet(path / "predictions.parquet")
    _validate_example_schema(examples)
    _validate_prediction_schema(predictions, examples, expected_top_k)
    _validate_item_mapping(path, examples, predictions)
    return manifest


def bundle_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    """Return expected local inherited bundle paths."""
    bundle = bundle_directory(root, config)
    return {
        "bundle": bundle,
        "manifest": bundle / "manifest.json",
        "test_examples": bundle / "test_examples.parquet",
        "predictions": bundle / "predictions.parquet",
        "item_vocab": bundle / "item_vocab.json",
        "metrics": bundle / "metrics.json",
        "model_artifact": bundle / "model_artifact",
        "baseline_metrics": bundle / "baseline_metrics.json",
        "item_metadata": bundle / "item_metadata.parquet",
    }


def _prepare_inference_bundle(
    *,
    root: Path,
    config: dict[str, Any],
    reference: RegistryReference,
    model_artifact: Path,
    destination: Path,
) -> None:
    """Create the DDM bundle from a downloaded trained model artifact."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    artifact_target = destination / "model_artifact"
    shutil.copytree(model_artifact, artifact_target)
    artifact_config = read_json(artifact_target / "config.json")
    model_metadata = read_json(artifact_target / "model.json")
    source_repo = _context_repo_path(root, config)
    data_cfg = artifact_config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Model artifact config.json must contain a data mapping.")

    # Prioritize files already inside the downloaded bundle (Option 1: Data Inheritance)
    bundle_test_examples = artifact_target / "test_examples.parquet"
    bundle_predictions = artifact_target / "predictions.parquet"
    bundle_item_vocab = artifact_target / "item_vocab.json"

    train_examples_source = _intermediate_artifact_path(config, source_repo, data_cfg, "train_examples_path")
    test_examples_source = bundle_test_examples if bundle_test_examples.exists() else _intermediate_artifact_path(config, source_repo, data_cfg, "test_examples_path")
    item_vocab_source = bundle_item_vocab if bundle_item_vocab.exists() else _intermediate_artifact_path(config, source_repo, data_cfg, "item_vocab_path")

    if not test_examples_source.exists():
        raise FileNotFoundError(
            f"Missing inherited test examples. Checked artifact bundle and {test_examples_source}. "
            "Ensure the MLOps pipeline exports test_examples.parquet or the context_repo_path is correct."
        )

    test_examples = _normalise_test_examples_for_bundle(
        pd.read_parquet(test_examples_source),
        model_metadata,
    )
    test_examples_path = destination / "test_examples.parquet"
    test_examples.to_parquet(test_examples_path, index=False)

    if item_vocab_source.exists():
        shutil.copy2(item_vocab_source, destination / "item_vocab.json")
    else:
        (destination / "item_vocab.json").write_text(
            json.dumps(_item_vocab_from_model_metadata(model_metadata), indent=2),
            encoding="utf-8",
        )

    if (artifact_target / "metrics.json").exists():
        shutil.copy2(artifact_target / "metrics.json", destination / "metrics.json")
    else:
        (destination / "metrics.json").write_text("{}", encoding="utf-8")

    manifest = {
        "model_name": reference.model_name,
        "model_alias": reference.model_alias,
        "model_version": reference.model_version,
        "run_id": reference.run_id,
        "artifact_uri": reference.artifact_uri or f"runs:/{reference.run_id}/{registry_config(config)['artifact_path']}",
        "data_version": str(artifact_config.get("lineage", {}).get("data_version") or _infer_data_version(data_cfg)),
        "model_profile": str(artifact_config.get("model", {}).get("name") or model_metadata.get("model_name", "")),
        "top_k": int(inheritance_config(config)["top_k"]),
        "export_timestamp": datetime.now(UTC).isoformat(),
        "source_repo": str(source_repo),
        "inference_source": "registered_model_artifact",
        "context_artifact_source": str(inheritance_config(config).get("context_artifact_source", "recsys-group-project")),
        "context_artifacts": {
            "train_examples_path": str(train_examples_source) if train_examples_source.exists() else "",
            "test_examples_path": str(test_examples_source),
            "item_vocab_path": str(item_vocab_source) if item_vocab_source.exists() else "",
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "model_card.md").write_text(_model_card_text(manifest, artifact_config), encoding="utf-8")

    if bundle_predictions.exists():
        shutil.copy2(bundle_predictions, destination / "predictions.parquet")
    else:
        predictions = _run_final_inference(
            root=root,
            config=config,
            model_artifact=artifact_target,
            test_examples=test_examples,
            model_metadata=model_metadata,
        )
        predictions.to_parquet(destination / "predictions.parquet", index=False)



def _run_final_inference(
    *,
    root: Path,
    config: dict[str, Any],
    model_artifact: Path,
    test_examples: pd.DataFrame,
    model_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Run top-K inference from inherited trained model artifact."""
    source_repo = _context_repo_path(root, config)
    source_src = source_repo / "src"
    if str(source_src) not in sys.path:
        sys.path.insert(0, str(source_src))
    from recsys.serving.predictor import _load_model_artifact  # type: ignore

    model = _load_model_artifact(model_artifact)
    top_k = int(inheritance_config(config)["top_k"])
    model_key = str(registry_config(config)["model_name"])
    idx_to_item = {int(k): int(v) for k, v in model_metadata.get("idx_to_item", {}).items()}
    item_to_idx = {int(k): int(v) for k, v in model_metadata.get("item_to_idx", {}).items()}

    rows: list[dict[str, object]] = []
    for row in test_examples.itertuples(index=False):
        x_values = list(getattr(row, "x"))
        alias_inputs = list(getattr(row, "alias_inputs"))
        internal_sequence = [int(x_values[int(alias)]) for alias in alias_inputs]
        raw_sequence = [idx_to_item.get(item, item) for item in internal_sequence]
        raw_recs = model.recommend(raw_sequence, top_k=top_k)
        for rank, raw_item in enumerate(raw_recs, start=1):
            raw_item_int = int(raw_item)
            rows.append(
                {
                    "model_key": model_key,
                    "example_id": int(getattr(row, "example_id")),
                    "session_id": int(getattr(row, "session_id")),
                    "rank": rank,
                    "pred_item_id_internal": item_to_idx.get(raw_item_int, pd.NA),
                    "pred_item_id_raw": raw_item_int,
                    "score": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def _normalise_test_examples_for_bundle(
    examples: pd.DataFrame,
    model_metadata: dict[str, Any],
) -> pd.DataFrame:
    out = examples.reset_index(drop=True).copy()
    if "example_id" not in out.columns:
        out.insert(0, "example_id", range(1, len(out) + 1))
    if "target_item_id_internal" not in out.columns and "pos_items" in out.columns:
        out["target_item_id_internal"] = out["pos_items"].astype("int64")
    idx_to_item = {int(k): int(v) for k, v in model_metadata.get("idx_to_item", {}).items()}
    if "target_item_id_raw" not in out.columns and "target_item_id_internal" in out.columns:
        out["target_item_id_raw"] = out["target_item_id_internal"].map(idx_to_item).astype("Int64")
    if "last_item_id_internal" not in out.columns and {"x", "alias_inputs"}.issubset(out.columns):
        out["last_item_id_internal"] = out.apply(_last_item_from_graph_row, axis=1).astype("Int64")
    if "last_item_id_raw" not in out.columns and "last_item_id_internal" in out.columns:
        out["last_item_id_raw"] = out["last_item_id_internal"].map(idx_to_item).astype("Int64")
    if "prefix_item_ids_internal" not in out.columns and {"x", "alias_inputs"}.issubset(out.columns):
        out["prefix_item_ids_internal"] = out.apply(_prefix_items_from_graph_row, axis=1)
    return out


def _prefix_items_from_graph_row(row: pd.Series) -> list[int]:
    x_values = list(row["x"])
    return [int(x_values[int(alias)]) for alias in list(row["alias_inputs"])]


def _last_item_from_graph_row(row: pd.Series) -> int | None:
    items = _prefix_items_from_graph_row(row)
    return items[-1] if items else None


def _item_vocab_from_model_metadata(model_metadata: dict[str, Any]) -> dict[str, Any]:
    item_to_idx = {str(k): int(v) for k, v in model_metadata.get("item_to_idx", {}).items()}
    idx_to_item = {str(k): int(v) for k, v in model_metadata.get("idx_to_item", {}).items()}
    return {"size": len(item_to_idx), "item2id": item_to_idx, "id2item": idx_to_item}


def _context_repo_path(root: Path, config: dict[str, Any]) -> Path:
    path_str = str(inheritance_config(config).get("context_repo_path") or "").strip()
    path = Path(path_str) if path_str else Path(".")
    return path if path.is_absolute() else (root / path).resolve()


def _source_repo_path(source_repo: Path, value: Any) -> Path:
    if not value:
        return source_repo / "__missing__"
    path = Path(str(value))
    return path if path.is_absolute() else source_repo / path


def _intermediate_artifact_path(
    config: dict[str, Any],
    source_repo: Path,
    data_cfg: dict[str, Any],
    key: str,
) -> Path:
    """Resolve an inherited intermediate artifact path.

    Explicit DDM inheritance config wins. If unset, use the trained model's
    saved data config so the evaluation context stays compatible with the
    downloaded model.
    """
    inheritance = inheritance_config(config)
    explicit = str(inheritance.get(key) or "").strip()
    return _source_repo_path(source_repo, explicit or data_cfg.get(key))


def _infer_data_version(data_cfg: dict[str, Any]) -> str:
    for key in ("processed_path", "test_examples_path", "item_vocab_path"):
        value = str(data_cfg.get(key) or "")
        match = re.search(r"(v\d+[\w-]*)", value)
        if match:
            return match.group(1)
    return "unknown"


def _model_card_text(manifest: dict[str, Any], artifact_config: dict[str, Any]) -> str:
    model_cfg = artifact_config.get("model", {})
    return "\n".join(
        [
            "# Inherited Recommendation Model",
            "",
            f"- Model name: `{manifest['model_name']}`",
            f"- Model version: `{manifest['model_version']}`",
            f"- Model alias: `{manifest.get('model_alias', '')}`",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Model profile: `{manifest['model_profile']}`",
            f"- Data version: `{manifest['data_version']}`",
            f"- Top K: `{manifest['top_k']}`",
            f"- Model type: `{model_cfg.get('type', '')}`",
            "",
            "This model is inherited from the MLflow registry and used only for offline DDM reporting inference.",
            "Compatible test examples and vocabulary may be inherited from intermediate recsys-group-project artifacts.",
            "Metrics and marketing KPIs are offline proxies, not real CTR, causal lift, ROAS, or audited revenue.",
        ]
    )


def _validate_example_schema(examples: pd.DataFrame) -> None:
    required = {"example_id", "session_id"}
    if not ({"target_item_id_internal", "target_item_id_raw"} & set(examples.columns)):
        required.add("target_item_id_internal or target_item_id_raw")
    missing = sorted(column for column in required if column not in examples.columns)
    if missing:
        raise ValueError(f"test_examples.parquet is missing required columns: {missing}")
    if examples["example_id"].duplicated().any():
        raise ValueError("test_examples.parquet must contain one row per example_id.")


def _validate_prediction_schema(predictions: pd.DataFrame, examples: pd.DataFrame, top_k: int) -> None:
    required = {"example_id", "rank"}
    if not ({"pred_item_id_internal", "pred_item_id_raw"} & set(predictions.columns)):
        required.add("pred_item_id_internal or pred_item_id_raw")
    missing = sorted(column for column in required if column not in predictions.columns)
    if missing:
        raise ValueError(f"predictions.parquet is missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("predictions.parquet must not be empty.")

    unknown_examples = set(predictions["example_id"].dropna()) - set(examples["example_id"].dropna())
    if unknown_examples:
        raise ValueError("predictions.parquet contains example_id values absent from test_examples.parquet.")

    frame = predictions.copy()
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    if frame["rank"].isna().any() or not frame["rank"].between(1, top_k, inclusive="both").all():
        raise ValueError(f"prediction ranks must be numeric and within 1..{top_k}.")

    group_cols = ["example_id"]
    if "model_key" in frame.columns:
        group_cols.insert(0, "model_key")
    counts = frame.groupby(group_cols).size()
    if not counts.eq(top_k).all():
        raise ValueError(f"predictions.parquet must contain exactly {top_k} rows per model/example.")


def _validate_item_mapping(path: Path, examples: pd.DataFrame, predictions: pd.DataFrame) -> None:
    vocab = read_json(path / "item_vocab.json")
    id2item = vocab.get("id2item")
    if id2item is None and "item2id" in vocab:
        id2item = {str(v): int(k) for k, v in vocab["item2id"].items()}
    if not id2item:
        raise ValueError("item_vocab.json must contain id2item or item2id mapping.")
    mapped_ids = {int(key) for key in id2item}

    for frame_name, frame, column in [
        ("test_examples.parquet", examples, "target_item_id_internal"),
        ("predictions.parquet", predictions, "pred_item_id_internal"),
    ]:
        if column not in frame.columns:
            continue
        missing_ids = set(frame[column].dropna().astype("int64")) - mapped_ids
        if missing_ids:
            sample = sorted(missing_ids)[:5]
            raise ValueError(f"{frame_name} has internal item IDs missing from item_vocab.json: {sample}")


def _configure_mlflow_env(config: dict[str, Any]) -> None:
    """Load local env and map DagsHub token vars to MLflow basic auth vars."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"))
    except Exception:
        pass

    cfg = registry_config(config)
    username_var = str(cfg.get("dagshub_username_env_var", "DAGSHUB_USERNAME"))
    token_var = str(cfg.get("dagshub_token_env_var", "DAGSHUB_USER_TOKEN"))
    username = os.getenv(username_var)
    token = os.getenv(token_var)
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if token:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        os.environ["DAGSHUB_USER_TOKEN"] = token


def _configure_tracking(config: dict[str, Any]):
    """Configure MLflow/DagsHub the same way the upstream recsys repo does."""
    cfg = registry_config(config)
    _configure_mlflow_env(config)
    mlflow = _import_mlflow()

    owner = str(cfg.get("dagshub_repo_owner") or "").strip()
    repo = str(cfg.get("dagshub_repo_name") or "").strip()
    if owner and repo:
        try:
            import dagshub
        except ImportError as exc:
            raise RuntimeError(
                "The `dagshub` package is required for registry inheritance from DagsHub."
            ) from exc
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
    else:
        mlflow.set_tracking_uri(tracking_uri_from_config(config))
    return mlflow


def _single_downloaded_directory(tmp_destination: Path, artifact_path: str) -> Path:
    direct = tmp_destination / Path(artifact_path).name
    if direct.exists() and direct.is_dir():
        return direct
    children = [child for child in tmp_destination.iterdir() if child.name != ".DS_Store"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return tmp_destination


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unnamed"


def _import_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MLflow is required for `make inherit`. Install project dependencies with "
            "`pip install -r requirements.txt`, then rerun after configuring DagsHub credentials."
        ) from exc
    return mlflow
