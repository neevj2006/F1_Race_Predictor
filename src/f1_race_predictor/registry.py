from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1_race_predictor.artifacts import (
    copy_exclusive,
    portable_path,
    read_json,
    sha256_file,
    write_json_exclusive,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_ROOT = PROJECT_ROOT / "models" / "registry"


def training_races(feature_path: Path, target_path: Path, cutoff_date: str) -> list[str]:
    with feature_path.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    with target_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    race_dates = {row["race_id"]: row["race_date"] for row in feature_rows}
    classified_races: set[str] = set()
    for row in rows:
        if row["target_classification_status"] == "CLASSIFIED":
            classified_races.add(row["race_id"])
    return sorted(
        (race_id for race_id in classified_races if race_dates[race_id] <= cutoff_date),
        key=lambda race_id: tuple(int(value) for value in race_id.split("-")),
    )


def register_model(
    version: str,
    model_dir: Path,
    feature_dir: Path,
    evaluation_dir: Path,
    registry_root: Path,
    created_at: str | None = None,
) -> Path:
    destination = registry_root / version
    destination.mkdir(parents=True, exist_ok=False)

    source_model = model_dir / "ranking_model.joblib"
    source_metadata = model_dir / "model_metadata.json"
    metadata = read_json(source_metadata)
    feature_file = feature_dir / Path(str(metadata["feature_dataset"])).name
    target_file = feature_dir / Path(str(metadata["target_dataset"])).name
    feature_manifest_path = feature_dir / "feature_manifest.json"
    evaluation_manifest_path = evaluation_dir / "evaluation_manifest.json"
    evaluation_summary_path = (
        evaluation_dir / "model_summary.csv"
        if (evaluation_dir / "model_summary.csv").is_file()
        else evaluation_dir / "model_summary.json"
    )
    required = [
        source_model,
        source_metadata,
        feature_file,
        target_file,
        feature_manifest_path,
        evaluation_manifest_path,
        evaluation_summary_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing registry inputs: " + ", ".join(missing))

    feature_manifest = read_json(feature_manifest_path)
    evaluation_manifest = read_json(evaluation_manifest_path)
    registered_model = destination / "model.joblib"
    copy_exclusive(source_model, registered_model)

    record: dict[str, Any] = {
        "model_version": version,
        "registered_at": created_at or datetime.now(timezone.utc).isoformat(),
        "model_id": metadata["model_id"],
        "model_family": metadata["model_family"],
        "feature_set": metadata["feature_set"],
        "history_scheme": metadata.get("history_scheme"),
        "context_group": metadata.get("context_group"),
        "feature_columns": metadata["feature_columns"],
        "parameters": metadata["parameters"],
        "random_seed": metadata["random_seed"],
        "season_weights": feature_manifest["season_weights"],
        "training_result_filter": metadata["training_result_filter"],
        "training_races": training_races(
            feature_file,
            target_file,
            metadata["training_cutoff_date"],
        ),
        "training_cutoff": {
            "season": metadata["training_cutoff_season"],
            "round": metadata["training_cutoff_round"],
            "date": metadata["training_cutoff_date"],
        },
        "evaluation": metadata["evaluation"],
        "evaluation_settings": evaluation_manifest,
        "artifacts": {
            "model": {
                "path": "model.joblib",
                "sha256": sha256_file(registered_model),
            },
            "source_model_metadata": {
                "path": portable_path(source_metadata, PROJECT_ROOT),
                "sha256": sha256_file(source_metadata),
            },
            "feature_dataset": {
                "path": portable_path(feature_file, PROJECT_ROOT),
                "sha256": sha256_file(feature_file),
            },
            "target_dataset": {
                "path": portable_path(target_file, PROJECT_ROOT),
                "sha256": sha256_file(target_file),
            },
            "feature_manifest": {
                "path": portable_path(feature_manifest_path, PROJECT_ROOT),
                "sha256": sha256_file(feature_manifest_path),
            },
            "evaluation_manifest": {
                "path": portable_path(evaluation_manifest_path, PROJECT_ROOT),
                "sha256": sha256_file(evaluation_manifest_path),
            },
            "evaluation_summary": {
                "path": portable_path(evaluation_summary_path, PROJECT_ROOT),
                "sha256": sha256_file(evaluation_summary_path),
            },
        },
    }
    write_json_exclusive(destination / "registry.json", record)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an immutable trained model.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--created-at", help="Explicit ISO registration timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = register_model(
        args.version,
        args.model_dir.resolve(),
        args.feature_dir.resolve(),
        args.evaluation_dir.resolve(),
        args.registry_root.resolve(),
        args.created_at,
    )
    print(destination)


if __name__ == "__main__":
    main()
