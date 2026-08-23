from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from f1_race_predictor.artifacts import (
    portable_path,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from f1_race_predictor.prediction import DEFAULT_REGISTRY_ROOT, generate_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "predictions" / "archive"
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "predictions" / "evaluations"


def validate_prediction(prediction: dict[str, Any]) -> None:
    drivers = prediction.get("drivers", [])
    if not drivers:
        raise ValueError("Prediction contains no drivers.")
    positions = [int(row["position"]) for row in drivers]
    driver_ids = [str(row["driver_id"]) for row in drivers]
    if positions != list(range(1, len(drivers) + 1)):
        raise ValueError("Prediction positions are not complete and ordered.")
    if len(driver_ids) != len(set(driver_ids)):
        raise ValueError("Prediction contains duplicate drivers.")
    if prediction.get("record_type") not in {"live", "historical_backtest"}:
        raise ValueError("Prediction record type is missing or invalid.")


def archive_prediction(
    prediction_id: str,
    prediction: dict[str, Any],
    archive_root: Path,
    registry_root: Path,
    archived_at: str | None = None,
) -> Path:
    validate_prediction(prediction)
    destination = archive_root / prediction_id
    destination.mkdir(parents=True, exist_ok=False)

    archived_prediction = dict(prediction)
    archived_prediction["prediction_id"] = prediction_id
    prediction_path = destination / "prediction.json"
    write_json_exclusive(prediction_path, archived_prediction)

    model_version = str(prediction["model_version"])
    registry_path = registry_root / model_version / "registry.json"
    manifest = {
        "prediction_id": prediction_id,
        "archived_at": archived_at or datetime.now(timezone.utc).isoformat(),
        "record_type": prediction["record_type"],
        "original_prediction": {
            "path": "prediction.json",
            "sha256": sha256_file(prediction_path),
        },
        "model_registry": {
            "model_version": model_version,
            "sha256": sha256_file(registry_path),
        },
        "evaluation_is_stored_separately": True,
    }
    write_json_exclusive(destination / "manifest.json", manifest)
    return destination


def spearman(predicted: list[int], actual: list[int]) -> float:
    count = len(predicted)
    squared_difference = sum((left - right) ** 2 for left, right in zip(predicted, actual))
    return 1.0 - (6.0 * squared_difference) / (count * (count**2 - 1))


def evaluate_prediction(
    prediction_id: str,
    archive_root: Path,
    evaluation_root: Path,
    results_file: Path,
    evaluated_at: str | None = None,
) -> Path:
    prediction = read_json(archive_root / prediction_id / "prediction.json")
    validate_prediction(prediction)
    race_id = prediction["race"]["race_id"]
    with results_file.open("r", encoding="utf-8", newline="") as handle:
        result_rows = [row for row in csv.DictReader(handle) if row["race_id"] == race_id]
    actual_by_driver = {row["driver_id"]: row for row in result_rows}
    predicted_ids = [row["driver_id"] for row in prediction["drivers"]]
    if set(predicted_ids) != set(actual_by_driver):
        raise ValueError("Prediction and official result do not contain the same drivers.")

    comparison = []
    for predicted in prediction["drivers"]:
        actual = actual_by_driver[predicted["driver_id"]]
        position_value = actual.get("actual_position") or actual.get("target_normalized_position")
        status_value = actual.get("actual_classification_status") or actual.get(
            "target_classification_status"
        )
        if position_value is None or status_value is None:
            raise ValueError("Official result columns are missing.")
        actual_position = int(position_value)
        comparison.append(
            {
                "driver_id": predicted["driver_id"],
                "predicted_position": int(predicted["position"]),
                "actual_position": actual_position,
                "classification_status": status_value,
                "absolute_position_error": abs(int(predicted["position"]) - actual_position),
            }
        )
    predicted_positions = [row["predicted_position"] for row in comparison]
    actual_positions = [row["actual_position"] for row in comparison]
    evaluation = {
        "prediction_id": prediction_id,
        "evaluated_at": evaluated_at or datetime.now(timezone.utc).isoformat(),
        "results_dataset": {
            "path": portable_path(results_file, PROJECT_ROOT),
            "sha256": sha256_file(results_file),
        },
        "metrics": {
            "spearman_correlation": spearman(predicted_positions, actual_positions),
            "mean_absolute_position_error": sum(
                row["absolute_position_error"] for row in comparison
            )
            / len(comparison),
            "winner_correct": next(
                row["driver_id"] for row in comparison if row["predicted_position"] == 1
            )
            == next(row["driver_id"] for row in comparison if row["actual_position"] == 1),
            "top_three_overlap": len(
                {row["driver_id"] for row in comparison if row["predicted_position"] <= 3}
                & {row["driver_id"] for row in comparison if row["actual_position"] <= 3}
            )
            / 3,
        },
        "official_result": sorted(comparison, key=lambda row: row["actual_position"]),
    }
    destination = evaluation_root / prediction_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(destination / "evaluation.json", evaluation)
    return destination


def reproduce_prediction(
    prediction_id: str,
    archive_root: Path,
    registry_root: Path,
    feature_file: Path,
) -> dict[str, Any]:
    archived = read_json(archive_root / prediction_id / "prediction.json")
    manifest = read_json(archive_root / prediction_id / "manifest.json")
    checks = {
        "archived_prediction_hash_matches": sha256_file(
            archive_root / prediction_id / "prediction.json"
        )
        == manifest["original_prediction"]["sha256"],
        "feature_dataset_hash_matches": sha256_file(feature_file)
        == archived["feature_dataset"]["sha256"],
        "model_registry_hash_matches": sha256_file(
            registry_root / archived["model_version"] / "registry.json"
        )
        == manifest["model_registry"]["sha256"],
    }
    reproduced = generate_prediction(
        registry_root,
        archived["model_version"],
        feature_file,
        archived["race"]["race_id"],
        archived["data_cutoff"],
        archived["record_type"],
        archived["created_at"],
    )
    checks["driver_order_matches"] = [
        (row["position"], row["driver_id"]) for row in reproduced["drivers"]
    ] == [(row["position"], row["driver_id"]) for row in archived["drivers"]]
    checks["model_scores_match"] = [row["model_score"] for row in reproduced["drivers"]] == [
        row["model_score"] for row in archived["drivers"]
    ]
    return {
        "prediction_id": prediction_id,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
    }


def archive_main() -> None:
    parser = argparse.ArgumentParser(description="Store a prediction without allowing replacement.")
    parser.add_argument("--prediction-id", required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--archived-at", help="Explicit ISO archive timestamp.")
    args = parser.parse_args()
    destination = archive_prediction(
        args.prediction_id,
        read_json(args.prediction_file.resolve()),
        args.archive_root.resolve(),
        args.registry_root.resolve(),
        args.archived_at,
    )
    print(destination)


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an archived prediction separately.")
    parser.add_argument("--prediction-id", required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--evaluated-at", help="Explicit ISO evaluation timestamp.")
    args = parser.parse_args()
    destination = evaluate_prediction(
        args.prediction_id,
        args.archive_root.resolve(),
        args.evaluation_root.resolve(),
        args.results_file.resolve(),
        args.evaluated_at,
    )
    print(destination)


def reproduce_main() -> None:
    parser = argparse.ArgumentParser(description="Verify and reproduce an archived prediction.")
    parser.add_argument("--prediction-id", required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    args = parser.parse_args()
    report = reproduce_prediction(
        args.prediction_id,
        args.archive_root.resolve(),
        args.registry_root.resolve(),
        args.feature_file.resolve(),
    )
    print(report)
    if report["status"] != "passed":
        raise ValueError("Archived prediction reproduction failed.")
