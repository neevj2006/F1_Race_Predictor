from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from f1_race_predictor.artifacts import (
    portable_path,
    read_json,
    sha256_file,
    write_json_exclusive,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_ROOT = PROJECT_ROOT / "models" / "registry"


def load_feature_rows(feature_file: Path, race_id: str) -> list[dict[str, str]]:
    with feature_file.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["race_id"] == race_id]
    if not rows:
        raise ValueError(f"No feature rows found for {race_id}.")
    driver_ids = [row["driver_id"] for row in rows]
    if len(driver_ids) != len(set(driver_ids)):
        raise ValueError(f"Duplicate drivers found for {race_id}.")
    return rows


def generate_prediction(
    registry_root: Path,
    model_version: str,
    feature_file: Path,
    race_id: str,
    data_cutoff: str,
    record_type: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    model_dir = registry_root / model_version
    registry = read_json(model_dir / "registry.json")
    expected_hash = registry["artifacts"]["feature_dataset"]["sha256"]
    actual_hash = sha256_file(feature_file)
    if actual_hash != expected_hash:
        raise ValueError("Feature dataset does not match the registered model data.")

    rows = load_feature_rows(feature_file, race_id)
    feature_columns = registry["feature_columns"]
    matrix = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in rows],
        dtype=float,
    )
    estimator = joblib.load(model_dir / registry["artifacts"]["model"]["path"])
    scores = estimator.predict(matrix)
    scored_rows = sorted(
        zip(rows, scores),
        key=lambda item: (-float(item[1]), item[0]["driver_id"]),
    )
    order = [
        {
            "position": position,
            "driver_id": row["driver_id"],
            "driver_name": row["driver_name"],
            "constructor_id": row["constructor_id"],
            "constructor_name": row["constructor_name"],
            "model_score": float(score),
        }
        for position, (row, score) in enumerate(scored_rows, start=1)
    ]
    if [row["position"] for row in order] != list(range(1, len(order) + 1)):
        raise ValueError("Prediction does not contain a complete finishing order.")

    first = rows[0]
    return {
        "record_type": record_type,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "data_cutoff": data_cutoff,
        "race": {
            "race_id": race_id,
            "season": int(first["season"]),
            "round": int(first["round"]),
            "race_date": first["race_date"],
            "circuit_id": first["circuit_id"],
        },
        "model_version": model_version,
        "model_id": registry["model_id"],
        "feature_dataset": {
            "path": portable_path(feature_file, PROJECT_ROOT),
            "sha256": actual_hash,
        },
        "drivers": order,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one complete race-order prediction.")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--registry-root", type=Path, default=DEFAULT_REGISTRY_ROOT)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--data-cutoff", required=True)
    parser.add_argument(
        "--record-type",
        choices=("live", "historical_backtest"),
        required=True,
    )
    parser.add_argument("--created-at", help="Explicit ISO creation timestamp.")
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = generate_prediction(
        args.registry_root.resolve(),
        args.model_version,
        args.feature_file.resolve(),
        args.race_id,
        args.data_cutoff,
        args.record_type,
        args.created_at,
    )
    write_json_exclusive(args.output_file.resolve(), prediction)
    print(args.output_file.resolve())


if __name__ == "__main__":
    main()
