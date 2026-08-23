from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from f1_race_predictor.artifacts import portable_path, sha256_file, write_json_exclusive

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Grid timestamps must include a timezone.")
    return parsed


def create_snapshot(
    input_file: Path,
    output_file: Path,
    race_id: str,
    qualifying_cutoff: str,
    recorded_at: str,
) -> Path:
    cutoff_time = validate_timestamp(qualifying_cutoff)
    recorded_time = validate_timestamp(recorded_at)
    if recorded_time < cutoff_time:
        raise ValueError("The grid snapshot cannot be recorded before qualifying ends.")

    with input_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"driver_id", "official_grid_position", "penalty_note"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Grid input must contain driver, position, and penalty columns.")
    driver_ids = [row["driver_id"] for row in rows]
    positions = [int(row["official_grid_position"]) for row in rows]
    if len(driver_ids) != len(set(driver_ids)):
        raise ValueError("Grid input contains duplicate drivers.")
    if sorted(positions) != list(range(1, len(rows) + 1)):
        raise ValueError("Grid input must contain one complete starting order.")

    snapshot: dict[str, Any] = {
        "race_id": race_id,
        "qualifying_cutoff": qualifying_cutoff,
        "recorded_at": recorded_at,
        "source_file": {
            "path": portable_path(input_file, PROJECT_ROOT),
            "sha256": sha256_file(input_file),
        },
        "drivers": sorted(
            (
                {
                    "driver_id": row["driver_id"],
                    "official_grid_position": int(row["official_grid_position"]),
                    "penalty_note": row["penalty_note"],
                }
                for row in rows
            ),
            key=lambda row: row["official_grid_position"],
        ),
    }
    write_json_exclusive(output_file, snapshot)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an immutable grid and penalty snapshot.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--qualifying-cutoff", required=True)
    parser.add_argument("--recorded-at", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = create_snapshot(
        args.input_file.resolve(),
        args.output_file.resolve(),
        args.race_id,
        args.qualifying_cutoff,
        args.recorded_at,
    )
    print(output)


if __name__ == "__main__":
    main()
