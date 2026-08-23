from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "fastf1"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "fastf1"
YEARS = range(2023, 2027)

FIELDS = [
    "source",
    "source_retrieval_id",
    "race_id",
    "season",
    "round",
    "race_name",
    "race_date",
    "circuit_id",
    "circuit_name",
    "circuit_locality",
    "circuit_country",
    "driver_id",
    "driver_code",
    "driver_given_name",
    "driver_family_name",
    "driver_name",
    "driver_date_of_birth",
    "driver_nationality",
    "car_number",
    "constructor_id",
    "constructor_name",
    "constructor_nationality",
    "grid_position",
    "official_position",
    "official_position_text",
    "normalized_position",
    "classification_status",
    "source_status",
    "status_corrected",
    "laps_completed",
    "points",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize historical race results.")
    parser.add_argument(
        "--retrieval",
        help="FastF1 retrieval identifier. The latest passing retrieval is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to data/processed/fastf1/<retrieval>.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Explicit raw retrieval directory. Takes precedence over --retrieval.",
    )
    parser.add_argument(
        "--years",
        default="2023,2024,2025,2026",
        help="Comma-separated seasons to normalize.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_passing_retrieval() -> Path:
    candidates = []
    for path in RAW_ROOT.iterdir():
        report_path = path / "validation_report.json"
        if not path.is_dir() or not report_path.exists():
            continue
        report = read_json(report_path)
        if report.get("status") == "passed":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No passing FastF1 retrieval was found.")
    return max(candidates, key=lambda path: path.name)


def choose_retrieval(retrieval_id: str | None) -> Path:
    if retrieval_id:
        path = RAW_ROOT / retrieval_id
        if not path.is_dir():
            raise FileNotFoundError(f"FastF1 retrieval does not exist: {retrieval_id}")
        return path
    return latest_passing_retrieval()


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def corrected_status(
    season: int,
    race_round: int,
    driver_id: str,
    source_status: str,
) -> tuple[str, bool]:
    if (season, race_round, driver_id) == (2025, 1, "hadjar"):
        return "Did not start", True
    return source_status, False


def classification_status(
    position_text: str,
    source_status: str,
    laps_completed: int,
) -> str:
    if source_status == "Disqualified":
        return "DSQ"
    if source_status == "Did not start" or (source_status == "Withdrew" and laps_completed == 0):
        return "DNS"
    if position_text.isdigit():
        return "CLASSIFIED"
    if source_status == "Lapped":
        return "NC"
    return "DNF"


def normalize_result(
    race: dict[str, Any],
    result: dict[str, Any],
    retrieval_id: str,
) -> dict[str, Any]:
    season = to_int(race["season"])
    race_round = to_int(race["round"])
    driver = result["Driver"]
    constructor = result["Constructor"]
    circuit = race["Circuit"]
    location = circuit["Location"]
    source_status = str(result.get("status", ""))
    laps_completed = to_int(result.get("laps"))
    normalized_source_status, was_corrected = corrected_status(
        season,
        race_round,
        driver["driverId"],
        source_status,
    )
    position_text = str(result.get("positionText", ""))

    return {
        "source": "FastF1/Jolpica",
        "source_retrieval_id": retrieval_id,
        "race_id": f"{season}-{race_round:02d}",
        "season": season,
        "round": race_round,
        "race_name": race["raceName"],
        "race_date": race["date"],
        "circuit_id": circuit["circuitId"],
        "circuit_name": circuit["circuitName"],
        "circuit_locality": location["locality"],
        "circuit_country": location["country"],
        "driver_id": driver["driverId"],
        "driver_code": driver.get("code", ""),
        "driver_given_name": driver["givenName"],
        "driver_family_name": driver["familyName"],
        "driver_name": f"{driver['givenName']} {driver['familyName']}",
        "driver_date_of_birth": driver["dateOfBirth"],
        "driver_nationality": driver["nationality"],
        "car_number": result.get("number", ""),
        "constructor_id": constructor["constructorId"],
        "constructor_name": constructor["name"],
        "constructor_nationality": constructor["nationality"],
        "grid_position": to_int(result.get("grid")),
        "official_position": to_int(result["position"]),
        "official_position_text": position_text,
        "normalized_position": 0,
        "classification_status": classification_status(
            position_text,
            normalized_source_status,
            laps_completed,
        ),
        "source_status": source_status,
        "status_corrected": was_corrected,
        "laps_completed": laps_completed,
        "points": to_float(result.get("points")),
    }


def ranking_key(row: dict[str, Any]) -> tuple[int, int, int]:
    status_group = {
        "CLASSIFIED": 0,
        "NC": 1,
        "DNF": 1,
        "DNS": 2,
        "DSQ": 3,
    }[row["classification_status"]]
    if status_group == 1:
        return status_group, -row["laps_completed"], row["official_position"]
    return status_group, row["official_position"], 0


def assign_normalized_positions(rows: list[dict[str, Any]]) -> None:
    rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_race[row["race_id"]].append(row)
    for race_rows in rows_by_race.values():
        for position, row in enumerate(sorted(race_rows, key=ranking_key), start=1):
            row["normalized_position"] = position


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    driver_identity: dict[str, tuple[str, str]] = {}

    for row in rows:
        rows_by_race[row["race_id"]].append(row)
        identity = (row["driver_name"], row["driver_date_of_birth"])
        previous = driver_identity.setdefault(row["driver_id"], identity)
        if previous != identity:
            errors.append(f"Driver identity changed for {row['driver_id']}.")

    for race_id, race_rows in rows_by_race.items():
        driver_ids = [row["driver_id"] for row in race_rows]
        if len(driver_ids) != len(set(driver_ids)):
            errors.append(f"Duplicate driver in {race_id}.")

        expected_positions = list(range(1, len(race_rows) + 1))
        official_positions = sorted(row["official_position"] for row in race_rows)
        normalized_positions = sorted(row["normalized_position"] for row in race_rows)
        if official_positions != expected_positions:
            errors.append(f"Invalid official order in {race_id}.")
        if normalized_positions != expected_positions:
            errors.append(f"Invalid normalized order in {race_id}.")

        ordered_rows = sorted(race_rows, key=lambda row: row["normalized_position"])
        expected_rows = sorted(race_rows, key=ranking_key)
        if [row["driver_id"] for row in ordered_rows] != [
            row["driver_id"] for row in expected_rows
        ]:
            errors.append(f"Status ordering rule was not applied in {race_id}.")

    corrected_rows = [row for row in rows if row["status_corrected"]]
    expected_correction = [
        row
        for row in corrected_rows
        if row["race_id"] == "2025-01"
        and row["driver_id"] == "hadjar"
        and row["classification_status"] == "DNS"
    ]
    if len(corrected_rows) != 1 or len(expected_correction) != 1:
        errors.append("The Hadjar DNS correction was not applied exactly once.")

    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "race_count": len(rows_by_race),
        "driver_count": len(driver_identity),
        "corrected_row_count": len(corrected_rows),
        "errors": errors,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_dataset(
    retrieval_dir: Path,
    output_dir: Path,
    years: tuple[int, ...] = tuple(YEARS),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for year in years:
        path = retrieval_dir / str(year) / "race_results.json"
        for race in read_json(path):
            rows.extend(
                normalize_result(race, result, retrieval_dir.name) for result in race["Results"]
            )

    assign_normalized_positions(rows)
    rows.sort(key=lambda row: (row["season"], row["round"], row["normalized_position"]))
    report = validate(rows)
    if report["status"] != "passed":
        write_json(output_dir / "validation_report.json", report)
        raise ValueError("Normalized data validation failed.")

    write_csv(output_dir / "race_results.csv", rows)
    write_json(output_dir / "validation_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    retrieval_dir = args.input_dir.resolve() if args.input_dir else choose_retrieval(args.retrieval)
    output_dir = args.output_dir or PROCESSED_ROOT / retrieval_dir.name
    years = tuple(int(value.strip()) for value in args.years.split(",") if value.strip())
    report = normalize_dataset(retrieval_dir, output_dir.resolve(), years)
    print(f"Wrote {report['row_count']} rows to {output_dir / 'race_results.csv'}")
    print(f"Validation status: {report['status']}")


if __name__ == "__main__":
    main()
