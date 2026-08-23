"""Collect Formula 1 schedules and race results with FastF1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import fastf1
import fastf1.ergast.interface
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

YEARS = (2023, 2024, 2025, 2026)


def to_json_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError, ValueError:
        pass
    return value


def save_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, default=to_json_value).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(payload)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def schedule_records(year: int) -> list[dict[str, Any]]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    records: list[dict[str, Any]] = []
    for _, event in schedule.iterrows():
        records.append({column: to_json_value(event[column]) for column in schedule.columns})
    return records


def race_results(client: Ergast, year: int) -> tuple[list[dict[str, Any]], int]:
    response = client.get_race_results(season=year, limit=1000)
    total_rows = int(response.total_results)
    collected_rows = 0
    pages = 0
    races_by_round: dict[int, dict[str, Any]] = {}

    while True:
        pages += 1
        for race in response:
            round_number = int(race["round"])
            if round_number not in races_by_round:
                race_copy = copy.deepcopy(race)
                race_copy["Results"] = []
                races_by_round[round_number] = race_copy
            race_rows = copy.deepcopy(race.get("Results", []))
            races_by_round[round_number]["Results"].extend(race_rows)
            collected_rows += len(race_rows)

        if collected_rows >= total_rows:
            break
        response = response.get_next_result_page()

    if collected_rows != total_rows:
        raise RuntimeError(
            f"FastF1 reported {total_rows} result rows for {year}, but {collected_rows} were collected"
        )
    return [races_by_round[key] for key in sorted(races_by_round)], pages


def validate(
    year: int, schedule: list[dict[str, Any]], races: list[dict[str, Any]]
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    race_rows = [row for race in races for row in race["Results"]]
    scheduled_rounds = {int(row["RoundNumber"]) for row in schedule}
    result_rounds = [int(race["round"]) for race in races]

    if len(result_rounds) != len(set(result_rounds)):
        errors.append("Duplicate race rounds")
    missing_schedule_rounds = sorted(set(result_rounds) - scheduled_rounds)
    if missing_schedule_rounds:
        errors.append(f"Result rounds missing from the schedule: {missing_schedule_rounds}")

    field_sizes: list[int] = []
    for race in races:
        rows = race["Results"]
        field_sizes.append(len(rows))
        numbers = [str(row["number"]) for row in rows]
        driver_ids = [row["Driver"]["driverId"] for row in rows]
        positions = [int(row["position"]) for row in rows]
        if len(numbers) != len(set(numbers)):
            errors.append(f"Round {race['round']} has duplicate driver numbers")
        if len(driver_ids) != len(set(driver_ids)):
            errors.append(f"Round {race['round']} has duplicate driver identities")
        if sorted(positions) != list(range(1, len(rows) + 1)):
            errors.append(f"Round {race['round']} does not contain one complete finishing order")

    if any(size < 19 for size in field_sizes):
        warnings.append("At least one race contains fewer than 19 result rows")

    return {
        "year": year,
        "scheduled_races": len(schedule),
        "completed_races_with_results": len(races),
        "result_rows": len(race_rows),
        "minimum_field_size": min(field_sizes) if field_sizes else 0,
        "maximum_field_size": max(field_sizes) if field_sizes else 0,
        "errors": errors,
        "warnings": warnings,
    }


def current_grid(races: list[dict[str, Any]]) -> dict[str, Any]:
    latest = races[-1]
    drivers = []
    for result in sorted(latest["Results"], key=lambda row: int(row["number"])):
        driver = result["Driver"]
        constructor = result["Constructor"]
        drivers.append(
            {
                "driver_number": int(result["number"]),
                "driver_id": driver["driverId"],
                "driver_code": driver.get("code"),
                "given_name": driver["givenName"],
                "family_name": driver["familyName"],
                "constructor_id": constructor["constructorId"],
                "constructor_name": constructor["name"],
            }
        )
    return {
        "selection_rule": "Drivers listed in the latest completed 2026 race result",
        "season": int(latest["season"]),
        "round": int(latest["round"]),
        "race_name": latest["raceName"],
        "race_date": latest["date"],
        "drivers": drivers,
    }


def collect(
    output_root: Path,
    cache_dir: Path,
    years: tuple[int, ...] = YEARS,
    as_of: datetime | None = None,
    retrieval_id: str | None = None,
) -> Path:
    started_at = datetime.now(timezone.utc)
    cutoff = as_of or started_at
    retrieval_id = retrieval_id or started_at.strftime("%Y%m%dT%H%M%SZ")
    retrieval_dir = output_root / retrieval_id
    retrieval_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    fastf1.ergast.interface.HEADERS["User-Agent"] = (
        f"F1RacePredictor/0.1 FastF1/{fastf1.__version__}"
    )

    client = Ergast(result_type="raw", auto_cast=False, limit=1000)
    files: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    result_pages: dict[str, int] = {}
    results_by_year: dict[int, list[dict[str, Any]]] = {}

    def store(path: Path, value: Any) -> None:
        metadata = save_json(path, value)
        metadata["path"] = path.relative_to(retrieval_dir).as_posix()
        files.append(metadata)

    for year in years:
        schedule = schedule_records(year)
        results, pages = race_results(client, year)
        results = [
            race
            for race in results
            if datetime.fromisoformat(str(race["date"])).replace(tzinfo=timezone.utc) <= cutoff
        ]
        results_by_year[year] = results
        result_pages[str(year)] = pages
        year_dir = retrieval_dir / str(year)
        store(year_dir / "race_schedule.json", schedule)
        store(year_dir / "race_results.json", results)
        validations.append(validate(year, schedule, results))

    errors = [error for result in validations for error in result["errors"]]
    report = {
        "retrieval_id": retrieval_id,
        "source": "FastF1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "yearly_validation": validations,
    }
    store(retrieval_dir / "validation_report.json", report)
    if 2026 in results_by_year and results_by_year[2026]:
        store(retrieval_dir / "current_2026_grid.json", current_grid(results_by_year[2026]))

    manifest = {
        "retrieval_id": retrieval_id,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "as_of": cutoff.isoformat(),
        "years": list(years),
        "source": "FastF1",
        "fastf1_version": fastf1.__version__,
        "python_version": platform.python_version(),
        "schedule_interface": "fastf1.get_event_schedule",
        "result_interface": "fastf1.ergast.Ergast.get_race_results",
        "result_backend": "Jolpica-F1 through FastF1",
        "result_pages": result_pages,
        "cache_directory": "data/cache/fastf1",
        "files": files,
    }
    save_json(retrieval_dir / "manifest.json", manifest)

    if errors:
        raise RuntimeError(f"Validation failed; inspect {retrieval_dir / 'validation_report.json'}")
    return retrieval_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "raw" / "fastf1",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data" / "cache" / "fastf1",
    )
    parser.add_argument(
        "--years",
        default=",".join(str(year) for year in YEARS),
        help="Comma-separated seasons to collect.",
    )
    parser.add_argument(
        "--as-of",
        help="UTC ISO timestamp used to exclude later race results. Defaults to the current time.",
    )
    parser.add_argument(
        "--retrieval-id",
        help="Explicit immutable output identifier. Defaults to the current UTC timestamp.",
    )
    args = parser.parse_args()
    years = tuple(int(value.strip()) for value in args.years.split(",") if value.strip())
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else None
    print(
        collect(
            args.output_root.resolve(),
            args.cache_dir.resolve(),
            years=years,
            as_of=as_of,
            retrieval_id=args.retrieval_id,
        )
    )


if __name__ == "__main__":
    main()
