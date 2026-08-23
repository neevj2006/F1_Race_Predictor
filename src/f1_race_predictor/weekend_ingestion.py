from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

from f1_race_predictor.artifacts import portable_path, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    PROJECT_ROOT / "data" / "processed" / "fastf1" / "20260810T200317Z" / "race_results.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "weekend" / "fastf1"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "fastf1"

QUALIFYING_COLUMNS = [
    "race_id",
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "qualifying_position",
    "qualifying_time_seconds",
    "qualifying_gap_percent",
    "qualifying_teammate_delta_percent",
    "qualifying_missing",
    "official_grid_position",
    "source_grid_position",
    "grid_change_from_qualifying",
    "penalty_note",
    "qualifying_cutoff",
    "grid_recorded_at",
]

PRACTICE_COLUMNS = [
    "race_id",
    "season",
    "round",
    "driver_id",
    "practice_sessions",
    "practice_laps",
    "accurate_laps",
    "practice_best_lap_seconds",
    "practice_gap_percent",
    "practice_sector_score",
    "long_run_laps",
    "long_run_median_seconds",
    "long_run_gap_percent",
    "degradation_percent_per_lap",
    "compounds_used",
    "practice_missing",
    "practice_data_timestamp",
]

CIRCUIT_COLUMNS = [
    "race_id",
    "season",
    "round",
    "circuit_id",
    "lap_distance_km",
    "corner_count",
    "high_speed_fraction",
    "corner_density",
    "brake_event_density",
    "tyre_degradation_percent_per_lap",
    "straight_demand",
    "cornering_demand",
    "braking_demand",
    "tyre_stress",
    "profile_missing",
    "profile_data_timestamp",
    "profile_source_race_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)


def seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timedelta, timedelta)):
        return float(value.total_seconds())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = text.split(":")
        if len(parts) == 2:
            return (float(parts[0]) * 60.0) + float(parts[1])
        if len(parts) == 3:
            return (float(parts[0]) * 3600.0) + (float(parts[1]) * 60.0) + float(parts[2])
        return float(text)
    return float(value)


def utc_text(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def session_end(session: Any) -> str:
    end = pd.Timestamp(session.date)
    if not session.session_status.empty and "Time" in session.session_status:
        end += session.session_status["Time"].max()
    return utc_text(end)


def race_groups(rows: Iterable[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return sorted(
        grouped.values(),
        key=lambda group: (int(group[0]["season"]), int(group[0]["round"])),
    )


def qualifying_time(row: dict[str, Any]) -> float | None:
    values = [seconds(row.get(column)) for column in ("Q1", "Q2", "Q3")]
    present = [value for value in values if value is not None]
    return min(present) if present else None


def normalized_grid(
    race_rows: list[dict[str, str]],
    qualifying_by_driver: dict[str, dict[str, Any]],
) -> dict[str, int]:
    ordered = sorted(
        race_rows,
        key=lambda row: (
            int(row["grid_position"]) <= 0,
            int(row["grid_position"]) if int(row["grid_position"]) > 0 else math.inf,
            qualifying_by_driver.get(row["driver_id"], {}).get("qualifying_position", math.inf),
            row["driver_id"],
        ),
    )
    return {row["driver_id"]: position for position, row in enumerate(ordered, start=1)}


def qualifying_rows(
    race_rows: list[dict[str, str]],
    qualifying_results: list[dict[str, Any]],
    qualifying_cutoff: str,
    grid_recorded_at: str,
) -> list[dict[str, Any]]:
    field_size = len(race_rows)
    qualifying_by_driver: dict[str, dict[str, Any]] = {}
    for result in qualifying_results:
        driver_id = str(result.get("driver_id", ""))
        if not driver_id:
            continue
        position_value = result.get("position")
        qualifying_by_driver[driver_id] = {
            "qualifying_position": (
                int(position_value)
                if position_value is not None and not pd.isna(position_value)
                else 0
            ),
            "qualifying_time_seconds": qualifying_time(result),
        }

    valid_times = [
        row["qualifying_time_seconds"]
        for row in qualifying_by_driver.values()
        if row["qualifying_time_seconds"] is not None
    ]
    fastest = min(valid_times) if valid_times else None
    normalized_grid_by_driver = normalized_grid(race_rows, qualifying_by_driver)
    constructor_times: dict[str, list[float]] = defaultdict(list)
    constructor_by_driver = {row["driver_id"]: row["constructor_id"] for row in race_rows}
    for driver_id, row in qualifying_by_driver.items():
        if row["qualifying_time_seconds"] is not None and driver_id in constructor_by_driver:
            constructor_times[constructor_by_driver[driver_id]].append(
                row["qualifying_time_seconds"]
            )

    output = []
    for race_row in race_rows:
        driver_id = race_row["driver_id"]
        qualifying = qualifying_by_driver.get(driver_id, {})
        q_position = int(qualifying.get("qualifying_position") or field_size)
        q_time = qualifying.get("qualifying_time_seconds")
        constructor = race_row["constructor_id"]
        teammate_times = [
            value
            for value in constructor_times.get(constructor, [])
            if q_time is None or not math.isclose(value, q_time)
        ]
        teammate_delta = (
            ((q_time - min(teammate_times)) / min(teammate_times)) * 100.0
            if q_time is not None and teammate_times
            else 0.0
        )
        grid_position = normalized_grid_by_driver[driver_id]
        source_grid = int(race_row["grid_position"])
        output.append(
            {
                "race_id": race_row["race_id"],
                "season": int(race_row["season"]),
                "round": int(race_row["round"]),
                "race_date": race_row["race_date"],
                "circuit_id": race_row["circuit_id"],
                "driver_id": driver_id,
                "qualifying_position": q_position,
                "qualifying_time_seconds": q_time if q_time is not None else "",
                "qualifying_gap_percent": (
                    ((q_time - fastest) / fastest) * 100.0
                    if q_time is not None and fastest is not None
                    else ""
                ),
                "qualifying_teammate_delta_percent": teammate_delta,
                "qualifying_missing": q_time is None,
                "official_grid_position": grid_position,
                "source_grid_position": source_grid,
                "grid_change_from_qualifying": grid_position - q_position,
                "penalty_note": (
                    "Official grid differs from qualifying order; the grid position is used without "
                    "guessing the cause."
                    if grid_position != q_position
                    else ""
                ),
                "qualifying_cutoff": qualifying_cutoff,
                "grid_recorded_at": grid_recorded_at,
            }
        )
    return output


def usable_laps(session: Any) -> pd.DataFrame:
    laps = session.laps.copy()
    if laps.empty:
        return laps
    mask = laps["LapTime"].notna()
    if "Deleted" in laps:
        mask &= ~laps["Deleted"].eq(True)
    return laps.loc[mask].copy()


def lap_records(session: Any, session_name: str) -> list[dict[str, Any]]:
    laps = usable_laps(session)
    records = []
    for _, lap in laps.iterrows():
        records.append(
            {
                "driver_code": str(lap.get("Driver", "")),
                "session_name": session_name,
                "lap_time": seconds(lap.get("LapTime")),
                "sector_1": seconds(lap.get("Sector1Time")),
                "sector_2": seconds(lap.get("Sector2Time")),
                "sector_3": seconds(lap.get("Sector3Time")),
                "compound": str(lap.get("Compound", "") or ""),
                "stint": int(lap.get("Stint") or 0) if not pd.isna(lap.get("Stint")) else 0,
                "tyre_life": float(lap.get("TyreLife") or 0.0)
                if not pd.isna(lap.get("TyreLife"))
                else 0.0,
                "accurate": bool(lap.get("IsAccurate", False)),
            }
        )
    return records


def long_run_values(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    lap_times: list[float] = []
    degradation_values: list[float] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["accurate"] and row["lap_time"] is not None and row["compound"]:
            grouped[(row["session_name"], row["stint"], row["compound"])].append(row)
    for stint_rows in grouped.values():
        if len(stint_rows) < 4:
            continue
        stint_rows = sorted(stint_rows, key=lambda row: row["tyre_life"])
        times = np.asarray([row["lap_time"] for row in stint_rows], dtype=float)
        tyre_life = np.asarray([row["tyre_life"] for row in stint_rows], dtype=float)
        median = float(np.median(times))
        if median <= 0 or np.max(tyre_life) == np.min(tyre_life):
            continue
        reasonable = times <= median * 1.07
        if int(reasonable.sum()) < 4:
            continue
        selected_times = times[reasonable]
        selected_life = tyre_life[reasonable]
        slope = float(np.polyfit(selected_life, selected_times, 1)[0])
        lap_times.extend(selected_times.tolist())
        degradation_values.append((slope / float(np.median(selected_times))) * 100.0)
    return lap_times, degradation_values


def practice_rows(
    race_rows: list[dict[str, str]],
    practice_records: list[dict[str, Any]],
    timestamps: list[str],
) -> list[dict[str, Any]]:
    code_to_driver = {row["driver_code"]: row["driver_id"] for row in race_rows}
    by_driver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in practice_records:
        driver_id = code_to_driver.get(record["driver_code"])
        if driver_id:
            by_driver[driver_id].append(record)

    best_by_driver = {
        driver_id: min(row["lap_time"] for row in records if row["lap_time"] is not None)
        for driver_id, records in by_driver.items()
        if any(row["lap_time"] is not None for row in records)
    }
    overall_best = min(best_by_driver.values()) if best_by_driver else None
    long_run_by_driver = {
        driver_id: long_run_values(records) for driver_id, records in by_driver.items()
    }
    long_run_medians = {
        driver_id: float(np.median(values[0]))
        for driver_id, values in long_run_by_driver.items()
        if values[0]
    }
    overall_long_run = min(long_run_medians.values()) if long_run_medians else None

    output = []
    for race_row in race_rows:
        driver_id = race_row["driver_id"]
        records = by_driver.get(driver_id, [])
        best = best_by_driver.get(driver_id)
        sector_scores = []
        for key in ("sector_1", "sector_2", "sector_3"):
            values = [row[key] for row in records if row[key] is not None]
            if values:
                sector_scores.append(min(values))
        long_laps, degradation = long_run_by_driver.get(driver_id, ([], []))
        long_median = long_run_medians.get(driver_id)
        output.append(
            {
                "race_id": race_row["race_id"],
                "season": int(race_row["season"]),
                "round": int(race_row["round"]),
                "driver_id": driver_id,
                "practice_sessions": len({row["session_name"] for row in records}),
                "practice_laps": len(records),
                "accurate_laps": sum(row["accurate"] for row in records),
                "practice_best_lap_seconds": best if best is not None else "",
                "practice_gap_percent": (
                    ((best - overall_best) / overall_best) * 100.0
                    if best is not None and overall_best is not None
                    else ""
                ),
                "practice_sector_score": sum(sector_scores) if sector_scores else "",
                "long_run_laps": len(long_laps),
                "long_run_median_seconds": long_median if long_median is not None else "",
                "long_run_gap_percent": (
                    ((long_median - overall_long_run) / overall_long_run) * 100.0
                    if long_median is not None and overall_long_run is not None
                    else ""
                ),
                "degradation_percent_per_lap": (
                    float(np.median(degradation)) if degradation else ""
                ),
                "compounds_used": "|".join(
                    sorted({row["compound"] for row in records if row["compound"]})
                ),
                "practice_missing": not records,
                "practice_data_timestamp": max(timestamps) if timestamps else "",
            }
        )
    return output


def telemetry_profile(session: Any) -> dict[str, float | int | None]:
    fastest_lap = session.laps.pick_fastest()
    if fastest_lap is None or pd.isna(fastest_lap.get("LapTime")):
        return {
            "lap_distance_km": None,
            "corner_count": None,
            "high_speed_fraction": None,
            "corner_density": None,
            "brake_event_density": None,
        }
    telemetry = fastest_lap.get_car_data().add_distance()
    if telemetry.empty or float(telemetry["Distance"].max()) <= 0:
        return {
            "lap_distance_km": None,
            "corner_count": None,
            "high_speed_fraction": None,
            "corner_density": None,
            "brake_event_density": None,
        }
    distance = telemetry["Distance"].astype(float).to_numpy()
    segment_distance = np.diff(distance, prepend=distance[0])
    segment_distance = np.maximum(segment_distance, 0.0)
    total_distance = float(segment_distance.sum())
    high_speed = telemetry["Speed"].astype(float).to_numpy() >= 250.0
    high_speed_fraction = (
        float(segment_distance[high_speed].sum() / total_distance) if total_distance else 0.0
    )
    brake = telemetry["Brake"].fillna(False).astype(bool).to_numpy()
    brake_events = int(np.logical_and(brake, ~np.roll(brake, 1)).sum())
    throttle = telemetry["Throttle"].fillna(0.0).astype(float).to_numpy()
    throttle_lifts = int(np.logical_and(throttle < 95.0, np.roll(throttle, 1) >= 95.0).sum())
    lap_distance_km = float(distance.max()) / 1000.0
    corner_count = max(brake_events, throttle_lifts)
    return {
        "lap_distance_km": lap_distance_km,
        "corner_count": corner_count,
        "high_speed_fraction": high_speed_fraction,
        "corner_density": corner_count / lap_distance_km if lap_distance_km else None,
        "brake_event_density": brake_events / lap_distance_km if lap_distance_km else None,
    }


def scale_profiles(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        high_speed = row["high_speed_fraction"]
        corner_density = row["corner_density"]
        brake_density = row["brake_event_density"]
        degradation = row["tyre_degradation_percent_per_lap"]
        row["straight_demand"] = float(high_speed) if high_speed not in (None, "") else 0.5
        row["cornering_demand"] = (
            float(np.clip(float(corner_density) / 5.0, 0.0, 1.0))
            if corner_density not in (None, "")
            else 0.5
        )
        row["braking_demand"] = (
            float(np.clip(float(brake_density) / 3.0, 0.0, 1.0))
            if brake_density not in (None, "")
            else 0.5
        )
        row["tyre_stress"] = (
            float(np.clip(max(float(degradation), 0.0) / 0.20, 0.0, 1.0))
            if degradation not in (None, "")
            else 0.5
        )


def grid_record_time(event: Any) -> str:
    race_date = pd.Timestamp(event.get_session_date("Race", utc=True))
    return utc_text(race_date - pd.Timedelta(2, unit="h"))


def qualifying_cutoff_time(event: Any) -> str:
    qualifying_date = pd.Timestamp(event.get_session_date("Qualifying", utc=True))
    return utc_text(qualifying_date + pd.Timedelta(2, unit="h"))


def fetch_qualifying_results(years: set[int]) -> dict[str, list[dict[str, Any]]]:
    client = Ergast(result_type="raw", auto_cast=False, limit=2000)
    by_race: dict[str, list[dict[str, Any]]] = {}
    for year in sorted(years):
        response = client.get_qualifying_results(season=year, limit=2000)
        total_rows = int(response.total_results)
        collected_rows = 0
        races_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
        while True:
            for race in response:
                race_results = copy.deepcopy(race.get("QualifyingResults", []))
                collected_rows += len(race_results)
                round_number = int(race["round"])
                for result in race_results:
                    races_by_round[round_number].append(
                        {
                            "driver_id": result["Driver"]["driverId"],
                            "position": result["position"],
                            "Q1": result.get("Q1"),
                            "Q2": result.get("Q2"),
                            "Q3": result.get("Q3"),
                        }
                    )
            if collected_rows >= total_rows:
                break
            response = response.get_next_result_page()
        if collected_rows != total_rows:
            raise RuntimeError(
                f"FastF1 reported {total_rows} qualifying rows for {year}, "
                f"but {collected_rows} were collected"
            )
        for round_number, results in races_by_round.items():
            by_race[f"{year}-{round_number:02d}"] = results
    return by_race


def collect(
    results_file: Path,
    output_root: Path,
    cache_dir: Path,
    retrieval_id: str,
) -> Path:
    output_dir = output_root / retrieval_id
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    fastf1.set_log_level("WARNING")

    source_rows = read_csv(results_file)
    qualifying_output: list[dict[str, Any]] = []
    practice_output: list[dict[str, Any]] = []
    circuit_output: list[dict[str, Any]] = []
    warnings: list[str] = []

    groups = race_groups(source_rows)
    qualifying_history = fetch_qualifying_results({int(group[0]["season"]) for group in groups})
    profile_by_circuit: dict[str, dict[str, Any]] = {}
    for index, race_rows in enumerate(groups, start=1):
        season = int(race_rows[0]["season"])
        race_round = int(race_rows[0]["round"])
        race_id = race_rows[0]["race_id"]
        circuit_id = race_rows[0]["circuit_id"]
        print(f"[{index}/{len(groups)}] {race_id}", flush=True)
        event = fastf1.get_event(season, race_round)
        qualifying_output.extend(
            qualifying_rows(
                race_rows,
                qualifying_history.get(race_id, []),
                qualifying_cutoff_time(event),
                grid_record_time(event),
            )
        )
        if race_id not in qualifying_history:
            warnings.append(f"{race_id} qualifying result unavailable; grid fallback used")

        practice_records: list[dict[str, Any]] = []
        practice_timestamps: list[str] = []
        practice_names = []
        for session_number in range(1, 6):
            session_name = str(event.get(f"Session{session_number}", ""))
            if session_name.startswith("Practice"):
                practice_names.append(session_name)
        for session_name in practice_names[-1:]:
            try:
                practice = fastf1.get_session(season, race_round, session_name)
                practice.load(laps=True, telemetry=False, weather=False, messages=False)
                practice_records.extend(lap_records(practice, session_name))
                practice_timestamps.append(session_end(practice))
            except Exception as error:
                warnings.append(
                    f"{race_id} {session_name} unavailable: {type(error).__name__}: {error}"
                )
        race_practice = practice_rows(race_rows, practice_records, practice_timestamps)
        practice_output.extend(race_practice)
        degradation_values = [
            float(row["degradation_percent_per_lap"])
            for row in race_practice
            if row["degradation_percent_per_lap"] not in (None, "")
        ]
        if circuit_id not in profile_by_circuit:
            try:
                qualifying = fastf1.get_session(season, race_round, "Q")
                qualifying.load(laps=True, telemetry=True, weather=False, messages=False)
                profile_by_circuit[circuit_id] = {
                    **telemetry_profile(qualifying),
                    "profile_data_timestamp": session_end(qualifying),
                    "profile_source_race_id": race_id,
                }
            except Exception as error:
                warnings.append(
                    f"{race_id} circuit telemetry unavailable: {type(error).__name__}: {error}"
                )
        raw_profile = profile_by_circuit.get(
            circuit_id,
            {
                "lap_distance_km": None,
                "corner_count": None,
                "high_speed_fraction": None,
                "corner_density": None,
                "brake_event_density": None,
                "profile_data_timestamp": "",
                "profile_source_race_id": "",
            },
        )
        circuit_output.append(
            {
                "race_id": race_id,
                "season": season,
                "round": race_round,
                "circuit_id": circuit_id,
                **raw_profile,
                "tyre_degradation_percent_per_lap": (
                    float(np.median(degradation_values)) if degradation_values else ""
                ),
                "profile_missing": raw_profile["lap_distance_km"] is None,
            }
        )

    scale_profiles(circuit_output)
    write_csv(output_dir / "qualifying_grid.csv", qualifying_output, QUALIFYING_COLUMNS)
    write_csv(output_dir / "practice_summary.csv", practice_output, PRACTICE_COLUMNS)
    write_csv(output_dir / "circuit_profiles.csv", circuit_output, CIRCUIT_COLUMNS)

    qualifying_keys = {(row["race_id"], row["driver_id"]) for row in qualifying_output}
    practice_keys = {(row["race_id"], row["driver_id"]) for row in practice_output}
    source_keys = {(row["race_id"], row["driver_id"]) for row in source_rows}
    checks = {
        "qualifying_has_one_row_per_driver_race": len(qualifying_keys) == len(qualifying_output),
        "qualifying_matches_result_grid": qualifying_keys == source_keys,
        "practice_has_one_row_per_driver_race": len(practice_keys) == len(practice_output),
        "practice_matches_result_grid": practice_keys == source_keys,
        "one_circuit_profile_per_race": len(circuit_output) == len(groups),
        "qualifying_precedes_grid_snapshot": all(
            row["qualifying_cutoff"] <= row["grid_recorded_at"] for row in qualifying_output
        ),
        "no_weather_fields": not any(
            "weather" in column.lower()
            for column in QUALIFYING_COLUMNS + PRACTICE_COLUMNS + CIRCUIT_COLUMNS
        ),
    }
    errors = [f"Failed check: {name}" for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not errors else "failed",
        "retrieval_id": retrieval_id,
        "race_count": len(groups),
        "qualifying_rows": len(qualifying_output),
        "practice_rows": len(practice_output),
        "circuit_profiles": len(circuit_output),
        "practice_driver_coverage": (
            sum(not row["practice_missing"] for row in practice_output) / len(practice_output)
            if practice_output
            else 0.0
        ),
        "profile_coverage": (
            sum(not row["profile_missing"] for row in circuit_output) / len(circuit_output)
            if circuit_output
            else 0.0
        ),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(output_dir / "validation_report.json", report)
    manifest = {
        "retrieval_id": retrieval_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "FastF1 session timing data and historical official race grids",
        "practice_selection": (
            "The latest available practice session before qualifying is used for each race."
        ),
        "circuit_profile_method": (
            "Telemetry is calculated once per circuit from its earliest available qualifying "
            "session, then reused as a physical track profile. Tyre degradation remains "
            "race-specific."
        ),
        "fastf1_version": fastf1.__version__,
        "python_version": platform.python_version(),
        "results_dataset": {
            "path": portable_path(results_file, PROJECT_ROOT),
            "sha256": sha256_file(results_file),
        },
        "cache_directory": portable_path(cache_dir, PROJECT_ROOT),
        "weather_included": False,
        "penalty_handling": (
            "The final official grid is stored. Grid changes are recorded without inferring an "
            "unverified penalty cause. Live predictions require a cutoff-time grid input."
        ),
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in (
                "qualifying_grid.csv",
                "practice_summary.csv",
                "circuit_profiles.csv",
                "validation_report.json",
            )
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    if errors:
        raise ValueError("Weekend data validation failed.")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect pre-race weekend summaries.")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--retrieval-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = collect(
        args.results_file.resolve(),
        args.output_root.resolve(),
        args.cache_dir.resolve(),
        args.retrieval_id,
    )
    print(output)


if __name__ == "__main__":
    main()
