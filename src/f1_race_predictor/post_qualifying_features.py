from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from f1_race_predictor.artifacts import portable_path, sha256_file
from f1_race_predictor.features import FEATURE_COLUMNS as PRE_WEEKEND_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRE_WEEKEND_DIR = PROJECT_ROOT / "data" / "features" / "fastf1" / "20260810T200317Z"
DEFAULT_RESULTS = (
    PROJECT_ROOT / "data" / "processed" / "fastf1" / "20260810T200317Z" / "race_results.csv"
)
DEFAULT_WEEKEND_DIR = PROJECT_ROOT / "data" / "weekend" / "fastf1" / "20260823T130000Z"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "features" / "post_qualifying"

IDENTIFIER_COLUMNS = [
    "race_id",
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "post_qualifying_cutoff",
]

QUALIFYING_FEATURES = [
    "official_grid_position",
    "official_grid_score",
    "qualifying_position",
    "qualifying_position_score",
    "qualifying_gap_percent",
    "qualifying_pace_score",
    "qualifying_teammate_delta_percent",
    "grid_change_from_qualifying",
    "qualifying_missing",
    "qualifying_weekend_score",
]

PRACTICE_FEATURES = [
    "practice_laps",
    "practice_gap_percent",
    "practice_best_lap_score",
    "practice_sector_score",
    "practice_sector_rank_score",
    "long_run_laps",
    "long_run_gap_percent",
    "long_run_score",
    "degradation_percent_per_lap",
    "practice_compound_count",
    "practice_missing",
    "practice_overall_score",
]

RELIABILITY_FEATURES = [
    "driver_prior_reliability_starts",
    "driver_recent_dnf_rate_10",
    "constructor_prior_reliability_starts",
    "constructor_recent_dnf_rate_5",
    "constructor_recent_dnf_rate_10_weekend",
    "reliability_score",
]

PROFILE_FEATURES = [
    "straight_demand",
    "cornering_demand",
    "braking_demand",
    "tyre_stress",
    "profile_missing",
    "constructor_profile_prior_races",
    "constructor_profile_prior_score",
]

FEATURE_COLUMNS = (
    list(PRE_WEEKEND_FEATURE_COLUMNS)
    + QUALIFYING_FEATURES
    + PRACTICE_FEATURES
    + RELIABILITY_FEATURES
    + PROFILE_FEATURES
)

FEATURE_OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
TARGET_COLUMNS = [
    "race_id",
    "driver_id",
    "training_season_weight",
    "target_normalized_position",
    "target_official_position",
    "target_classification_status",
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


def as_float(value: Any, fallback: float = 0.0) -> float:
    if value in (None, ""):
        return fallback
    return float(value)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def keyed(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    values = {(row["race_id"], row["driver_id"]): row for row in rows}
    if len(values) != len(list(rows)):
        raise ValueError("Duplicate driver/race rows were found.")
    return values


def grouped_results(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return sorted(
        grouped.values(),
        key=lambda group: (group[0]["race_date"], int(group[0]["round"])),
    )


def lower_is_better_score(value: float | None, scale: float = 5.0) -> float:
    if value is None:
        return 0.5
    return float(np.clip(1.0 - (max(value, 0.0) / scale), 0.0, 1.0))


def field_score(position: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return 1.0 - ((position - 1) / (field_size - 1))


def finish_score(position: int, field_size: int) -> float:
    return field_score(position, field_size)


def recent_rate(records: list[bool], window: int, fallback: float) -> float:
    selected = records[-window:]
    return sum(selected) / len(selected) if selected else fallback


def profile_vector(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        as_float(row.get("straight_demand"), 0.5),
        as_float(row.get("cornering_demand"), 0.5),
        as_float(row.get("braking_demand"), 0.5),
        as_float(row.get("tyre_stress"), 0.5),
    )


def profile_prior(
    records: list[dict[str, Any]],
    current_profile: tuple[float, float, float, float],
    fallback: float,
) -> tuple[int, float]:
    weighted = []
    for record in records:
        distance = sum(
            abs(left - right) for left, right in zip(current_profile, record["profile"])
        ) / len(current_profile)
        similarity = max(0.0, 1.0 - distance)
        if similarity > 0:
            weighted.append((record["finish_score"], similarity))
    weight = sum(item[1] for item in weighted)
    score = (
        (sum(value * item_weight for value, item_weight in weighted) + 2.0 * fallback)
        / (weight + 2.0)
        if weighted
        else fallback
    )
    return len(weighted), score


def practice_feature_values(row: dict[str, str]) -> dict[str, Any]:
    best_gap = as_float(row.get("practice_gap_percent"), 5.0)
    long_gap = as_float(row.get("long_run_gap_percent"), 5.0)
    sector_value = as_float(row.get("practice_sector_score"), 0.0)
    degradation = as_float(row.get("degradation_percent_per_lap"), 0.0)
    missing = as_bool(row.get("practice_missing", True))
    best_score = lower_is_better_score(best_gap)
    long_score = lower_is_better_score(long_gap)
    return {
        "practice_laps": int(as_float(row.get("practice_laps"))),
        "practice_gap_percent": best_gap,
        "practice_best_lap_score": best_score,
        "practice_sector_score": sector_value,
        "practice_sector_rank_score": 0.5,
        "long_run_laps": int(as_float(row.get("long_run_laps"))),
        "long_run_gap_percent": long_gap,
        "long_run_score": long_score,
        "degradation_percent_per_lap": degradation,
        "practice_compound_count": len(
            [value for value in row.get("compounds_used", "").split("|") if value]
        ),
        "practice_missing": int(missing),
        "practice_overall_score": 0.5 if missing else (best_score + long_score + 0.5) / 3.0,
    }


def add_sector_rank_scores(rows: list[dict[str, Any]]) -> None:
    present = [
        row for row in rows if not row["practice_missing"] and row["practice_sector_score"] > 0
    ]
    ordered = sorted(present, key=lambda row: (row["practice_sector_score"], row["driver_id"]))
    field_size = len(ordered)
    for position, row in enumerate(ordered, start=1):
        row["practice_sector_rank_score"] = field_score(position, field_size)
        row["practice_overall_score"] = (
            row["practice_best_lap_score"]
            + row["long_run_score"]
            + row["practice_sector_rank_score"]
        ) / 3.0


def build_features(
    pre_weekend_rows: list[dict[str, str]],
    result_rows: list[dict[str, str]],
    qualifying_rows: list[dict[str, str]],
    practice_rows: list[dict[str, str]],
    profile_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    pre_by_key = keyed(pre_weekend_rows)
    qualifying_by_key = keyed(qualifying_rows)
    practice_by_key = keyed(practice_rows)
    profile_by_race = {row["race_id"]: row for row in profile_rows}
    if len(profile_by_race) != len(profile_rows):
        raise ValueError("Duplicate circuit profiles were found.")

    driver_reliability: dict[str, list[bool]] = defaultdict(list)
    constructor_reliability: dict[str, list[bool]] = defaultdict(list)
    constructor_profile_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_nonfinishes: list[bool] = []
    output: list[dict[str, Any]] = []

    for race_rows in grouped_results(result_rows):
        race_id = race_rows[0]["race_id"]
        profile_row = profile_by_race[race_id]
        current_profile = profile_vector(profile_row)
        field_size = len(race_rows)
        global_fallback = (
            sum(global_nonfinishes) / len(global_nonfinishes) if global_nonfinishes else 0.0
        )
        race_features = []
        for result in race_rows:
            key = (race_id, result["driver_id"])
            pre = pre_by_key[key]
            qualifying = qualifying_by_key[key]
            practice = practice_by_key[key]
            grid_position = int(qualifying["official_grid_position"])
            q_position = int(qualifying["qualifying_position"])
            q_gap = as_float(qualifying.get("qualifying_gap_percent"), 5.0)
            grid_score = field_score(grid_position, field_size)
            q_position_score = field_score(q_position, field_size)
            q_pace_score = lower_is_better_score(q_gap)
            weekend_score = 0.65 * grid_score + 0.25 * q_position_score + 0.10 * q_pace_score

            driver_history = driver_reliability[result["driver_id"]]
            constructor_history = constructor_reliability[result["constructor_id"]]
            driver_dnf_rate = recent_rate(driver_history, 10, global_fallback)
            constructor_dnf_5 = recent_rate(constructor_history, 5, global_fallback)
            constructor_dnf_10 = recent_rate(constructor_history, 10, global_fallback)
            reliability_score = 1.0 - (0.25 * driver_dnf_rate + 0.75 * constructor_dnf_10)
            profile_count, constructor_profile_score = profile_prior(
                constructor_profile_history[result["constructor_id"]],
                current_profile,
                as_float(pre["constructor_recent_finish_score_10"], 0.5),
            )

            feature_row: dict[str, Any] = {
                **{column: pre[column] for column in IDENTIFIER_COLUMNS if column in pre},
                "post_qualifying_cutoff": qualifying["grid_recorded_at"],
                **{column: as_float(pre[column]) for column in PRE_WEEKEND_FEATURE_COLUMNS},
                "official_grid_position": grid_position,
                "official_grid_score": grid_score,
                "qualifying_position": q_position,
                "qualifying_position_score": q_position_score,
                "qualifying_gap_percent": q_gap,
                "qualifying_pace_score": q_pace_score,
                "qualifying_teammate_delta_percent": as_float(
                    qualifying.get("qualifying_teammate_delta_percent")
                ),
                "grid_change_from_qualifying": int(
                    as_float(qualifying.get("grid_change_from_qualifying"))
                ),
                "qualifying_missing": int(as_bool(qualifying.get("qualifying_missing"))),
                "qualifying_weekend_score": weekend_score,
                **practice_feature_values(practice),
                "driver_prior_reliability_starts": len(driver_history),
                "driver_recent_dnf_rate_10": driver_dnf_rate,
                "constructor_prior_reliability_starts": len(constructor_history),
                "constructor_recent_dnf_rate_5": constructor_dnf_5,
                "constructor_recent_dnf_rate_10_weekend": constructor_dnf_10,
                "reliability_score": reliability_score,
                "straight_demand": current_profile[0],
                "cornering_demand": current_profile[1],
                "braking_demand": current_profile[2],
                "tyre_stress": current_profile[3],
                "profile_missing": int(as_bool(profile_row.get("profile_missing"))),
                "constructor_profile_prior_races": profile_count,
                "constructor_profile_prior_score": constructor_profile_score,
            }
            race_features.append(feature_row)
        add_sector_rank_scores(race_features)
        output.extend(race_features)

        profile_finish_by_constructor: dict[str, list[float]] = defaultdict(list)
        for result in race_rows:
            started = result["classification_status"] != "DNS"
            dnf = result["classification_status"] == "DNF"
            if started:
                driver_reliability[result["driver_id"]].append(dnf)
                constructor_reliability[result["constructor_id"]].append(dnf)
                global_nonfinishes.append(dnf)
            if result["classification_status"] == "CLASSIFIED":
                profile_finish_by_constructor[result["constructor_id"]].append(
                    finish_score(int(result["normalized_position"]), field_size)
                )
        for constructor_id, values in profile_finish_by_constructor.items():
            constructor_profile_history[constructor_id].append(
                {
                    "profile": current_profile,
                    "finish_score": sum(values) / len(values),
                }
            )
    return output


def apply_grid_snapshot(
    qualifying_rows: list[dict[str, str]],
    snapshot_path: Path,
) -> None:
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    snapshot_by_driver = {row["driver_id"]: row for row in snapshot["drivers"]}
    race_rows = [row for row in qualifying_rows if row["race_id"] == snapshot["race_id"]]
    if {row["driver_id"] for row in race_rows} != set(snapshot_by_driver):
        raise ValueError("Grid snapshot drivers do not match the qualifying feature rows.")
    for row in race_rows:
        grid = snapshot_by_driver[row["driver_id"]]
        row["official_grid_position"] = str(grid["official_grid_position"])
        row["grid_change_from_qualifying"] = str(
            int(grid["official_grid_position"]) - int(row["qualifying_position"])
        )
        row["penalty_note"] = grid["penalty_note"]
        row["grid_recorded_at"] = snapshot["recorded_at"]


def signature(rows: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (row["race_id"], row["driver_id"]): tuple(row[column] for column in FEATURE_COLUMNS)
        for row in rows
    }


def validate(
    source_rows: list[dict[str, str]],
    feature_rows: list[dict[str, Any]],
    qualifying_rows: list[dict[str, str]],
    practice_rows: list[dict[str, str]],
    pre_weekend_hash_before: str,
    pre_weekend_file: Path,
) -> dict[str, Any]:
    source_keys = {(row["race_id"], row["driver_id"]) for row in source_rows}
    feature_keys = {(row["race_id"], row["driver_id"]) for row in feature_rows}
    qualifying_by_key = keyed(qualifying_rows)
    practice_by_key = keyed(practice_rows)
    checks = {
        "one_feature_row_per_driver_race": len(feature_keys) == len(feature_rows),
        "feature_keys_match_results": feature_keys == source_keys,
        "feature_values_complete": all(
            row[column] not in (None, "") for row in feature_rows for column in FEATURE_COLUMNS
        ),
        "grid_snapshot_follows_qualifying": all(
            row["qualifying_cutoff"] <= row["grid_recorded_at"] for row in qualifying_rows
        ),
        "practice_precedes_qualifying": all(
            not practice_by_key[key]["practice_data_timestamp"]
            or practice_by_key[key]["practice_data_timestamp"]
            <= qualifying_by_key[key]["qualifying_cutoff"]
            for key in feature_keys
        ),
        "pre_weekend_dataset_unchanged": sha256_file(pre_weekend_file) == pre_weekend_hash_before,
        "no_weather_features": not any("weather" in column.lower() for column in FEATURE_COLUMNS),
        "no_target_columns_in_features": not any(
            "target" in column.lower() or "actual" in column.lower() for column in FEATURE_COLUMNS
        ),
    }
    errors = [f"Failed check: {name}" for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "feature_rows": len(feature_rows),
        "race_count": len({row["race_id"] for row in feature_rows}),
        "feature_count": len(FEATURE_COLUMNS),
        "checks": checks,
        "errors": errors,
    }


def generate(
    pre_weekend_dir: Path,
    results_file: Path,
    weekend_dir: Path,
    output_dir: Path,
    grid_snapshot: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    pre_weekend_file = pre_weekend_dir / "pre_weekend_features.csv"
    pre_weekend_hash = sha256_file(pre_weekend_file)
    with (pre_weekend_dir / "feature_manifest.json").open("r", encoding="utf-8") as handle:
        pre_weekend_manifest = json.load(handle)
    pre_weekend_rows = read_csv(pre_weekend_file)
    target_rows = read_csv(pre_weekend_dir / "race_targets.csv")
    result_rows = read_csv(results_file)
    qualifying_rows = read_csv(weekend_dir / "qualifying_grid.csv")
    if grid_snapshot:
        apply_grid_snapshot(qualifying_rows, grid_snapshot)
    practice_rows = read_csv(weekend_dir / "practice_summary.csv")
    profile_rows = read_csv(weekend_dir / "circuit_profiles.csv")
    feature_rows = build_features(
        pre_weekend_rows,
        result_rows,
        qualifying_rows,
        practice_rows,
        profile_rows,
    )
    report = validate(
        result_rows,
        feature_rows,
        qualifying_rows,
        practice_rows,
        pre_weekend_hash,
        pre_weekend_file,
    )
    write_json(output_dir / "validation_report.json", report)
    if report["status"] != "passed":
        raise ValueError("Post-qualifying feature validation failed.")
    write_csv(
        output_dir / "post_qualifying_features.csv",
        feature_rows,
        FEATURE_OUTPUT_COLUMNS,
    )
    write_csv(output_dir / "race_targets.csv", target_rows, TARGET_COLUMNS)
    write_json(
        output_dir / "feature_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pre_weekend_dataset": {
                "path": portable_path(pre_weekend_file, PROJECT_ROOT),
                "sha256": pre_weekend_hash,
            },
            "weekend_dataset": {
                "path": portable_path(weekend_dir, PROJECT_ROOT),
                "manifest_sha256": sha256_file(weekend_dir / "manifest.json"),
            },
            "results_dataset": {
                "path": portable_path(results_file, PROJECT_ROOT),
                "sha256": sha256_file(results_file),
            },
            "feature_columns": FEATURE_COLUMNS,
            "season_weights": pre_weekend_manifest["season_weights"],
            "feature_groups": {
                "pre_weekend": list(PRE_WEEKEND_FEATURE_COLUMNS),
                "qualifying": QUALIFYING_FEATURES,
                "practice": PRACTICE_FEATURES,
                "reliability": RELIABILITY_FEATURES,
                "circuit_profile": PROFILE_FEATURES,
            },
            "cutoff": "Final recorded grid after qualifying and known grid changes.",
            "weather_included": False,
            "grid_snapshot": (
                {
                    "path": portable_path(grid_snapshot, PROJECT_ROOT),
                    "sha256": sha256_file(grid_snapshot),
                }
                if grid_snapshot
                else None
            ),
            "fallbacks": {
                "qualifying": "Official grid position, field-end position, and neutral pace values.",
                "practice": "Neutral 0.5 score with a missing-data flag.",
                "reliability": "Earlier field non-finish rate, or zero before any race.",
                "circuit_profile": "Neutral 0.5 values and constructor recent form.",
                "penalties": "The recorded official grid is used without guessing an unverified cause.",
            },
            "dnf_handling": (
                "Non-finishes remain excluded from driver pace. They appear only in separately "
                "named reliability features."
            ),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate separate post-qualifying features.")
    parser.add_argument("--pre-weekend-dir", type=Path, default=DEFAULT_PRE_WEEKEND_DIR)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--weekend-dir", type=Path, default=DEFAULT_WEEKEND_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--grid-snapshot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate(
        args.pre_weekend_dir.resolve(),
        args.results_file.resolve(),
        args.weekend_dir.resolve(),
        args.output_root.resolve() / args.dataset_id,
        args.grid_snapshot.resolve() if args.grid_snapshot else None,
    )
    print(output)


if __name__ == "__main__":
    main()
