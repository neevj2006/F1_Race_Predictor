from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from f1_race_predictor.artifacts import portable_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "fastf1"
FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "fastf1"

SEASON_WEIGHTS = {2023: 0.25, 2024: 0.50, 2025: 0.75, 2026: 1.00}
RECENT_WINDOWS: tuple[int, ...] = (5, 10)
CIRCUIT_PRIOR_STRENGTH = 2.0
NEUTRAL_PERFORMANCE = 0.5
NEUTRAL_TEAMMATE_DELTA = 0.0

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
]

FEATURE_COLUMNS = [
    "driver_prior_starts",
    "driver_prior_seasons",
    "driver_prior_classified_finishes",
    "driver_recent_finish_score_5",
    "driver_recent_finish_score_10",
    "driver_weighted_finish_score",
    "driver_recent_teammate_delta_5",
    "driver_recent_teammate_delta_10",
    "driver_weighted_teammate_delta",
    "driver_circuit_prior_starts",
    "driver_circuit_classified_finishes",
    "driver_circuit_teammate_score",
    "constructor_prior_races",
    "constructor_recent_finish_score_5",
    "constructor_recent_finish_score_10",
    "constructor_weighted_finish_score",
    "constructor_recent_dnf_rate_10",
    "constructor_weighted_dnf_rate",
    "constructor_circuit_prior_races",
    "constructor_circuit_finish_score",
]

TRAINING_COLUMNS = [
    "training_season_weight",
    "target_normalized_position",
    "target_official_position",
    "target_classification_status",
]

FEATURE_OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
TARGET_OUTPUT_COLUMNS = ["race_id", "driver_id"] + TRAINING_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pre-weekend race features.")
    parser.add_argument(
        "--retrieval",
        help="Processed FastF1 retrieval identifier. The latest passing dataset is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to data/features/fastf1/<retrieval>.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Explicit normalized race-results CSV. Takes precedence over --retrieval.",
    )
    parser.add_argument(
        "--season-weights",
        default="2023=0.25,2024=0.50,2025=0.75,2026=1.00",
        help="Comma-separated season=weight values.",
    )
    parser.add_argument("--recent-windows", default="5,10")
    parser.add_argument("--circuit-prior-strength", type=float, default=2.0)
    return parser.parse_args()


def parse_season_weights(value: str) -> dict[int, float]:
    weights = {
        int(item.split("=", 1)[0].strip()): float(item.split("=", 1)[1].strip())
        for item in value.split(",")
        if item.strip()
    }
    ordered = [weights[season] for season in sorted(weights)]
    if any(weight <= 0 for weight in ordered):
        raise ValueError("Season weights must be positive.")
    if any(left >= right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Season weights must increase toward the most recent season.")
    return weights


def configure_feature_settings(
    season_weights: dict[int, float],
    recent_windows: tuple[int, ...],
    circuit_prior_strength: float,
) -> None:
    global SEASON_WEIGHTS, RECENT_WINDOWS, CIRCUIT_PRIOR_STRENGTH
    if not recent_windows or any(window <= 0 for window in recent_windows):
        raise ValueError("Recent windows must contain positive integers.")
    if circuit_prior_strength < 0:
        raise ValueError("Circuit prior strength cannot be negative.")
    SEASON_WEIGHTS = season_weights
    RECENT_WINDOWS = recent_windows
    CIRCUIT_PRIOR_STRENGTH = circuit_prior_strength


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def latest_passing_dataset() -> Path:
    candidates = []
    for path in PROCESSED_ROOT.iterdir():
        report_path = path / "validation_report.json"
        if path.is_dir() and report_path.exists():
            report = read_json(report_path)
            if report.get("status") == "passed":
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No passing processed FastF1 dataset was found.")
    return max(candidates, key=lambda path: path.name)


def choose_dataset(retrieval_id: str | None) -> Path:
    if retrieval_id:
        path = PROCESSED_ROOT / retrieval_id
        if not path.is_dir():
            raise FileNotFoundError(f"Processed FastF1 dataset does not exist: {retrieval_id}")
        return path
    return latest_passing_dataset()


def load_results(path: Path) -> list[dict[str, Any]]:
    integer_columns = {
        "season",
        "round",
        "official_position",
        "normalized_position",
        "laps_completed",
    }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw_row)
            for column in integer_columns:
                row[column] = int(row[column])
            rows.append(row)
    return sorted(
        rows, key=lambda row: (row["race_date"], row["season"], row["round"], row["driver_id"])
    )


def mean_or(values: Iterable[float], fallback: float) -> float:
    values = list(values)
    return sum(values) / len(values) if values else fallback


def weighted_mean_or(
    values: Iterable[tuple[float, float]],
    fallback: float,
) -> float:
    values = list(values)
    total_weight = sum(weight for _, weight in values)
    if not values or total_weight == 0:
        return fallback
    return sum(value * weight for value, weight in values) / total_weight


def finishing_score(row: dict[str, Any], field_size: int) -> float:
    if field_size <= 1:
        return NEUTRAL_PERFORMANCE
    return 1.0 - ((row["normalized_position"] - 1) / (field_size - 1))


def race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_race_id: str | None = None
    for row in rows:
        if row["race_id"] != current_race_id:
            groups.append([])
            current_race_id = row["race_id"]
        groups[-1].append(row)
    return groups


def recent_mean(
    records: list[dict[str, Any]],
    value_key: str,
    window: int,
    fallback: float,
) -> float:
    values = [record[value_key] for record in records if record.get(value_key) is not None]
    return mean_or(values[-window:], fallback)


def weighted_record_mean(
    records: list[dict[str, Any]],
    value_key: str,
    fallback: float,
) -> float:
    values = [
        (record[value_key], SEASON_WEIGHTS[record["season"]])
        for record in records
        if record.get(value_key) is not None
    ]
    return weighted_mean_or(values, fallback)


def constructor_dnf_rate(
    records: list[dict[str, Any]],
    fallback: float,
    window: int | None = None,
) -> float:
    selected = records[-window:] if window else records
    starts = sum(record["starts"] for record in selected)
    if starts == 0:
        return fallback
    return sum(record["dnfs"] for record in selected) / starts


def weighted_constructor_dnf_rate(
    records: list[dict[str, Any]],
    fallback: float,
) -> float:
    weighted_starts = sum(record["starts"] * SEASON_WEIGHTS[record["season"]] for record in records)
    if weighted_starts == 0:
        return fallback
    weighted_dnfs = sum(record["dnfs"] * SEASON_WEIGHTS[record["season"]] for record in records)
    return weighted_dnfs / weighted_starts


def build_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    driver_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    driver_circuit_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    constructor_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    constructor_circuit_history: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    global_classified_scores: list[float] = []
    global_constructor_scores: list[float] = []
    global_starts = 0
    global_dnfs = 0
    feature_rows: list[dict[str, Any]] = []

    for race_rows in race_groups(rows):
        field_size = len(race_rows)
        global_driver_fallback = mean_or(global_classified_scores, NEUTRAL_PERFORMANCE)
        global_constructor_fallback = mean_or(global_constructor_scores, NEUTRAL_PERFORMANCE)
        global_dnf_fallback = global_dnfs / global_starts if global_starts else 0.0

        for row in race_rows:
            driver_records = driver_history[row["driver_id"]]
            driver_circuit_records = driver_circuit_history[(row["driver_id"], row["circuit_id"])]
            constructor_records = constructor_history[row["constructor_id"]]
            constructor_circuit_records = constructor_circuit_history[
                (row["constructor_id"], row["circuit_id"])
            ]

            classified_driver_records = [
                record for record in driver_records if record["finish_score"] is not None
            ]
            driver_weighted_finish = weighted_record_mean(
                classified_driver_records,
                "finish_score",
                global_driver_fallback,
            )
            teammate_records = [
                record for record in driver_records if record["teammate_delta"] is not None
            ]
            driver_weighted_teammate = weighted_record_mean(
                teammate_records,
                "teammate_delta",
                NEUTRAL_TEAMMATE_DELTA,
            )
            circuit_teammate_values = [
                (record["teammate_delta"], SEASON_WEIGHTS[record["season"]])
                for record in driver_circuit_records
                if record["teammate_delta"] is not None
            ]
            circuit_teammate_weight = sum(weight for _, weight in circuit_teammate_values)
            driver_circuit_teammate = (
                sum(value * weight for value, weight in circuit_teammate_values)
                + CIRCUIT_PRIOR_STRENGTH * driver_weighted_teammate
            ) / (circuit_teammate_weight + CIRCUIT_PRIOR_STRENGTH)

            constructor_weighted_finish = weighted_record_mean(
                constructor_records,
                "finish_score",
                global_constructor_fallback,
            )
            circuit_constructor_values = [
                (record["finish_score"], SEASON_WEIGHTS[record["season"]])
                for record in constructor_circuit_records
                if record["finish_score"] is not None
            ]
            circuit_constructor_weight = sum(weight for _, weight in circuit_constructor_values)
            constructor_circuit_finish = (
                sum(value * weight for value, weight in circuit_constructor_values)
                + CIRCUIT_PRIOR_STRENGTH * constructor_weighted_finish
            ) / (circuit_constructor_weight + CIRCUIT_PRIOR_STRENGTH)

            feature_rows.append(
                {
                    "race_id": row["race_id"],
                    "season": row["season"],
                    "round": row["round"],
                    "race_date": row["race_date"],
                    "circuit_id": row["circuit_id"],
                    "driver_id": row["driver_id"],
                    "driver_name": row["driver_name"],
                    "constructor_id": row["constructor_id"],
                    "constructor_name": row["constructor_name"],
                    "driver_prior_starts": sum(record["started"] for record in driver_records),
                    "driver_prior_seasons": len(
                        {record["season"] for record in driver_records if record["started"]}
                    ),
                    "driver_prior_classified_finishes": len(classified_driver_records),
                    "driver_recent_finish_score_5": recent_mean(
                        classified_driver_records,
                        "finish_score",
                        5,
                        global_driver_fallback,
                    ),
                    "driver_recent_finish_score_10": recent_mean(
                        classified_driver_records,
                        "finish_score",
                        10,
                        global_driver_fallback,
                    ),
                    "driver_weighted_finish_score": driver_weighted_finish,
                    "driver_recent_teammate_delta_5": recent_mean(
                        teammate_records,
                        "teammate_delta",
                        5,
                        NEUTRAL_TEAMMATE_DELTA,
                    ),
                    "driver_recent_teammate_delta_10": recent_mean(
                        teammate_records,
                        "teammate_delta",
                        10,
                        NEUTRAL_TEAMMATE_DELTA,
                    ),
                    "driver_weighted_teammate_delta": driver_weighted_teammate,
                    "driver_circuit_prior_starts": sum(
                        record["started"] for record in driver_circuit_records
                    ),
                    "driver_circuit_classified_finishes": sum(
                        record["finish_score"] is not None for record in driver_circuit_records
                    ),
                    "driver_circuit_teammate_score": driver_circuit_teammate,
                    "constructor_prior_races": len(constructor_records),
                    "constructor_recent_finish_score_5": recent_mean(
                        constructor_records,
                        "finish_score",
                        5,
                        global_constructor_fallback,
                    ),
                    "constructor_recent_finish_score_10": recent_mean(
                        constructor_records,
                        "finish_score",
                        10,
                        global_constructor_fallback,
                    ),
                    "constructor_weighted_finish_score": constructor_weighted_finish,
                    "constructor_recent_dnf_rate_10": constructor_dnf_rate(
                        constructor_records,
                        global_dnf_fallback,
                        10,
                    ),
                    "constructor_weighted_dnf_rate": weighted_constructor_dnf_rate(
                        constructor_records,
                        global_dnf_fallback,
                    ),
                    "constructor_circuit_prior_races": len(constructor_circuit_records),
                    "constructor_circuit_finish_score": constructor_circuit_finish,
                    "training_season_weight": SEASON_WEIGHTS[row["season"]],
                    "target_normalized_position": row["normalized_position"],
                    "target_official_position": row["official_position"],
                    "target_classification_status": row["classification_status"],
                }
            )

        results_by_constructor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in race_rows:
            row["_finish_score"] = finishing_score(row, field_size)
            results_by_constructor[row["constructor_id"]].append(row)

        teammate_delta_by_driver: dict[str, float] = {}
        for constructor_rows in results_by_constructor.values():
            classified_rows = [
                row for row in constructor_rows if row["classification_status"] == "CLASSIFIED"
            ]
            if len(classified_rows) == 2:
                first, second = classified_rows
                teammate_delta_by_driver[first["driver_id"]] = (
                    first["_finish_score"] - second["_finish_score"]
                )
                teammate_delta_by_driver[second["driver_id"]] = (
                    second["_finish_score"] - first["_finish_score"]
                )

        for row in race_rows:
            started = row["classification_status"] != "DNS"
            classified = row["classification_status"] == "CLASSIFIED"
            record = {
                "season": row["season"],
                "race_id": row["race_id"],
                "started": started,
                "finish_score": row["_finish_score"] if classified else None,
                "teammate_delta": teammate_delta_by_driver.get(row["driver_id"]),
            }
            driver_history[row["driver_id"]].append(record)
            driver_circuit_history[(row["driver_id"], row["circuit_id"])].append(record)
            if classified:
                global_classified_scores.append(row["_finish_score"])
            if started:
                global_starts += 1
            if row["classification_status"] == "DNF":
                global_dnfs += 1

        for constructor_id, constructor_rows in results_by_constructor.items():
            classified_scores = [
                row["_finish_score"]
                for row in constructor_rows
                if row["classification_status"] == "CLASSIFIED"
            ]
            team_finish_score = (
                sum(classified_scores) / len(classified_scores) if classified_scores else None
            )
            team_record = {
                "season": constructor_rows[0]["season"],
                "race_id": constructor_rows[0]["race_id"],
                "finish_score": team_finish_score,
                "starts": sum(row["classification_status"] != "DNS" for row in constructor_rows),
                "dnfs": sum(row["classification_status"] == "DNF" for row in constructor_rows),
            }
            constructor_history[constructor_id].append(team_record)
            constructor_circuit_history[(constructor_id, constructor_rows[0]["circuit_id"])].append(
                team_record
            )
            if team_finish_score is not None:
                global_constructor_scores.append(team_finish_score)

        for row in race_rows:
            del row["_finish_score"]

    return feature_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_signature(rows: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[Any, ...]]:
    return {
        (row["race_id"], row["driver_id"]): tuple(row[column] for column in FEATURE_COLUMNS)
        for row in rows
    }


def validate_features(
    source_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    source_keys = {(row["race_id"], row["driver_id"]) for row in source_rows}
    feature_keys = {(row["race_id"], row["driver_id"]) for row in feature_rows}

    checks["row_count_matches_source"] = len(source_rows) == len(feature_rows)
    checks["one_row_per_driver_race"] = len(feature_keys) == len(feature_rows)
    checks["driver_race_keys_match_source"] = source_keys == feature_keys
    checks["feature_values_complete"] = all(
        row[column] is not None for row in feature_rows for column in FEATURE_COLUMNS
    )
    checks["no_post_cutoff_columns_in_features"] = not any(
        forbidden in column.lower()
        for column in FEATURE_COLUMNS
        for forbidden in ("grid", "qualifying", "practice", "target", "official_position", "result")
    )

    first_race_id = feature_rows[0]["race_id"]
    first_race_rows = [row for row in feature_rows if row["race_id"] == first_race_id]
    checks["first_race_uses_neutral_fallbacks"] = all(
        row["driver_prior_starts"] == 0
        and row["driver_recent_finish_score_5"] == NEUTRAL_PERFORMANCE
        and row["driver_weighted_teammate_delta"] == NEUTRAL_TEAMMATE_DELTA
        and row["constructor_prior_races"] == 0
        for row in first_race_rows
    )

    grouped = race_groups(source_rows)
    cutoff_index = max(1, (len(grouped) * 3) // 4)
    prefix_source = [row for group in grouped[:cutoff_index] for row in group]
    prefix_features = build_features(copy.deepcopy(prefix_source))
    full_signature = feature_signature(feature_rows)
    prefix_signature = feature_signature(prefix_features)
    checks["truncating_future_races_keeps_earlier_features"] = all(
        full_signature[key] == signature for key, signature in prefix_signature.items()
    )

    changed_source = copy.deepcopy(source_rows)
    future_race_ids = {group[0]["race_id"] for group in grouped[cutoff_index:]}
    for row in changed_source:
        if row["race_id"] in future_race_ids:
            row["normalized_position"] = 1
            row["official_position"] = 1
            row["classification_status"] = "DNF"
    changed_features = build_features(changed_source)
    changed_signature = feature_signature(changed_features)
    prefix_keys = set(prefix_signature)
    checks["changing_future_results_keeps_earlier_features"] = all(
        full_signature[key] == changed_signature[key] for key in prefix_keys
    )

    current_race_index = len(grouped) // 2
    current_race_id = grouped[current_race_index][0]["race_id"]
    changed_current_source = copy.deepcopy(source_rows)
    for row in changed_current_source:
        if row["race_id"] == current_race_id:
            row["normalized_position"] = 1
            row["official_position"] = 1
            row["classification_status"] = "DNF"
    changed_current_features = build_features(changed_current_source)
    changed_current_signature = feature_signature(changed_current_features)
    current_keys = {key for key in full_signature if key[0] == current_race_id}
    checks["current_race_result_does_not_enter_its_features"] = all(
        full_signature[key] == changed_current_signature[key] for key in current_keys
    )

    dnf_source_rows = [row for row in source_rows if row["classification_status"] == "DNF"]
    checks["dnfs_exist_for_separate_reliability_history"] = len(dnf_source_rows) > 0
    checks["driver_pace_features_have_no_dnf_field"] = not any(
        "dnf" in column for column in FEATURE_COLUMNS if column.startswith("driver_")
    )

    for name, passed in checks.items():
        if not passed:
            errors.append(f"Failed check: {name}")

    example_keys = [
        (first_race_id, first_race_rows[0]["driver_id"]),
        ("2025-01", "hadjar"),
        (feature_rows[-1]["race_id"], feature_rows[-1]["driver_id"]),
    ]
    rows_by_key = {(row["race_id"], row["driver_id"]): row for row in feature_rows}
    race_order = {
        group[0]["race_id"]: index for index, group in enumerate(race_groups(source_rows))
    }
    examples = []
    for key in example_keys:
        if key not in rows_by_key:
            continue
        feature_row = rows_by_key[key]
        target_order = race_order[feature_row["race_id"]]
        prior_rows = [row for row in source_rows if race_order[row["race_id"]] < target_order]
        prior_driver_rows = [
            row for row in prior_rows if row["driver_id"] == feature_row["driver_id"]
        ]
        prior_constructor_rows = [
            row for row in prior_rows if row["constructor_id"] == feature_row["constructor_id"]
        ]
        examples.append(
            {
                "feature_row": feature_row,
                "history_trace": {
                    "recent_classified_driver_races": [
                        row["race_id"]
                        for row in prior_driver_rows
                        if row["classification_status"] == "CLASSIFIED"
                    ][-10:],
                    "prior_driver_circuit_races": [
                        row["race_id"]
                        for row in prior_driver_rows
                        if row["circuit_id"] == feature_row["circuit_id"]
                        and row["classification_status"] != "DNS"
                    ],
                    "recent_constructor_races": list(
                        dict.fromkeys(row["race_id"] for row in prior_constructor_rows)
                    )[-10:],
                    "recent_constructor_dnf_races": list(
                        dict.fromkeys(
                            row["race_id"]
                            for row in prior_constructor_rows
                            if row["classification_status"] == "DNF"
                        )
                    )[-10:],
                },
            }
        )

    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source_row_count": len(source_rows),
        "feature_row_count": len(feature_rows),
        "race_count": len({row["race_id"] for row in feature_rows}),
        "feature_count": len(FEATURE_COLUMNS),
        "checks": checks,
        "errors": errors,
        "representative_rows": examples,
    }


def generate_feature_dataset(source_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_results(source_path)
    feature_rows = build_features(copy.deepcopy(source_rows))
    report = validate_features(source_rows, feature_rows)

    write_json(output_dir / "validation_report.json", report)
    if report["status"] != "passed":
        raise ValueError("Pre-weekend feature validation failed.")

    write_csv(
        output_dir / "pre_weekend_features.csv",
        feature_rows,
        FEATURE_OUTPUT_COLUMNS,
    )
    write_csv(
        output_dir / "race_targets.csv",
        feature_rows,
        TARGET_OUTPUT_COLUMNS,
    )
    write_json(
        output_dir / "feature_manifest.json",
        {
            "source_dataset": portable_path(source_path, PROJECT_ROOT),
            "feature_dataset": portable_path(output_dir / "pre_weekend_features.csv", PROJECT_ROOT),
            "target_dataset": portable_path(output_dir / "race_targets.csv", PROJECT_ROOT),
            "season_weights": {str(year): weight for year, weight in SEASON_WEIGHTS.items()},
            "recent_windows": list(RECENT_WINDOWS),
            "circuit_prior_strength": CIRCUIT_PRIOR_STRENGTH,
            "identifier_columns": IDENTIFIER_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "training_columns": TRAINING_COLUMNS,
            "dnf_handling": (
                "DNFs are excluded from driver pace and teammate features. They are retained only "
                "in constructor reliability history and training targets."
            ),
            "fallbacks": {
                "driver_performance": "Prior field mean, or 0.5 when no earlier race exists.",
                "driver_teammate_delta": "Neutral value of 0.0.",
                "constructor_performance": "Prior field mean, or 0.5 when no earlier race exists.",
                "constructor_dnf_rate": "Prior field DNF rate, or 0.0 when no earlier race exists.",
            },
        },
    )
    return report


def main() -> None:
    args = parse_args()
    if args.input_file:
        source_path = args.input_file.resolve()
        dataset_name = source_path.parent.name
    else:
        dataset_dir = choose_dataset(args.retrieval)
        source_path = dataset_dir / "race_results.csv"
        dataset_name = dataset_dir.name
    output_dir = args.output_dir or FEATURE_ROOT / dataset_name
    configure_feature_settings(
        parse_season_weights(args.season_weights),
        tuple(int(value.strip()) for value in args.recent_windows.split(",") if value.strip()),
        args.circuit_prior_strength,
    )
    report = generate_feature_dataset(source_path, output_dir.resolve())
    print(
        f"Wrote {report['feature_row_count']} feature rows to "
        f"{output_dir / 'pre_weekend_features.csv'}"
    )
    print(f"Wrote {report['feature_row_count']} target rows to {output_dir / 'race_targets.csv'}")
    print(f"Validation status: {report['status']}")


if __name__ == "__main__":
    main()
