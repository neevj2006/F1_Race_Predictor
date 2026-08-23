from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "fastf1"
EVALUATION_ROOT = PROJECT_ROOT / "data" / "evaluations" / "baselines"

HELD_OUT_SEASON = 2026
SEASON_WEIGHTS = {2023: 0.25, 2024: 0.50, 2025: 0.75, 2026: 1.00}
NEUTRAL_DRIVER_SCORE = 0.5
NEUTRAL_TEAMMATE_DELTA = 0.0
CONSTRUCTOR_SHARE = 0.70
DRIVER_SHARE = 0.30

BASELINES = [
    "previous_race_order",
    "driver_form_unweighted",
    "driver_form_season_weighted",
    "driver_form_recent_5",
    "driver_form_recent_10",
    "constructor_teammate_unweighted",
    "constructor_teammate_season_weighted",
    "constructor_teammate_recent_5",
    "constructor_teammate_recent_10",
]

PREDICTION_COLUMNS = [
    "baseline",
    "race_id",
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "baseline_score",
    "predicted_position",
    "actual_position",
    "actual_classification_status",
    "history_race_count",
    "history_latest_race_id",
    "history_latest_race_date",
]

RACE_SCORE_COLUMNS = [
    "baseline",
    "race_id",
    "season",
    "round",
    "race_date",
    "field_size",
    "spearman_correlation",
    "mean_absolute_position_error",
    "kendall_correlation",
    "winner_accuracy",
    "top_three_overlap",
]

SUMMARY_COLUMNS = [
    "baseline",
    "held_out_season",
    "races",
    "average_spearman_correlation",
    "average_mean_absolute_position_error",
    "average_kendall_correlation",
    "winner_accuracy",
    "average_top_three_overlap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate chronological race-order baselines.")
    parser.add_argument(
        "--retrieval",
        help="Feature retrieval identifier. The latest passing feature dataset is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to data/evaluations/baselines/<retrieval>.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def latest_passing_feature_dataset() -> Path:
    candidates = []
    for path in FEATURE_ROOT.iterdir():
        report_path = path / "validation_report.json"
        if path.is_dir() and report_path.exists():
            report = read_json(report_path)
            if report.get("status") == "passed":
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No passing feature dataset was found.")
    return max(candidates, key=lambda path: path.name)


def choose_feature_dataset(retrieval_id: str | None) -> Path:
    if retrieval_id:
        path = FEATURE_ROOT / retrieval_id
        if not path.is_dir():
            raise FileNotFoundError(f"Feature dataset does not exist: {retrieval_id}")
        return path
    return latest_passing_feature_dataset()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    features = load_csv(dataset_dir / "pre_weekend_features.csv")
    targets = load_csv(dataset_dir / "race_targets.csv")
    targets_by_key = {(row["race_id"], row["driver_id"]): row for row in targets}
    if len(targets_by_key) != len(targets):
        raise ValueError("Duplicate driver-race target rows were found.")

    rows = []
    for feature_row in features:
        key = (feature_row["race_id"], feature_row["driver_id"])
        if key not in targets_by_key:
            raise ValueError(f"Missing target row for {key}.")
        target = targets_by_key[key]
        rows.append(
            {
                **feature_row,
                "season": int(feature_row["season"]),
                "round": int(feature_row["round"]),
                "actual_position": int(target["target_normalized_position"]),
                "actual_official_position": int(target["target_official_position"]),
                "actual_classification_status": target["target_classification_status"],
            }
        )
    if len(rows) != len(targets):
        raise ValueError("Feature and target row counts do not match.")
    return sorted(
        rows,
        key=lambda row: (row["race_date"], row["season"], row["round"], row["driver_id"]),
    )


def race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_race_id: str | None = None
    for row in rows:
        if row["race_id"] != current_race_id:
            groups.append([])
            current_race_id = row["race_id"]
        groups[-1].append(row)
    return groups


def mean_or(values: Iterable[float], fallback: float) -> float:
    values = list(values)
    return sum(values) / len(values) if values else fallback


def weighted_mean_or(
    records: list[tuple[int, float]],
    fallback: float,
) -> float:
    total_weight = sum(SEASON_WEIGHTS[season] for season, _ in records)
    if total_weight == 0:
        return fallback
    return sum(SEASON_WEIGHTS[season] * value for season, value in records) / total_weight


def history_scores(
    records: list[tuple[int, float]],
    fallback: float,
) -> dict[str, float]:
    values = [value for _, value in records]
    return {
        "unweighted": mean_or(values, fallback),
        "season_weighted": weighted_mean_or(records, fallback),
        "recent_5": mean_or(values[-5:], fallback),
        "recent_10": mean_or(values[-10:], fallback),
    }


def finish_score(position: int, field_size: int) -> float:
    if field_size <= 1:
        return NEUTRAL_DRIVER_SCORE
    return 1.0 - ((position - 1) / (field_size - 1))


def scaled_teammate_score(delta: float) -> float:
    return min(1.0, max(0.0, 0.5 + delta / 2.0))


def baseline_scores_for_driver(
    row: dict[str, Any],
    previous_driver_score: dict[str, float],
    driver_history: dict[str, list[tuple[int, float]]],
    teammate_history: dict[str, list[tuple[int, float]]],
    constructor_history: dict[str, list[tuple[int, float]]],
    global_constructor_history: list[float],
) -> dict[str, float]:
    driver_scores = history_scores(
        driver_history[row["driver_id"]],
        NEUTRAL_DRIVER_SCORE,
    )
    teammate_scores = history_scores(
        teammate_history[row["driver_id"]],
        NEUTRAL_TEAMMATE_DELTA,
    )
    constructor_fallback = mean_or(global_constructor_history, NEUTRAL_DRIVER_SCORE)
    constructor_scores = history_scores(
        constructor_history[row["constructor_id"]],
        constructor_fallback,
    )

    scores = {
        "previous_race_order": previous_driver_score.get(
            row["driver_id"],
            NEUTRAL_DRIVER_SCORE,
        ),
        "driver_form_unweighted": driver_scores["unweighted"],
        "driver_form_season_weighted": driver_scores["season_weighted"],
        "driver_form_recent_5": driver_scores["recent_5"],
        "driver_form_recent_10": driver_scores["recent_10"],
    }
    for version in ("unweighted", "season_weighted", "recent_5", "recent_10"):
        scores[f"constructor_teammate_{version}"] = (
            CONSTRUCTOR_SHARE * constructor_scores[version]
            + DRIVER_SHARE * scaled_teammate_score(teammate_scores[version])
        )
    return scores


def rank_predictions(
    race_rows: list[dict[str, Any]],
    scores_by_driver: dict[str, dict[str, float]],
    history_race_count: int,
    history_latest_race_id: str,
    history_latest_race_date: str,
) -> list[dict[str, Any]]:
    predictions = []
    for baseline in BASELINES:
        ordered = sorted(
            race_rows,
            key=lambda row: (-scores_by_driver[row["driver_id"]][baseline], row["driver_id"]),
        )
        for position, row in enumerate(ordered, start=1):
            predictions.append(
                {
                    "baseline": baseline,
                    "race_id": row["race_id"],
                    "season": row["season"],
                    "round": row["round"],
                    "race_date": row["race_date"],
                    "circuit_id": row["circuit_id"],
                    "driver_id": row["driver_id"],
                    "driver_name": row["driver_name"],
                    "constructor_id": row["constructor_id"],
                    "constructor_name": row["constructor_name"],
                    "baseline_score": scores_by_driver[row["driver_id"]][baseline],
                    "predicted_position": position,
                    "actual_position": row["actual_position"],
                    "actual_classification_status": row["actual_classification_status"],
                    "history_race_count": history_race_count,
                    "history_latest_race_id": history_latest_race_id,
                    "history_latest_race_date": history_latest_race_date,
                }
            )
    return predictions


def generate_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_driver_score: dict[str, float] = {}
    driver_history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    teammate_history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    constructor_history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    global_constructor_history: list[float] = []
    predictions: list[dict[str, Any]] = []
    history_race_count = 0
    history_latest_race_id = ""
    history_latest_race_date = ""

    for race_rows in race_groups(rows):
        field_size = len(race_rows)
        scores_by_driver = {
            row["driver_id"]: baseline_scores_for_driver(
                row,
                previous_driver_score,
                driver_history,
                teammate_history,
                constructor_history,
                global_constructor_history,
            )
            for row in race_rows
        }
        if race_rows[0]["season"] == HELD_OUT_SEASON:
            predictions.extend(
                rank_predictions(
                    race_rows,
                    scores_by_driver,
                    history_race_count,
                    history_latest_race_id,
                    history_latest_race_date,
                )
            )

        rows_by_constructor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in race_rows:
            row["_finish_score"] = finish_score(row["actual_position"], field_size)
            rows_by_constructor[row["constructor_id"]].append(row)
            previous_driver_score[row["driver_id"]] = row["_finish_score"]
            if row["actual_classification_status"] == "CLASSIFIED":
                driver_history[row["driver_id"]].append((row["season"], row["_finish_score"]))

        for constructor_rows in rows_by_constructor.values():
            classified_rows = [
                row
                for row in constructor_rows
                if row["actual_classification_status"] == "CLASSIFIED"
            ]
            if len(classified_rows) == 2:
                first, second = classified_rows
                teammate_history[first["driver_id"]].append(
                    (first["season"], first["_finish_score"] - second["_finish_score"])
                )
                teammate_history[second["driver_id"]].append(
                    (second["season"], second["_finish_score"] - first["_finish_score"])
                )
            if classified_rows:
                team_score = mean_or(
                    (row["_finish_score"] for row in classified_rows),
                    NEUTRAL_DRIVER_SCORE,
                )
                constructor_id = constructor_rows[0]["constructor_id"]
                constructor_history[constructor_id].append(
                    (constructor_rows[0]["season"], team_score)
                )
                global_constructor_history.append(team_score)

        for row in race_rows:
            del row["_finish_score"]
        history_race_count += 1
        history_latest_race_id = race_rows[0]["race_id"]
        history_latest_race_date = race_rows[0]["race_date"]

    return predictions


def spearman_correlation(predicted: list[int], actual: list[int]) -> float:
    count = len(predicted)
    if count < 2:
        return 0.0
    squared_difference = sum((left - right) ** 2 for left, right in zip(predicted, actual))
    return 1.0 - (6.0 * squared_difference) / (count * (count**2 - 1))


def kendall_correlation(predicted: list[int], actual: list[int]) -> float:
    concordant = 0
    discordant = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_order = predicted[left] - predicted[right]
            actual_order = actual[left] - actual[right]
            if predicted_order * actual_order > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def score_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[(row["baseline"], row["race_id"])].append(row)

    scores = []
    for (baseline, race_id), group in groups.items():
        ordered = sorted(group, key=lambda row: row["driver_id"])
        predicted = [row["predicted_position"] for row in ordered]
        actual = [row["actual_position"] for row in ordered]
        predicted_winner = min(group, key=lambda row: row["predicted_position"])["driver_id"]
        actual_winner = min(group, key=lambda row: row["actual_position"])["driver_id"]
        predicted_top_three = {
            row["driver_id"] for row in group if row["predicted_position"] <= 3
        }
        actual_top_three = {row["driver_id"] for row in group if row["actual_position"] <= 3}
        scores.append(
            {
                "baseline": baseline,
                "race_id": race_id,
                "season": group[0]["season"],
                "round": group[0]["round"],
                "race_date": group[0]["race_date"],
                "field_size": len(group),
                "spearman_correlation": spearman_correlation(predicted, actual),
                "mean_absolute_position_error": mean_or(
                    (abs(left - right) for left, right in zip(predicted, actual)),
                    0.0,
                ),
                "kendall_correlation": kendall_correlation(predicted, actual),
                "winner_accuracy": float(predicted_winner == actual_winner),
                "top_three_overlap": len(predicted_top_three & actual_top_three) / 3.0,
            }
        )
    return sorted(scores, key=lambda row: (row["round"], row["baseline"]))


def summarize_scores(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_baseline[row["baseline"]].append(row)
    summaries = []
    for baseline in BASELINES:
        rows = by_baseline[baseline]
        summaries.append(
            {
                "baseline": baseline,
                "held_out_season": HELD_OUT_SEASON,
                "races": len(rows),
                "average_spearman_correlation": mean_or(
                    (row["spearman_correlation"] for row in rows),
                    0.0,
                ),
                "average_mean_absolute_position_error": mean_or(
                    (row["mean_absolute_position_error"] for row in rows),
                    0.0,
                ),
                "average_kendall_correlation": mean_or(
                    (row["kendall_correlation"] for row in rows),
                    0.0,
                ),
                "winner_accuracy": mean_or(
                    (row["winner_accuracy"] for row in rows),
                    0.0,
                ),
                "average_top_three_overlap": mean_or(
                    (row["top_three_overlap"] for row in rows),
                    0.0,
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (-row["average_spearman_correlation"], row["baseline"]),
    )


def prediction_signature(
    predictions: list[dict[str, Any]],
    race_ids: set[str] | None = None,
) -> dict[tuple[str, str, str], tuple[float, int]]:
    return {
        (row["baseline"], row["race_id"], row["driver_id"]): (
            row["baseline_score"],
            row["predicted_position"],
        )
        for row in predictions
        if race_ids is None or row["race_id"] in race_ids
    }


def validate_evaluation(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    race_scores: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    held_out_groups = [
        group for group in race_groups(source_rows) if group[0]["season"] == HELD_OUT_SEASON
    ]
    held_out_race_ids = {group[0]["race_id"] for group in held_out_groups}
    prediction_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[(row["baseline"], row["race_id"])].append(row)

    checks["every_baseline_scores_every_held_out_race"] = (
        len(race_scores) == len(BASELINES) * len(held_out_groups)
        and {(row["baseline"], row["race_id"]) for row in race_scores}
        == {(baseline, race_id) for baseline in BASELINES for race_id in held_out_race_ids}
    )
    checks["every_prediction_contains_the_complete_field"] = all(
        len(prediction_groups[(baseline, group[0]["race_id"])]) == len(group)
        for baseline in BASELINES
        for group in held_out_groups
    )
    checks["predicted_positions_are_complete_and_unique"] = all(
        sorted(row["predicted_position"] for row in group) == list(range(1, len(group) + 1))
        for group in prediction_groups.values()
    )
    checks["actual_positions_are_complete_and_unique"] = all(
        sorted(row["actual_position"] for row in group) == list(range(1, len(group) + 1))
        for group in prediction_groups.values()
    )
    checks["history_cutoff_precedes_each_prediction"] = all(
        row["history_latest_race_date"] < row["race_date"] for row in predictions
    )
    checks["metrics_are_in_valid_ranges"] = all(
        -1.0 <= row["spearman_correlation"] <= 1.0
        and row["mean_absolute_position_error"] >= 0.0
        and -1.0 <= row["kendall_correlation"] <= 1.0
        and row["winner_accuracy"] in (0.0, 1.0)
        and 0.0 <= row["top_three_overlap"] <= 1.0
        for row in race_scores
    )
    checks["one_summary_per_baseline"] = (
        len(summaries) == len(BASELINES)
        and {row["baseline"] for row in summaries} == set(BASELINES)
    )

    repeated_predictions = generate_predictions(copy.deepcopy(source_rows))
    checks["repeated_evaluation_is_identical"] = (
        prediction_signature(predictions) == prediction_signature(repeated_predictions)
    )

    cutoff_index = max(1, len(held_out_groups) // 2)
    earlier_held_out_ids = {
        group[0]["race_id"] for group in held_out_groups[: cutoff_index + 1]
    }
    last_earlier_race = held_out_groups[cutoff_index][0]["race_id"]
    all_groups = race_groups(source_rows)
    truncated_source = [
        row
        for group in all_groups
        for row in group
        if group[0]["race_date"] <= held_out_groups[cutoff_index][0]["race_date"]
    ]
    truncated_predictions = generate_predictions(copy.deepcopy(truncated_source))
    checks["truncating_future_races_keeps_earlier_predictions"] = (
        prediction_signature(predictions, earlier_held_out_ids)
        == prediction_signature(truncated_predictions, earlier_held_out_ids)
    )

    changed_source = copy.deepcopy(source_rows)
    for row in changed_source:
        if row["race_id"] == last_earlier_race:
            row["actual_position"] = 1
            row["actual_classification_status"] = "DNF"
    changed_predictions = generate_predictions(changed_source)
    checks["current_race_result_does_not_change_its_prediction"] = (
        prediction_signature(predictions, {last_earlier_race})
        == prediction_signature(changed_predictions, {last_earlier_race})
    )

    for name, passed in checks.items():
        if not passed:
            errors.append(f"Failed check: {name}")

    strongest = summaries[0] if summaries else None
    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "held_out_season": HELD_OUT_SEASON,
        "held_out_races": len(held_out_groups),
        "baseline_count": len(BASELINES),
        "prediction_row_count": len(predictions),
        "race_score_count": len(race_scores),
        "strongest_baseline": strongest,
        "checks": checks,
        "errors": errors,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = choose_feature_dataset(args.retrieval)
    output_dir = args.output_dir or EVALUATION_ROOT / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(dataset_dir)
    predictions = generate_predictions(copy.deepcopy(source_rows))
    race_scores = score_predictions(predictions)
    summaries = summarize_scores(race_scores)
    report = validate_evaluation(source_rows, predictions, race_scores, summaries)

    write_json(output_dir / "validation_report.json", report)
    if report["status"] != "passed":
        raise ValueError("Baseline evaluation validation failed.")

    write_csv(output_dir / "baseline_predictions.csv", predictions, PREDICTION_COLUMNS)
    write_csv(output_dir / "baseline_race_scores.csv", race_scores, RACE_SCORE_COLUMNS)
    write_csv(output_dir / "baseline_summary.csv", summaries, SUMMARY_COLUMNS)
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "feature_dataset": f"data/features/fastf1/{dataset_dir.name}/pre_weekend_features.csv",
            "target_dataset": f"data/features/fastf1/{dataset_dir.name}/race_targets.csv",
            "held_out_season": HELD_OUT_SEASON,
            "history_start_season": 2023,
            "baselines": {
                "previous_race_order": (
                    "The driver's literal result from their previous appearance, including exceptional results."
                ),
                "driver_form_unweighted": "Mean of all earlier classified finishing scores.",
                "driver_form_season_weighted": (
                    "Mean of all earlier classified finishing scores using the configured season weights."
                ),
                "driver_form_recent_5": "Mean of the five most recent classified finishing scores.",
                "driver_form_recent_10": "Mean of the ten most recent classified finishing scores.",
                "constructor_teammate_unweighted": (
                    "70% constructor form and 30% teammate-adjusted driver form using all earlier races."
                ),
                "constructor_teammate_season_weighted": (
                    "70% constructor form and 30% teammate-adjusted driver form using season weights."
                ),
                "constructor_teammate_recent_5": (
                    "70% constructor form and 30% teammate-adjusted driver form from five recent values."
                ),
                "constructor_teammate_recent_10": (
                    "70% constructor form and 30% teammate-adjusted driver form from ten recent values."
                ),
            },
            "season_weights": {str(year): weight for year, weight in SEASON_WEIGHTS.items()},
            "constructor_share": CONSTRUCTOR_SHARE,
            "driver_share": DRIVER_SHARE,
            "tie_breaker": "driver_id ascending",
            "rookie_driver_fallback": NEUTRAL_DRIVER_SCORE,
            "new_constructor_fallback": "Mean constructor score from earlier races, or 0.5.",
            "primary_metric": "Average Spearman rank correlation across held-out races.",
            "other_metrics": [
                "Mean absolute position error",
                "Kendall rank correlation",
                "Winner accuracy",
                "Top-three overlap",
            ],
        },
    )
    print(f"Evaluated {len(BASELINES)} baselines over {report['held_out_races']} races.")
    print(f"Strongest baseline: {report['strongest_baseline']['baseline']}")
    print(
        "Average Spearman correlation: "
        f"{report['strongest_baseline']['average_spearman_correlation']:.3f}"
    )
    print(f"Validation status: {report['status']}")


if __name__ == "__main__":
    main()
