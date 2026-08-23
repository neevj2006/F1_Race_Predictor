from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "fastf1"
BASELINE_ROOT = PROJECT_ROOT / "data" / "evaluations" / "baselines"
MODEL_EVALUATION_ROOT = PROJECT_ROOT / "data" / "evaluations" / "models"
MODEL_ROOT = PROJECT_ROOT / "models"

SELECTION_SEASON = 2026
SELECTION_ROUNDS = tuple(range(1, 7))
TEST_ROUNDS = tuple(range(7, 12))
RECENT_RACE_COUNT = 20
RANDOM_SEED = 42
MVP_SPEARMAN_TARGET = 0.60
GRADIENT_REQUIRED_TEST_WINS = 3

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
HISTORY_SCHEMES = ("unweighted", "season_weighted", "recent_20")
FEATURE_SETS = ("driver_only", "full")

DRIVER_BASE_FEATURES = [
    "driver_prior_starts",
    "driver_prior_seasons",
    "driver_prior_classified_finishes",
    "driver_recent_finish_score_5",
    "driver_recent_finish_score_10",
    "driver_circuit_prior_starts",
    "driver_circuit_classified_finishes",
]

FULL_BASE_FEATURES = DRIVER_BASE_FEATURES + [
    "driver_recent_teammate_delta_5",
    "driver_recent_teammate_delta_10",
    "constructor_prior_races",
    "constructor_recent_finish_score_5",
    "constructor_recent_finish_score_10",
    "constructor_recent_dnf_rate_10",
    "constructor_circuit_prior_races",
]

DRIVER_WEIGHTED_FEATURES = ["driver_weighted_finish_score"]

FULL_WEIGHTED_FEATURES = DRIVER_WEIGHTED_FEATURES + [
    "driver_weighted_teammate_delta",
    "driver_circuit_teammate_score",
    "constructor_weighted_finish_score",
    "constructor_weighted_dnf_rate",
    "constructor_circuit_finish_score",
]

PREDICTION_COLUMNS = [
    "model_id",
    "model_family",
    "feature_set",
    "history_scheme",
    "race_id",
    "season",
    "round",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "model_score",
    "predicted_position",
    "actual_position",
    "actual_classification_status",
    "absolute_position_error",
]

RACE_SCORE_COLUMNS = [
    "model_id",
    "model_family",
    "feature_set",
    "history_scheme",
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

EXPERIMENT_COLUMNS = [
    "model_id",
    "model_family",
    "feature_set",
    "history_scheme",
    "parameters",
    "selection_races",
    "average_spearman_correlation",
    "average_mean_absolute_position_error",
    "average_kendall_correlation",
    "winner_accuracy",
    "average_top_three_overlap",
]

SUMMARY_COLUMNS = [
    "model_id",
    "model_family",
    "feature_set",
    "history_scheme",
    "test_races",
    "average_spearman_correlation",
    "average_mean_absolute_position_error",
    "average_kendall_correlation",
    "winner_accuracy",
    "average_top_three_overlap",
    "races_beating_ridge",
    "selected_model",
]

SEGMENT_COLUMNS = [
    "model_id",
    "segment_type",
    "segment",
    "driver_rows",
    "mean_absolute_position_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate race ranking models.")
    parser.add_argument(
        "--retrieval",
        help="Feature retrieval identifier. The latest passing feature dataset is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Evaluation directory. Defaults to data/evaluations/models/<retrieval>.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Saved model directory. Defaults to models/<retrieval>.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(dataset_dir / "feature_manifest.json")
    numeric_features = manifest["feature_columns"]
    features = load_csv(dataset_dir / "pre_weekend_features.csv")
    targets = load_csv(dataset_dir / "race_targets.csv")
    targets_by_key = {(row["race_id"], row["driver_id"]): row for row in targets}
    if len(targets_by_key) != len(targets):
        raise ValueError("Duplicate target rows were found.")

    rows = []
    for feature_row in features:
        key = (feature_row["race_id"], feature_row["driver_id"])
        if key not in targets_by_key:
            raise ValueError(f"Missing target for {key}.")
        target = targets_by_key[key]
        row: dict[str, Any] = {
            **feature_row,
            "season": int(feature_row["season"]),
            "round": int(feature_row["round"]),
            "training_season_weight": float(target["training_season_weight"]),
            "actual_position": int(target["target_normalized_position"]),
            "actual_classification_status": target["target_classification_status"],
        }
        for column in numeric_features:
            row[column] = float(feature_row[column])
        rows.append(row)
    if len(rows) != len(targets):
        raise ValueError("Feature and target row counts do not match.")

    rows = sorted(
        rows,
        key=lambda row: (row["race_date"], row["season"], row["round"], row["driver_id"]),
    )
    for group in race_groups(rows):
        field_size = len(group)
        for row in group:
            row["field_size"] = field_size
            row["target_finish_score"] = finish_score(row["actual_position"], field_size)
    return rows


def race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_race_id: str | None = None
    for row in rows:
        if row["race_id"] != current_race_id:
            groups.append([])
            current_race_id = row["race_id"]
        groups[-1].append(row)
    return groups


def finish_score(position: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return 1.0 - ((position - 1) / (field_size - 1))


def mean_or(values: Iterable[float], fallback: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else fallback


def feature_columns(feature_set: str, history_scheme: str) -> list[str]:
    columns = list(DRIVER_BASE_FEATURES if feature_set == "driver_only" else FULL_BASE_FEATURES)
    if history_scheme == "season_weighted":
        columns.extend(
            DRIVER_WEIGHTED_FEATURES if feature_set == "driver_only" else FULL_WEIGHTED_FEATURES
        )
    return columns


def training_rows_before(
    rows: list[dict[str, Any]],
    cutoff_date: str,
    history_scheme: str,
) -> list[dict[str, Any]]:
    earlier = [
        row
        for row in rows
        if row["race_date"] < cutoff_date
        and row["actual_classification_status"] == "CLASSIFIED"
    ]
    if history_scheme != "recent_20":
        return earlier
    race_ids = []
    for row in earlier:
        if row["race_id"] not in race_ids:
            race_ids.append(row["race_id"])
    recent_race_ids = set(race_ids[-RECENT_RACE_COUNT:])
    return [row for row in earlier if row["race_id"] in recent_race_ids]


def final_training_rows(
    rows: list[dict[str, Any]],
    cutoff_date: str,
    history_scheme: str,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["race_date"] <= cutoff_date
        and row["actual_classification_status"] == "CLASSIFIED"
    ]
    if history_scheme != "recent_20":
        return eligible
    race_ids = []
    for row in eligible:
        if row["race_id"] not in race_ids:
            race_ids.append(row["race_id"])
    recent_race_ids = set(race_ids[-RECENT_RACE_COUNT:])
    return [row for row in eligible if row["race_id"] in recent_race_ids]


def build_estimator(model_family: str, parameters: dict[str, Any]) -> Any:
    if model_family == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=parameters["alpha"])),
            ]
        )
    if model_family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            random_state=RANDOM_SEED,
            loss="squared_error",
        )
    raise ValueError(f"Unknown model family: {model_family}")


def fit_estimator(
    train_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> Any:
    columns = config["feature_columns"]
    matrix = np.asarray([[row[column] for column in columns] for row in train_rows], dtype=float)
    target = np.asarray([row["target_finish_score"] for row in train_rows], dtype=float)
    weights = (
        np.asarray([row["training_season_weight"] for row in train_rows], dtype=float)
        if config["history_scheme"] == "season_weighted"
        else np.ones(len(train_rows), dtype=float)
    )
    estimator = build_estimator(config["model_family"], config["parameters"])
    if config["model_family"] == "ridge":
        estimator.fit(matrix, target, model__sample_weight=weights)
    else:
        estimator.fit(matrix, target, sample_weight=weights)
    return estimator


def rank_race(
    estimator: Any,
    race_rows: list[dict[str, Any]],
    config: dict[str, Any],
    model_id: str,
) -> list[dict[str, Any]]:
    columns = config["feature_columns"]
    matrix = np.asarray([[row[column] for column in columns] for row in race_rows], dtype=float)
    scores = estimator.predict(matrix)
    score_by_driver = {
        row["driver_id"]: float(score) for row, score in zip(race_rows, scores)
    }
    ordered = sorted(
        race_rows,
        key=lambda row: (-score_by_driver[row["driver_id"]], row["driver_id"]),
    )
    predictions = []
    for position, row in enumerate(ordered, start=1):
        predictions.append(
            {
                "model_id": model_id,
                "model_family": config["model_family"],
                "feature_set": config["feature_set"],
                "history_scheme": config["history_scheme"],
                "race_id": row["race_id"],
                "season": row["season"],
                "round": row["round"],
                "race_date": row["race_date"],
                "driver_id": row["driver_id"],
                "driver_name": row["driver_name"],
                "constructor_id": row["constructor_id"],
                "constructor_name": row["constructor_name"],
                "model_score": score_by_driver[row["driver_id"]],
                "predicted_position": position,
                "actual_position": row["actual_position"],
                "actual_classification_status": row["actual_classification_status"],
                "absolute_position_error": abs(position - row["actual_position"]),
                "driver_prior_starts": row["driver_prior_starts"],
            }
        )
    return predictions


def spearman_correlation(predicted: list[int], actual: list[int]) -> float:
    count = len(predicted)
    squared_difference = sum((left - right) ** 2 for left, right in zip(predicted, actual))
    return 1.0 - (6.0 * squared_difference) / (count * (count**2 - 1))


def kendall_correlation(predicted: list[int], actual: list[int]) -> float:
    concordant = 0
    discordant = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            direction = (predicted[left] - predicted[right]) * (actual[left] - actual[right])
            if direction > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def score_race(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(predictions, key=lambda row: row["driver_id"])
    predicted = [row["predicted_position"] for row in ordered]
    actual = [row["actual_position"] for row in ordered]
    predicted_winner = min(predictions, key=lambda row: row["predicted_position"])["driver_id"]
    actual_winner = min(predictions, key=lambda row: row["actual_position"])["driver_id"]
    predicted_top_three = {
        row["driver_id"] for row in predictions if row["predicted_position"] <= 3
    }
    actual_top_three = {row["driver_id"] for row in predictions if row["actual_position"] <= 3}
    return {
        "model_id": predictions[0]["model_id"],
        "model_family": predictions[0]["model_family"],
        "feature_set": predictions[0]["feature_set"],
        "history_scheme": predictions[0]["history_scheme"],
        "race_id": predictions[0]["race_id"],
        "season": predictions[0]["season"],
        "round": predictions[0]["round"],
        "race_date": predictions[0]["race_date"],
        "field_size": len(predictions),
        "spearman_correlation": spearman_correlation(predicted, actual),
        "mean_absolute_position_error": mean_or(
            abs(left - right) for left, right in zip(predicted, actual)
        ),
        "kendall_correlation": kendall_correlation(predicted, actual),
        "winner_accuracy": float(predicted_winner == actual_winner),
        "top_three_overlap": len(predicted_top_three & actual_top_three) / 3.0,
    }


def summarize_race_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "average_spearman_correlation": mean_or(
            row["spearman_correlation"] for row in scores
        ),
        "average_mean_absolute_position_error": mean_or(
            row["mean_absolute_position_error"] for row in scores
        ),
        "average_kendall_correlation": mean_or(
            row["kendall_correlation"] for row in scores
        ),
        "winner_accuracy": mean_or(row["winner_accuracy"] for row in scores),
        "average_top_three_overlap": mean_or(row["top_three_overlap"] for row in scores),
    }


def selection_race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        group
        for group in race_groups(rows)
        if group[0]["season"] == SELECTION_SEASON
        and group[0]["round"] in SELECTION_ROUNDS
    ]


def test_race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        group
        for group in race_groups(rows)
        if group[0]["season"] == SELECTION_SEASON and group[0]["round"] in TEST_ROUNDS
    ]


def evaluate_selection_config(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_id = config["model_id"]
    race_scores = []
    for group in selection_race_groups(rows):
        train_rows = training_rows_before(rows, group[0]["race_date"], config["history_scheme"])
        estimator = fit_estimator(train_rows, config)
        predictions = rank_race(estimator, group, config, model_id)
        race_scores.append(score_race(predictions))
    summary = summarize_race_scores(race_scores)
    experiment = {
        "model_id": model_id,
        "model_family": config["model_family"],
        "feature_set": config["feature_set"],
        "history_scheme": config["history_scheme"],
        "parameters": json.dumps(config["parameters"], sort_keys=True),
        "selection_races": len(race_scores),
        **summary,
    }
    return experiment, race_scores


def ridge_configs() -> list[dict[str, Any]]:
    configs = []
    for feature_set in FEATURE_SETS:
        for history_scheme in HISTORY_SCHEMES:
            for alpha in RIDGE_ALPHAS:
                configs.append(
                    {
                        "model_id": f"ridge_{feature_set}_{history_scheme}_alpha_{alpha:g}",
                        "model_family": "ridge",
                        "feature_set": feature_set,
                        "history_scheme": history_scheme,
                        "feature_columns": feature_columns(feature_set, history_scheme),
                        "parameters": {"alpha": alpha},
                    }
                )
    return configs


def gradient_configs() -> list[dict[str, Any]]:
    parameters = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 10,
    }
    return [
        {
            "model_id": f"gradient_boosting_{feature_set}_{history_scheme}",
            "model_family": "gradient_boosting",
            "feature_set": feature_set,
            "history_scheme": history_scheme,
            "feature_columns": feature_columns(feature_set, history_scheme),
            "parameters": dict(parameters),
        }
        for feature_set in FEATURE_SETS
        for history_scheme in HISTORY_SCHEMES
    ]


def best_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        experiments,
        key=lambda row: (
            -row["average_spearman_correlation"],
            row["average_mean_absolute_position_error"],
            row["model_id"],
        ),
    )[0]


def config_by_id(configs: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    return next(config for config in configs if config["model_id"] == model_id)


def train_frozen_model(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selection_groups = selection_race_groups(rows)
    cutoff_date = selection_groups[-1][0]["race_date"]
    train_rows = final_training_rows(rows, cutoff_date, config["history_scheme"])
    estimator = fit_estimator(train_rows, config)
    predictions = []
    race_scores = []
    for group in test_race_groups(rows):
        group_predictions = rank_race(estimator, group, config, config["model_id"])
        predictions.extend(group_predictions)
        race_scores.append(score_race(group_predictions))
    return estimator, train_rows, predictions, race_scores


def baseline_benchmark(retrieval_id: str) -> dict[str, Any]:
    path = BASELINE_ROOT / retrieval_id / "baseline_race_scores.csv"
    rows = [
        row
        for row in load_csv(path)
        if int(row["season"]) == SELECTION_SEASON and int(row["round"]) in TEST_ROUNDS
    ]
    by_baseline: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_baseline[row["baseline"]].append(row)
    summaries = []
    for baseline, baseline_rows in by_baseline.items():
        summaries.append(
            {
                "baseline": baseline,
                "average_spearman_correlation": mean_or(
                    float(row["spearman_correlation"]) for row in baseline_rows
                ),
                "races": len(baseline_rows),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (-row["average_spearman_correlation"], row["baseline"]),
    )[0]


def model_summary(
    config: dict[str, Any],
    race_scores: list[dict[str, Any]],
    ridge_scores: list[dict[str, Any]],
    selected_model_id: str,
) -> dict[str, Any]:
    summary = summarize_race_scores(race_scores)
    ridge_by_race = {row["race_id"]: row for row in ridge_scores}
    races_beating_ridge = sum(
        row["spearman_correlation"] > ridge_by_race[row["race_id"]]["spearman_correlation"]
        for row in race_scores
    ) if config["model_family"] != "ridge" else 0
    return {
        "model_id": config["model_id"],
        "model_family": config["model_family"],
        "feature_set": config["feature_set"],
        "history_scheme": config["history_scheme"],
        "test_races": len(race_scores),
        **summary,
        "races_beating_ridge": races_beating_ridge,
        "selected_model": config["model_id"] == selected_model_id,
    }


def choose_final_model(
    ridge_config: dict[str, Any],
    ridge_scores: list[dict[str, Any]],
    gradient_config: dict[str, Any] | None,
    gradient_scores: list[dict[str, Any]] | None,
) -> str:
    if gradient_config is None or gradient_scores is None:
        return ridge_config["model_id"]
    ridge_summary = summarize_race_scores(ridge_scores)
    gradient_summary = summarize_race_scores(gradient_scores)
    ridge_by_race = {row["race_id"]: row for row in ridge_scores}
    gradient_wins = sum(
        row["spearman_correlation"] > ridge_by_race[row["race_id"]]["spearman_correlation"]
        for row in gradient_scores
    )
    if (
        gradient_summary["average_spearman_correlation"]
        > ridge_summary["average_spearman_correlation"]
        and gradient_wins >= GRADIENT_REQUIRED_TEST_WINS
    ):
        return gradient_config["model_id"]
    return ridge_config["model_id"]


def segment_summary(
    predictions: list[dict[str, Any]],
    model_id: str,
) -> list[dict[str, Any]]:
    segments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["model_id"] != model_id:
            continue
        position = row["actual_position"]
        field_segment = "front_runner" if position <= 7 else "midfield" if position <= 15 else "backmarker"
        experience_segment = "rookie" if row["driver_prior_starts"] < 10 else "experienced"
        if row["actual_classification_status"] == "DNF":
            classification_segment = "DNF"
        elif row["actual_classification_status"] == "CLASSIFIED":
            classification_segment = "classified"
        else:
            classification_segment = "other_exceptional"
        segments[("field", field_segment)].append(row)
        segments[("experience", experience_segment)].append(row)
        segments[("classification", classification_segment)].append(row)
    return [
        {
            "model_id": model_id,
            "segment_type": segment_type,
            "segment": segment,
            "driver_rows": len(rows),
            "mean_absolute_position_error": mean_or(
                row["absolute_position_error"] for row in rows
            ),
        }
        for (segment_type, segment), rows in sorted(segments.items())
    ]


def error_analysis(
    predictions: list[dict[str, Any]],
    model_id: str,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in predictions if row["model_id"] == model_id]
    largest = sorted(
        selected,
        key=lambda row: (-row["absolute_position_error"], row["race_id"], row["driver_id"]),
    )[:15]
    dnf_segment = next(
        (row for row in segments if row["segment_type"] == "classification" and row["segment"] == "DNF"),
        None,
    )
    classified_segment = next(
        (
            row
            for row in segments
            if row["segment_type"] == "classification" and row["segment"] == "classified"
        ),
        None,
    )
    repeat_counts: dict[str, int] = defaultdict(int)
    for row in largest:
        repeat_counts[row["driver_id"]] += 1
    recurring_drivers = {
        driver_id: count for driver_id, count in repeat_counts.items() if count > 1
    }
    return {
        "model_id": model_id,
        "largest_errors": [
            {
                key: row[key]
                for key in (
                    "race_id",
                    "driver_id",
                    "driver_name",
                    "constructor_id",
                    "predicted_position",
                    "actual_position",
                    "actual_classification_status",
                    "absolute_position_error",
                )
            }
            for row in largest
        ],
        "recurring_drivers_in_largest_errors": recurring_drivers,
        "dnf_mean_absolute_error": (
            dnf_segment["mean_absolute_position_error"] if dnf_segment else None
        ),
        "classified_mean_absolute_error": (
            classified_segment["mean_absolute_position_error"] if classified_segment else None
        ),
        "notes": [
            "Exceptional results are evaluated in the final order but were excluded from pace-model training.",
            "Circuit history is used as a prior summary and never as a guarantee of repeating a past result.",
        ],
    }


def model_inspection(estimator: Any, config: dict[str, Any]) -> dict[str, Any]:
    if config["model_family"] == "ridge":
        model = estimator.named_steps["model"]
        effects = [
            {"feature": feature, "standardized_coefficient": float(coefficient)}
            for feature, coefficient in zip(config["feature_columns"], model.coef_)
        ]
        return {
            "model_id": config["model_id"],
            "intercept": float(model.intercept_),
            "feature_effects": sorted(
                effects,
                key=lambda row: (-abs(row["standardized_coefficient"]), row["feature"]),
            ),
        }
    effects = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(
            config["feature_columns"],
            estimator.feature_importances_,
        )
    ]
    return {
        "model_id": config["model_id"],
        "feature_effects": sorted(
            effects,
            key=lambda row: (-row["importance"], row["feature"]),
        ),
    }


def prediction_signature(predictions: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["model_id"],
            row["race_id"],
            row["driver_id"],
            round(row["model_score"], 12),
            row["predicted_position"],
        )
        for row in predictions
    ]


def validate_outputs(
    rows: list[dict[str, Any]],
    selected_config: dict[str, Any],
    selected_estimator: Any,
    train_rows: list[dict[str, Any]],
    all_predictions: list[dict[str, Any]],
    selected_predictions: list[dict[str, Any]],
    selected_race_scores: list[dict[str, Any]],
    model_path: Path,
    performance_gate_passed: bool,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    test_groups = test_race_groups(rows)
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_predictions:
        prediction_groups[row["race_id"]].append(row)

    checks["test_rounds_are_exactly_seven_through_eleven"] = (
        sorted(group[0]["round"] for group in test_groups) == list(TEST_ROUNDS)
    )
    checks["training_contains_only_classified_results"] = all(
        row["actual_classification_status"] == "CLASSIFIED" for row in train_rows
    )
    checks["training_cutoff_is_round_six"] = max(
        (row["season"], row["round"]) for row in train_rows
    ) == (SELECTION_SEASON, SELECTION_ROUNDS[-1])
    checks["every_test_race_has_every_driver_once"] = all(
        len(prediction_groups[group[0]["race_id"]]) == len(group)
        and len({row["driver_id"] for row in prediction_groups[group[0]["race_id"]]}) == len(group)
        for group in test_groups
    )
    checks["predicted_positions_are_complete"] = all(
        sorted(row["predicted_position"] for row in group) == list(range(1, len(group) + 1))
        for group in prediction_groups.values()
    )
    checks["one_score_per_test_race"] = len(selected_race_scores) == len(TEST_ROUNDS)
    checks["saved_model_exists"] = model_path.exists()

    loaded_estimator = joblib.load(model_path)
    loaded_predictions = []
    for group in test_groups:
        loaded_predictions.extend(
            rank_race(
                loaded_estimator,
                group,
                selected_config,
                selected_config["model_id"],
            )
        )
    checks["saved_model_reproduces_predictions"] = (
        prediction_signature(selected_predictions) == prediction_signature(loaded_predictions)
    )

    repeated_estimator = fit_estimator(copy.deepcopy(train_rows), selected_config)
    repeated_predictions = []
    for group in test_groups:
        repeated_predictions.extend(
            rank_race(
                repeated_estimator,
                group,
                selected_config,
                selected_config["model_id"],
            )
        )
    checks["fixed_seed_reproduces_predictions"] = (
        prediction_signature(selected_predictions) == prediction_signature(repeated_predictions)
    )
    checks["performance_gate_passed"] = performance_gate_passed

    for name, passed in checks.items():
        if not passed:
            errors.append(f"Failed check: {name}")
    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_config["model_id"],
        "training_rows": len(train_rows),
        "test_races": len(TEST_ROUNDS),
        "prediction_rows": len(selected_predictions),
        "compared_model_prediction_rows": len(all_predictions),
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = choose_feature_dataset(args.retrieval)
    output_dir = args.output_dir or MODEL_EVALUATION_ROOT / dataset_dir.name
    model_dir = args.model_dir or MODEL_ROOT / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset_dir)

    ridge_candidates = ridge_configs()
    ridge_experiments = []
    for config in ridge_candidates:
        experiment, _ = evaluate_selection_config(rows, config)
        ridge_experiments.append(experiment)
    best_ridge_experiment = best_experiment(ridge_experiments)
    best_ridge_config = config_by_id(ridge_candidates, best_ridge_experiment["model_id"])

    gradient_candidates: list[dict[str, Any]] = []
    gradient_experiments: list[dict[str, Any]] = []
    best_gradient_config: dict[str, Any] | None = None
    if best_ridge_experiment["average_spearman_correlation"] < MVP_SPEARMAN_TARGET:
        gradient_candidates = gradient_configs()
        for config in gradient_candidates:
            experiment, _ = evaluate_selection_config(rows, config)
            gradient_experiments.append(experiment)
        best_gradient_experiment = best_experiment(gradient_experiments)
        best_gradient_config = config_by_id(
            gradient_candidates,
            best_gradient_experiment["model_id"],
        )

    ridge_estimator, ridge_train_rows, ridge_predictions, ridge_scores = train_frozen_model(
        rows,
        best_ridge_config,
    )
    compared_configs = [best_ridge_config]
    compared_estimators = {best_ridge_config["model_id"]: ridge_estimator}
    compared_train_rows = {best_ridge_config["model_id"]: ridge_train_rows}
    all_predictions = list(ridge_predictions)
    all_race_scores = list(ridge_scores)

    gradient_scores = None
    if best_gradient_config is not None:
        gradient_estimator, gradient_train_rows, gradient_predictions, gradient_scores = (
            train_frozen_model(rows, best_gradient_config)
        )
        compared_configs.append(best_gradient_config)
        compared_estimators[best_gradient_config["model_id"]] = gradient_estimator
        compared_train_rows[best_gradient_config["model_id"]] = gradient_train_rows
        all_predictions.extend(gradient_predictions)
        all_race_scores.extend(gradient_scores)

    selected_model_id = choose_final_model(
        best_ridge_config,
        ridge_scores,
        best_gradient_config,
        gradient_scores,
    )
    selected_config = next(config for config in compared_configs if config["model_id"] == selected_model_id)
    selected_estimator = compared_estimators[selected_model_id]
    selected_train_rows = compared_train_rows[selected_model_id]
    selected_predictions = [row for row in all_predictions if row["model_id"] == selected_model_id]
    selected_race_scores = [row for row in all_race_scores if row["model_id"] == selected_model_id]

    summaries = [
        model_summary(config, [row for row in all_race_scores if row["model_id"] == config["model_id"]], ridge_scores, selected_model_id)
        for config in compared_configs
    ]
    baseline = baseline_benchmark(dataset_dir.name)
    selected_summary = next(row for row in summaries if row["selected_model"])
    performance_gate_passed = (
        selected_summary["average_spearman_correlation"] >= MVP_SPEARMAN_TARGET
        and selected_summary["average_spearman_correlation"]
        > baseline["average_spearman_correlation"]
    )

    model_path = model_dir / "ranking_model.joblib"
    joblib.dump(selected_estimator, model_path)
    segments = segment_summary(all_predictions, selected_model_id)
    errors = error_analysis(all_predictions, selected_model_id, segments)
    inspection = model_inspection(selected_estimator, selected_config)
    report = validate_outputs(
        rows,
        selected_config,
        selected_estimator,
        selected_train_rows,
        all_predictions,
        selected_predictions,
        selected_race_scores,
        model_path,
        performance_gate_passed,
    )

    write_csv(
        output_dir / "selection_experiments.csv",
        ridge_experiments + gradient_experiments,
        EXPERIMENT_COLUMNS,
    )
    write_csv(output_dir / "model_predictions.csv", all_predictions, PREDICTION_COLUMNS)
    write_csv(output_dir / "model_race_scores.csv", all_race_scores, RACE_SCORE_COLUMNS)
    write_csv(output_dir / "model_summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(output_dir / "segment_summary.csv", segments, SEGMENT_COLUMNS)
    write_json(output_dir / "error_analysis.json", errors)
    write_json(model_dir / "model_inspection.json", inspection)
    write_json(output_dir / "validation_report.json", report)
    write_json(
        model_dir / "model_metadata.json",
        {
            "model_id": selected_model_id,
            "model_family": selected_config["model_family"],
            "feature_set": selected_config["feature_set"],
            "history_scheme": selected_config["history_scheme"],
            "feature_columns": selected_config["feature_columns"],
            "parameters": selected_config["parameters"],
            "random_seed": RANDOM_SEED,
            "training_result_filter": "CLASSIFIED",
            "training_cutoff_season": SELECTION_SEASON,
            "training_cutoff_round": SELECTION_ROUNDS[-1],
            "training_cutoff_date": selection_race_groups(rows)[-1][0]["race_date"],
            "training_rows": len(selected_train_rows),
            "test_rounds": list(TEST_ROUNDS),
            "feature_dataset": (
                f"data/features/fastf1/{dataset_dir.name}/pre_weekend_features.csv"
            ),
            "target_dataset": f"data/features/fastf1/{dataset_dir.name}/race_targets.csv",
            "baseline_benchmark": baseline,
            "evaluation": selected_summary,
            "performance_gate": {
                "required_spearman": MVP_SPEARMAN_TARGET,
                "must_beat_baseline": True,
                "passed": performance_gate_passed,
            },
            "scikit_learn_version": sklearn.__version__,
        },
    )
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "selection_rounds": list(SELECTION_ROUNDS),
            "test_rounds": list(TEST_ROUNDS),
            "model_frozen_after_round": SELECTION_ROUNDS[-1],
            "classified_results_only_for_training": True,
            "recent_history_races": RECENT_RACE_COUNT,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "gradient_boosting_evaluated": bool(gradient_experiments),
            "gradient_selection_rule": (
                "Higher average test Spearman than Ridge and better race-level Spearman in at least "
                f"{GRADIENT_REQUIRED_TEST_WINS} of {len(TEST_ROUNDS)} races."
            ),
            "baseline_benchmark": baseline,
            "selected_model": selected_model_id,
            "performance_gate_passed": performance_gate_passed,
        },
    )

    if report["status"] != "passed":
        raise ValueError("Model evaluation did not pass all required checks.")
    print(f"Selected model: {selected_model_id}")
    print(
        "Average Spearman correlation: "
        f"{selected_summary['average_spearman_correlation']:.3f}"
    )
    print(
        "Strongest matching baseline: "
        f"{baseline['baseline']} ({baseline['average_spearman_correlation']:.3f})"
    )
    print(f"Performance gate passed: {performance_gate_passed}")


if __name__ == "__main__":
    main()
