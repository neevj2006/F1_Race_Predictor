from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_race_predictor.artifacts import portable_path, sha256_file
from f1_race_predictor.post_qualifying_features import (
    PRACTICE_FEATURES,
    RELIABILITY_FEATURES,
)
from f1_race_predictor.post_qualifying_model import BlendedRankingModel
from f1_race_predictor.training import FULL_BASE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "post_qualifying"
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "data" / "evaluations" / "post_qualifying"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "post_qualifying"
DEFAULT_PRE_WEEKEND_PREDICTIONS = (
    PROJECT_ROOT / "data" / "evaluations" / "models" / "20260810T200317Z" / "model_predictions.csv"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT / "data" / "processed" / "fastf1" / "20260810T200317Z" / "race_results.csv"
)

SELECTION_SEASON = 2026
SELECTION_ROUNDS = tuple(range(1, 7))
TEST_ROUNDS = tuple(range(7, 12))
WEEKEND_WEIGHTS = (0.60, 0.70, 0.80)
RIDGE_ALPHA = 100.0
RANDOM_SEED = 42
REQUIRED_SELECTION_WINS = 3

CONTEXT_GROUPS = {
    "pre_weekend": [],
    "reliability": list(RELIABILITY_FEATURES),
    "practice": list(PRACTICE_FEATURES),
    "circuit_profile": [
        "profile_missing",
        "constructor_profile_prior_races",
        "constructor_profile_prior_score",
    ],
    "all_context": list(RELIABILITY_FEATURES)
    + list(PRACTICE_FEATURES)
    + [
        "profile_missing",
        "constructor_profile_prior_races",
        "constructor_profile_prior_score",
    ],
}

PREDICTION_COLUMNS = [
    "model_id",
    "context_group",
    "weekend_weight",
    "race_id",
    "season",
    "round",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "support_score",
    "qualifying_weekend_score",
    "model_score",
    "predicted_position",
    "actual_position",
    "actual_classification_status",
    "absolute_position_error",
]

RACE_SCORE_COLUMNS = [
    "model_id",
    "context_group",
    "weekend_weight",
    "race_id",
    "season",
    "round",
    "spearman_correlation",
    "mean_absolute_position_error",
    "kendall_correlation",
    "winner_accuracy",
    "top_three_overlap",
]

SELECTION_COLUMNS = [
    "model_id",
    "context_group",
    "weekend_weight",
    "support_feature_count",
    "selection_races",
    "average_spearman_correlation",
    "average_mean_absolute_position_error",
    "average_kendall_correlation",
    "winner_accuracy",
    "average_top_three_overlap",
    "spearman_change_from_matching_pre_weekend",
    "races_beating_matching_pre_weekend",
    "eligible_for_selection",
    "selected_model",
]

CONSTRUCTOR_PROFILE_COLUMNS = [
    "constructor_id",
    "constructor_name",
    "dominant_profile",
    "races",
    "classified_results",
    "average_normalized_finish_score",
    "average_normalized_position",
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


def mean(values: Iterable[float], fallback: float = 0.0) -> float:
    values = list(values)
    return sum(values) / len(values) if values else fallback


def finish_score(position: int, field_size: int) -> float:
    if field_size <= 1:
        return 0.5
    return 1.0 - ((position - 1) / (field_size - 1))


def load_rows(feature_dir: Path) -> list[dict[str, Any]]:
    with (feature_dir / "feature_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    numeric_columns = manifest["feature_columns"]
    features = read_csv(feature_dir / "post_qualifying_features.csv")
    targets = read_csv(feature_dir / "race_targets.csv")
    target_by_key = {(row["race_id"], row["driver_id"]): row for row in targets}
    rows = []
    for feature in features:
        key = (feature["race_id"], feature["driver_id"])
        target = target_by_key[key]
        row: dict[str, Any] = {
            **feature,
            "season": int(feature["season"]),
            "round": int(feature["round"]),
            "actual_position": int(target["target_normalized_position"]),
            "actual_classification_status": target["target_classification_status"],
            "training_season_weight": float(target["training_season_weight"]),
        }
        for column in numeric_columns:
            row[column] = float(feature[column])
        rows.append(row)
    rows.sort(key=lambda row: (row["race_date"], row["round"], row["driver_id"]))
    for group in race_groups(rows):
        for row in group:
            row["target_finish_score"] = finish_score(row["actual_position"], len(group))
    return rows


def race_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["race_id"]].append(row)
    return sorted(
        grouped.values(),
        key=lambda group: (group[0]["race_date"], group[0]["round"]),
    )


def support_columns(context_group: str) -> list[str]:
    return list(FULL_BASE_FEATURES) + CONTEXT_GROUPS[context_group]


def model_columns(context_group: str) -> list[str]:
    return support_columns(context_group) + ["qualifying_weekend_score"]


def classified_before(
    rows: list[dict[str, Any]],
    cutoff_date: str,
    include_cutoff: bool,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (row["race_date"] <= cutoff_date if include_cutoff else row["race_date"] < cutoff_date)
        and row["actual_classification_status"] == "CLASSIFIED"
    ]


def fit_model(
    train_rows: list[dict[str, Any]],
    context_group: str,
    weekend_weight: float,
) -> BlendedRankingModel:
    support = support_columns(context_group)
    columns = model_columns(context_group)
    support_matrix = np.asarray(
        [[row[column] for column in support] for row in train_rows],
        dtype=float,
    )
    target = np.asarray([row["target_finish_score"] for row in train_rows], dtype=float)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )
    estimator.fit(support_matrix, target)
    return BlendedRankingModel(
        estimator,
        list(range(len(support))),
        columns.index("qualifying_weekend_score"),
        weekend_weight,
    )


def model_id(context_group: str, weekend_weight: float) -> str:
    return f"post_qualifying_{context_group}_weekend_{int(weekend_weight * 100)}"


def rank_race(
    model: BlendedRankingModel,
    race_rows: list[dict[str, Any]],
    context_group: str,
    weekend_weight: float,
) -> list[dict[str, Any]]:
    columns = model_columns(context_group)
    matrix = np.asarray(
        [[row[column] for column in columns] for row in race_rows],
        dtype=float,
    )
    scores = model.predict(matrix)
    support_scores = np.clip(
        model.support_estimator.predict(matrix[:, model.support_indices]),
        0.0,
        1.0,
    )
    values = {
        row["driver_id"]: (float(score), float(support_score))
        for row, score, support_score in zip(race_rows, scores, support_scores)
    }
    ordered = sorted(
        race_rows,
        key=lambda row: (-values[row["driver_id"]][0], row["driver_id"]),
    )
    output = []
    for position, row in enumerate(ordered, start=1):
        score, support_score = values[row["driver_id"]]
        output.append(
            {
                "model_id": model_id(context_group, weekend_weight),
                "context_group": context_group,
                "weekend_weight": weekend_weight,
                "race_id": row["race_id"],
                "season": row["season"],
                "round": row["round"],
                "race_date": row["race_date"],
                "driver_id": row["driver_id"],
                "driver_name": row["driver_name"],
                "constructor_id": row["constructor_id"],
                "constructor_name": row["constructor_name"],
                "support_score": support_score,
                "qualifying_weekend_score": row["qualifying_weekend_score"],
                "model_score": score,
                "predicted_position": position,
                "actual_position": row["actual_position"],
                "actual_classification_status": row["actual_classification_status"],
                "absolute_position_error": abs(position - row["actual_position"]),
            }
        )
    return output


def spearman(predicted: list[int], actual: list[int]) -> float:
    count = len(predicted)
    difference = sum((left - right) ** 2 for left, right in zip(predicted, actual))
    return 1.0 - (6.0 * difference) / (count * (count**2 - 1))


def kendall(predicted: list[int], actual: list[int]) -> float:
    concordant = 0
    discordant = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_order = predicted[left] - predicted[right]
            actual_order = actual[left] - actual[right]
            if predicted_order * actual_order > 0:
                concordant += 1
            elif predicted_order * actual_order < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def score_race(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [row["predicted_position"] for row in predictions]
    actual = [row["actual_position"] for row in predictions]
    predicted_top_three = {
        row["driver_id"] for row in predictions if row["predicted_position"] <= 3
    }
    actual_top_three = {row["driver_id"] for row in predictions if row["actual_position"] <= 3}
    return {
        "model_id": predictions[0]["model_id"],
        "context_group": predictions[0]["context_group"],
        "weekend_weight": predictions[0]["weekend_weight"],
        "race_id": predictions[0]["race_id"],
        "season": predictions[0]["season"],
        "round": predictions[0]["round"],
        "spearman_correlation": spearman(predicted, actual),
        "mean_absolute_position_error": mean(row["absolute_position_error"] for row in predictions),
        "kendall_correlation": kendall(predicted, actual),
        "winner_accuracy": float(
            next(row["driver_id"] for row in predictions if row["predicted_position"] == 1)
            == next(row["driver_id"] for row in predictions if row["actual_position"] == 1)
        ),
        "top_three_overlap": len(predicted_top_three & actual_top_three) / 3.0,
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "average_spearman_correlation": mean(row["spearman_correlation"] for row in scores),
        "average_mean_absolute_position_error": mean(
            row["mean_absolute_position_error"] for row in scores
        ),
        "average_kendall_correlation": mean(row["kendall_correlation"] for row in scores),
        "winner_accuracy": mean(row["winner_accuracy"] for row in scores),
        "average_top_three_overlap": mean(row["top_three_overlap"] for row in scores),
    }


def selection_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        group
        for group in race_groups(rows)
        if group[0]["season"] == SELECTION_SEASON and group[0]["round"] in SELECTION_ROUNDS
    ]


def test_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        group
        for group in race_groups(rows)
        if group[0]["season"] == SELECTION_SEASON and group[0]["round"] in TEST_ROUNDS
    ]


def evaluate_selection(
    rows: list[dict[str, Any]],
    context_group: str,
    weekend_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    scores = []
    for group in selection_groups(rows):
        train_rows = classified_before(rows, group[0]["race_date"], include_cutoff=False)
        model = fit_model(train_rows, context_group, weekend_weight)
        race_predictions = rank_race(model, group, context_group, weekend_weight)
        predictions.extend(race_predictions)
        scores.append(score_race(race_predictions))
    return predictions, scores


def selection_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    scores_by_id: dict[str, list[dict[str, Any]]] = {}
    for context_group in CONTEXT_GROUPS:
        for weekend_weight in WEEKEND_WEIGHTS:
            _, scores = evaluate_selection(rows, context_group, weekend_weight)
            experiment_id = model_id(context_group, weekend_weight)
            scores_by_id[experiment_id] = scores
            experiments.append(
                {
                    "model_id": experiment_id,
                    "context_group": context_group,
                    "weekend_weight": weekend_weight,
                    "support_feature_count": len(support_columns(context_group)),
                    "selection_races": len(scores),
                    **summarize(scores),
                }
            )

    base_by_weight: dict[float, dict[str, Any]] = {
        row["weekend_weight"]: row for row in experiments if row["context_group"] == "pre_weekend"
    }
    for row in experiments:
        base = base_by_weight[row["weekend_weight"]]
        context_scores = scores_by_id[row["model_id"]]
        base_scores = scores_by_id[base["model_id"]]
        wins = sum(
            context["spearman_correlation"] > baseline["spearman_correlation"]
            for context, baseline in zip(context_scores, base_scores)
        )
        change = row["average_spearman_correlation"] - base["average_spearman_correlation"]
        row["spearman_change_from_matching_pre_weekend"] = change
        row["races_beating_matching_pre_weekend"] = wins
        row["eligible_for_selection"] = row["context_group"] == "pre_weekend" or (
            change > 0 and wins >= REQUIRED_SELECTION_WINS
        )
        row["selected_model"] = False

    eligible = [row for row in experiments if row["eligible_for_selection"]]
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["average_spearman_correlation"],
            row["average_mean_absolute_position_error"],
            row["support_feature_count"],
            row["model_id"],
        ),
    )[0]
    selected["selected_model"] = True
    return experiments, selected


def frozen_test(
    rows: list[dict[str, Any]],
    context_group: str,
    weekend_weight: float,
) -> tuple[BlendedRankingModel, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff_group = next(
        group for group in selection_groups(rows) if group[0]["round"] == SELECTION_ROUNDS[-1]
    )
    train_rows = classified_before(rows, cutoff_group[0]["race_date"], include_cutoff=True)
    model = fit_model(train_rows, context_group, weekend_weight)
    predictions = []
    scores = []
    for group in test_groups(rows):
        race_predictions = rank_race(model, group, context_group, weekend_weight)
        predictions.extend(race_predictions)
        scores.append(score_race(race_predictions))
    return model, train_rows, predictions, scores


def pre_weekend_benchmark(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    rows = [
        row
        for row in read_csv(path)
        if row["model_id"] == "ridge_full_unweighted_alpha_100"
        and int(row["season"]) == SELECTION_SEASON
        and int(row["round"]) in TEST_ROUNDS
    ]
    by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_race[row["race_id"]].append(row)
    scores = []
    spearman_by_race = {}
    for race_id, race_rows in sorted(by_race.items()):
        predicted = [int(row["predicted_position"]) for row in race_rows]
        actual = [int(row["actual_position"]) for row in race_rows]
        value = spearman(predicted, actual)
        spearman_by_race[race_id] = value
        scores.append(value)
    return (
        {
            "model_id": "ridge_full_unweighted_alpha_100",
            "average_spearman_correlation": mean(scores),
            "races": len(scores),
        },
        spearman_by_race,
    )


def constructor_profile_comparison(
    results_file: Path,
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_by_key = {(row["race_id"], row["driver_id"]): row for row in read_csv(results_file)}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    profile_names = {
        "straight_demand": "straight",
        "cornering_demand": "cornering",
        "braking_demand": "braking",
        "tyre_stress": "tyre_stress",
    }
    for row in feature_rows:
        profile_column = max(profile_names, key=lambda column: row[column])
        result = result_by_key[(row["race_id"], row["driver_id"])]
        grouped[
            (row["constructor_id"], row["constructor_name"], profile_names[profile_column])
        ].append(
            {
                "race_id": row["race_id"],
                "position": int(result["normalized_position"]),
                "field_size": len(
                    [value for value in feature_rows if value["race_id"] == row["race_id"]]
                ),
                "classified": result["classification_status"] == "CLASSIFIED",
            }
        )
    output = []
    for (constructor_id, constructor_name, profile), values in grouped.items():
        classified = [row for row in values if row["classified"]]
        output.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": constructor_name,
                "dominant_profile": profile,
                "races": len({row["race_id"] for row in values}),
                "classified_results": len(classified),
                "average_normalized_finish_score": mean(
                    finish_score(row["position"], row["field_size"]) for row in classified
                ),
                "average_normalized_position": mean(row["position"] for row in classified),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["dominant_profile"],
            -row["average_normalized_finish_score"],
            row["constructor_id"],
        ),
    )


def model_inspection(model: BlendedRankingModel, context_group: str) -> dict[str, Any]:
    ridge = model.support_estimator.named_steps["model"]
    effects: list[dict[str, Any]] = sorted(
        (
            {
                "feature": feature,
                "standardized_coefficient": float(coefficient),
            }
            for feature, coefficient in zip(support_columns(context_group), ridge.coef_)
        ),
        key=lambda row: -abs(float(row["standardized_coefficient"])),
    )
    return {
        "weekend_weight": model.weekend_weight,
        "support_weight": 1.0 - model.weekend_weight,
        "weekend_score_components": {
            "official_grid_score": 0.65,
            "qualifying_position_score": 0.25,
            "qualifying_pace_score": 0.10,
        },
        "support_feature_effects": effects,
    }


def validate_outputs(
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    train_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    pre_weekend: dict[str, Any],
    pre_weekend_by_race: dict[str, float],
    model_path: Path,
) -> dict[str, Any]:
    post_summary = summarize(scores)
    post_by_race = {row["race_id"]: row["spearman_correlation"] for row in scores}
    test_wins = sum(
        post_by_race[race_id] > pre_weekend_by_race[race_id] for race_id in post_by_race
    )
    checks = {
        "test_rounds_are_exact": sorted(row["round"] for row in scores) == list(TEST_ROUNDS),
        "training_contains_only_classified_results": all(
            row["actual_classification_status"] == "CLASSIFIED" for row in train_rows
        ),
        "training_cutoff_is_round_six": max((row["season"], row["round"]) for row in train_rows)
        == (SELECTION_SEASON, SELECTION_ROUNDS[-1]),
        "every_test_race_has_every_driver_once": all(
            len(group) == len({row["driver_id"] for row in group})
            for group in race_groups(predictions)
        ),
        "valid_finishing_orders": all(
            sorted(row["predicted_position"] for row in group) == list(range(1, len(group) + 1))
            for group in race_groups(predictions)
        ),
        "strong_weekend_weight": selected["weekend_weight"] in WEEKEND_WEIGHTS,
        "post_qualifying_beats_pre_weekend_average": post_summary["average_spearman_correlation"]
        > pre_weekend["average_spearman_correlation"],
        "post_qualifying_beats_pre_weekend_in_three_races": test_wins >= 3,
        "saved_model_exists": model_path.is_file(),
        "no_weather_features": all("weather" not in key.lower() for key in rows[0]),
    }
    errors = [f"Failed check: {name}" for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected["model_id"],
        "training_rows": len(train_rows),
        "prediction_rows": len(predictions),
        "test_races": len(scores),
        "test_races_beating_pre_weekend": test_wins,
        "checks": checks,
        "errors": errors,
    }


def train(
    feature_dir: Path,
    evaluation_dir: Path,
    model_dir: Path,
    pre_weekend_predictions: Path,
    results_file: Path,
) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    rows = load_rows(feature_dir)
    experiments, selected = selection_table(rows)
    context_group = selected["context_group"]
    weekend_weight = float(selected["weekend_weight"])
    model, train_rows, predictions, scores = frozen_test(rows, context_group, weekend_weight)
    post_summary = summarize(scores)
    pre_weekend, pre_weekend_by_race = pre_weekend_benchmark(pre_weekend_predictions)
    selected_summary = {
        "model_id": selected["model_id"],
        "context_group": context_group,
        "weekend_weight": weekend_weight,
        "test_races": len(scores),
        **post_summary,
        "pre_weekend_average_spearman": pre_weekend["average_spearman_correlation"],
        "spearman_improvement": post_summary["average_spearman_correlation"]
        - pre_weekend["average_spearman_correlation"],
    }

    model_path = model_dir / "ranking_model.joblib"
    joblib.dump(model, model_path)
    report = validate_outputs(
        rows,
        selected,
        train_rows,
        predictions,
        scores,
        pre_weekend,
        pre_weekend_by_race,
        model_path,
    )
    write_csv(evaluation_dir / "feature_group_selection.csv", experiments, SELECTION_COLUMNS)
    write_csv(evaluation_dir / "model_predictions.csv", predictions, PREDICTION_COLUMNS)
    write_csv(evaluation_dir / "model_race_scores.csv", scores, RACE_SCORE_COLUMNS)
    write_csv(
        evaluation_dir / "constructor_profile_comparison.csv",
        constructor_profile_comparison(results_file, rows),
        CONSTRUCTOR_PROFILE_COLUMNS,
    )
    write_json(evaluation_dir / "model_summary.json", selected_summary)
    write_json(evaluation_dir / "validation_report.json", report)
    write_json(model_dir / "model_inspection.json", model_inspection(model, context_group))
    write_json(
        model_dir / "model_metadata.json",
        {
            "model_id": selected["model_id"],
            "model_family": "ridge_support_with_qualifying_blend",
            "feature_set": "post_qualifying",
            "context_group": context_group,
            "feature_columns": model_columns(context_group),
            "support_feature_columns": support_columns(context_group),
            "parameters": {
                "ridge_alpha": RIDGE_ALPHA,
                "weekend_weight": weekend_weight,
                "support_weight": 1.0 - weekend_weight,
                "weekend_score_components": {
                    "official_grid_score": 0.65,
                    "qualifying_position_score": 0.25,
                    "qualifying_pace_score": 0.10,
                },
            },
            "random_seed": RANDOM_SEED,
            "training_result_filter": "CLASSIFIED",
            "training_cutoff_season": SELECTION_SEASON,
            "training_cutoff_round": SELECTION_ROUNDS[-1],
            "training_cutoff_date": next(
                group[0]["race_date"]
                for group in selection_groups(rows)
                if group[0]["round"] == SELECTION_ROUNDS[-1]
            ),
            "training_rows": len(train_rows),
            "test_rounds": list(TEST_ROUNDS),
            "feature_dataset": portable_path(
                feature_dir / "post_qualifying_features.csv", PROJECT_ROOT
            ),
            "target_dataset": portable_path(feature_dir / "race_targets.csv", PROJECT_ROOT),
            "pre_weekend_benchmark": pre_weekend,
            "evaluation": selected_summary,
            "weather_included": False,
            "scikit_learn_version": sklearn.__version__,
        },
    )
    write_json(
        evaluation_dir / "evaluation_manifest.json",
        {
            "selection_rounds": list(SELECTION_ROUNDS),
            "test_rounds": list(TEST_ROUNDS),
            "model_frozen_after_round": SELECTION_ROUNDS[-1],
            "weekend_weights_tested": list(WEEKEND_WEIGHTS),
            "context_groups_tested": list(CONTEXT_GROUPS),
            "context_retention_rule": (
                "A context group must improve average selection Spearman and beat the matching "
                f"pre-weekend support model in at least {REQUIRED_SELECTION_WINS} selection races."
            ),
            "selected_model": selected["model_id"],
            "pre_weekend_benchmark": pre_weekend,
            "feature_dataset_sha256": sha256_file(feature_dir / "post_qualifying_features.csv"),
            "weather_included": False,
            "circuit_comparison_is_descriptive": True,
            "circuit_results_are_not_treated_as_guarantees": True,
        },
    )
    if report["status"] != "passed":
        raise ValueError("Post-qualifying model validation failed.")
    print(f"Selected model: {selected['model_id']}")
    print(f"Post-qualifying average Spearman: {post_summary['average_spearman_correlation']:.3f}")
    print(f"Pre-weekend average Spearman: {pre_weekend['average_spearman_correlation']:.3f}")
    print(f"Validation status: {report['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the separate post-qualifying model.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--pre-weekend-predictions",
        type=Path,
        default=DEFAULT_PRE_WEEKEND_PREDICTIONS,
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        args.feature_dir.resolve(),
        args.evaluation_dir.resolve(),
        args.model_dir.resolve(),
        args.pre_weekend_predictions.resolve(),
        args.results_file.resolve(),
    )


if __name__ == "__main__":
    main()
