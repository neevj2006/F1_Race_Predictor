from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_race_predictor.artifacts import portable_path, sha256_file
from f1_race_predictor.post_qualifying_features import PRACTICE_FEATURES
from f1_race_predictor.post_qualifying_training import (
    TEST_ROUNDS,
    classified_before,
    evaluate_selection,
    frozen_test,
    load_rows,
    mean,
    model_columns,
    race_groups,
    score_race,
    selection_groups,
    summarize,
    test_groups,
)
from f1_race_predictor.ranking_model import PairwiseRankingModel
from f1_race_predictor.training import FULL_BASE_FEATURES, FULL_WEIGHTED_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "data" / "features" / "post_qualifying" / "20260823T130000Z"
DEFAULT_EVALUATION_DIR = (
    PROJECT_ROOT / "data" / "evaluations" / "probabilities" / "20260823T130000Z"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "probabilities" / "20260823T130000Z"

RANDOM_SEED = 42
PAIRWISE_C_VALUES = (0.01, 0.1, 1.0)
HISTORY_SCHEMES = ("unweighted", "season_weighted", "recent_20")
CALIBRATION_C_VALUES = (0.1, 1.0, 10.0)
SIMULATION_CONFIGS = (
    (0.35, 0.15),
    (0.50, 0.15),
    (0.50, 0.25),
    (0.70, 0.25),
)
SELECTION_SIMULATIONS = 3000
TEST_SIMULATIONS = 10000
OUTCOMES = ("win", "podium", "top_10", "dnf")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)


def ranking_columns(history_scheme: str) -> list[str]:
    columns = list(FULL_BASE_FEATURES) + list(PRACTICE_FEATURES)
    if history_scheme == "season_weighted":
        columns.extend(FULL_WEIGHTED_FEATURES)
    columns.append("qualifying_weekend_score")
    return columns


def history_rows(
    rows: list[dict[str, Any]], cutoff_date: str, include_cutoff: bool, history_scheme: str
) -> list[dict[str, Any]]:
    eligible = classified_before(rows, cutoff_date, include_cutoff)
    if history_scheme != "recent_20":
        return eligible
    race_ids: list[str] = []
    for row in eligible:
        if row["race_id"] not in race_ids:
            race_ids.append(row["race_id"])
    keep = set(race_ids[-20:])
    return [row for row in eligible if row["race_id"] in keep]


def pairwise_matrix(
    rows: list[dict[str, Any]], columns: list[str], weighted: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for group in race_groups(rows):
        matrix = np.asarray([[row[column] for column in columns] for row in group], dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                difference = matrix[left] - matrix[right]
                left_better = group[left]["actual_position"] < group[right]["actual_position"]
                pair_weight = (
                    (group[left]["training_season_weight"] + group[right]["training_season_weight"])
                    / 2.0
                    if weighted
                    else 1.0
                )
                differences.extend((difference, -difference))
                labels.extend((int(left_better), int(not left_better)))
                weights.extend((pair_weight, pair_weight))
    return (
        np.asarray(differences, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=float),
    )


def fit_pairwise(
    train_rows: list[dict[str, Any]], history_scheme: str, c_value: float
) -> PairwiseRankingModel:
    columns = ranking_columns(history_scheme)
    matrix, labels, weights = pairwise_matrix(
        train_rows, columns, weighted=history_scheme == "season_weighted"
    )
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    fit_intercept=False,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    estimator.fit(matrix, labels, model__sample_weight=weights)
    return PairwiseRankingModel(estimator)


def rank_pairwise(
    model: PairwiseRankingModel,
    rows: list[dict[str, Any]],
    history_scheme: str,
    model_id: str,
) -> list[dict[str, Any]]:
    columns = ranking_columns(history_scheme)
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=float)
    scores = model.predict(matrix)
    score_by_driver = {row["driver_id"]: float(score) for row, score in zip(rows, scores)}
    ordered = sorted(rows, key=lambda row: (-score_by_driver[row["driver_id"]], row["driver_id"]))
    return [
        {
            "model_id": model_id,
            "context_group": "pairwise",
            "weekend_weight": "",
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
        }
        for position, row in enumerate(ordered, start=1)
    ]


def pairwise_id(history_scheme: str, c_value: float) -> str:
    return f"pairwise_logistic_{history_scheme}_c_{c_value:g}"


def evaluate_pairwise_selection(
    rows: list[dict[str, Any]], history_scheme: str, c_value: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    experiment_id = pairwise_id(history_scheme, c_value)
    for group in selection_groups(rows):
        training = history_rows(rows, group[0]["race_date"], False, history_scheme)
        model = fit_pairwise(training, history_scheme, c_value)
        race_predictions = rank_pairwise(model, group, history_scheme, experiment_id)
        predictions.extend(race_predictions)
        scores.append(score_race(race_predictions))
    return predictions, scores


def select_ranking_model(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    experiments: list[dict[str, Any]] = []
    predictions_by_id: dict[str, list[dict[str, Any]]] = {}
    current_predictions, current_scores = evaluate_selection(rows, "practice", 0.70)
    current_id = "post_qualifying_practice_weekend_70"
    predictions_by_id[current_id] = current_predictions
    experiments.append(
        {
            "model_id": current_id,
            "model_family": "ridge_support_with_qualifying_blend",
            "history_scheme": "unweighted",
            "parameters": "weekend_weight=0.7",
            "feature_count": len(model_columns("practice")),
            "selection_races": len(current_scores),
            **summarize(current_scores),
        }
    )
    for history_scheme in HISTORY_SCHEMES:
        for c_value in PAIRWISE_C_VALUES:
            predictions, scores = evaluate_pairwise_selection(rows, history_scheme, c_value)
            experiment_id = pairwise_id(history_scheme, c_value)
            predictions_by_id[experiment_id] = predictions
            experiments.append(
                {
                    "model_id": experiment_id,
                    "model_family": "pairwise_logistic",
                    "history_scheme": history_scheme,
                    "parameters": f"C={c_value:g}",
                    "feature_count": len(ranking_columns(history_scheme)),
                    "selection_races": len(scores),
                    **summarize(scores),
                }
            )
    current = experiments[0]
    current_by_race = {row["race_id"]: row for row in current_scores}
    for experiment in experiments:
        experiment_predictions = predictions_by_id[experiment["model_id"]]
        experiment_scores = [score_race(group) for group in race_groups(experiment_predictions)]
        wins = sum(
            row["spearman_correlation"] > current_by_race[row["race_id"]]["spearman_correlation"]
            for row in experiment_scores
        )
        experiment["spearman_change_from_current"] = (
            experiment["average_spearman_correlation"] - current["average_spearman_correlation"]
        )
        experiment["races_beating_current"] = wins
        experiment["eligible_for_use"] = experiment["model_id"] == current_id or (
            experiment["spearman_change_from_current"] > 0 and wins >= 3
        )
        experiment["selected_model"] = False
    eligible = [row for row in experiments if row["eligible_for_use"]]
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["average_spearman_correlation"],
            row["average_mean_absolute_position_error"],
            row["feature_count"],
            row["model_id"],
        ),
    )[0]
    selected["selected_model"] = True
    return experiments, selected, predictions_by_id


def stable_seed(race_id: str, extra: int = 0) -> int:
    digest = hashlib.sha256(f"{RANDOM_SEED}:{race_id}:{extra}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def global_dnf_rate(rows: list[dict[str, Any]], cutoff_date: str, include_cutoff: bool) -> float:
    earlier = [
        row
        for row in rows
        if (row["race_date"] <= cutoff_date if include_cutoff else row["race_date"] < cutoff_date)
    ]
    return mean(
        (float(row["actual_classification_status"] == "DNF") for row in earlier),
        fallback=0.10,
    )


def simulate_race(
    race_rows: list[dict[str, Any]],
    base_scores: dict[str, float],
    driver_sigma: float,
    constructor_sigma: float,
    field_dnf_rate: float,
    simulations: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered_rows = sorted(race_rows, key=lambda row: row["driver_id"])
    count = len(ordered_rows)
    raw_scores = np.asarray([base_scores[row["driver_id"]] for row in ordered_rows], dtype=float)
    score_std = float(np.std(raw_scores))
    centered_scores = (raw_scores - float(np.mean(raw_scores))) / (score_std or 1.0)
    constructors = sorted({row["constructor_id"] for row in ordered_rows})
    constructor_index = {constructor: index for index, constructor in enumerate(constructors)}
    driver_constructor = np.asarray(
        [constructor_index[row["constructor_id"]] for row in ordered_rows], dtype=int
    )
    constructor_rates = np.asarray(
        [row["constructor_recent_dnf_rate_10_weekend"] for row in ordered_rows], dtype=float
    )
    dnf_probabilities = np.clip(0.70 * constructor_rates + 0.30 * field_dnf_rate, 0.01, 0.60)
    rng = np.random.default_rng(seed)
    driver_noise = rng.normal(0.0, driver_sigma, size=(simulations, count))
    constructor_noise = rng.normal(0.0, constructor_sigma, size=(simulations, len(constructors)))[
        :, driver_constructor
    ]
    utility = centered_scores + driver_noise + constructor_noise
    dnf = rng.random((simulations, count)) < dnf_probabilities
    failure_time = rng.random((simulations, count))
    ordering_score = np.where(dnf, -100.0 + failure_time, utility)
    order = np.argsort(-ordering_score, axis=1, kind="stable")
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:, None], order] = np.arange(1, count + 1)
    return [
        {
            "race_id": row["race_id"],
            "season": row["season"],
            "round": row["round"],
            "race_date": row["race_date"],
            "driver_id": row["driver_id"],
            "driver_name": row["driver_name"],
            "constructor_id": row["constructor_id"],
            "constructor_name": row["constructor_name"],
            "raw_win_probability": float(np.mean(positions[:, index] == 1)),
            "raw_podium_probability": float(np.mean(positions[:, index] <= 3)),
            "raw_top_10_probability": float(np.mean(positions[:, index] <= min(10, count))),
            "raw_dnf_probability": float(np.mean(dnf[:, index])),
            "expected_position": float(np.mean(positions[:, index])),
            "actual_position": row["actual_position"],
            "actual_classification_status": row["actual_classification_status"],
        }
        for index, row in enumerate(ordered_rows)
    ]


def scores_by_driver(predictions: list[dict[str, Any]]) -> dict[str, float]:
    return {row["driver_id"]: float(row["model_score"]) for row in predictions}


def simulation_order_score(probabilities: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    ordered = sorted(probabilities, key=lambda row: (row["expected_position"], row["driver_id"]))
    predictions = [
        {
            **row,
            "model_id": model_id,
            "context_group": "simulation",
            "weekend_weight": "",
            "predicted_position": position,
            "absolute_position_error": abs(position - row["actual_position"]),
        }
        for position, row in enumerate(ordered, start=1)
    ]
    return score_race(predictions)


def choose_simulation(
    rows: list[dict[str, Any]],
    selected_predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_by_race = {
        group[0]["race_id"]: scores_by_driver(group) for group in race_groups(selected_predictions)
    }
    experiments: list[dict[str, Any]] = []
    for driver_sigma, constructor_sigma in SIMULATION_CONFIGS:
        race_scores = []
        for group in selection_groups(rows):
            rate = global_dnf_rate(rows, group[0]["race_date"], False)
            probabilities = simulate_race(
                group,
                predictions_by_race[group[0]["race_id"]],
                driver_sigma,
                constructor_sigma,
                rate,
                SELECTION_SIMULATIONS,
                stable_seed(
                    group[0]["race_id"], int(driver_sigma * 1000 + constructor_sigma * 100)
                ),
            )
            race_scores.append(
                simulation_order_score(
                    probabilities,
                    f"simulation_driver_{driver_sigma:g}_constructor_{constructor_sigma:g}",
                )
            )
        experiments.append(
            {
                "driver_sigma": driver_sigma,
                "constructor_sigma": constructor_sigma,
                "selection_races": len(race_scores),
                **summarize(race_scores),
                "selected": False,
            }
        )
    selected = sorted(
        experiments,
        key=lambda row: (
            -row["average_spearman_correlation"],
            row["average_mean_absolute_position_error"],
            row["driver_sigma"],
            row["constructor_sigma"],
        ),
    )[0]
    selected["selected"] = True
    return experiments, selected


def actual_outcome(row: dict[str, Any], outcome: str) -> int:
    if outcome == "win":
        return int(row["actual_position"] == 1)
    if outcome == "podium":
        return int(row["actual_position"] <= 3)
    if outcome == "top_10":
        return int(row["actual_position"] <= 10)
    return int(row["actual_classification_status"] == "DNF")


def raw_probability(row: dict[str, Any], outcome: str) -> float:
    return float(row[f"raw_{outcome}_probability"])


def logit(value: float) -> float:
    clipped = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fit_sigmoid(rows: list[dict[str, Any]], outcome: str, c_value: float) -> dict[str, float]:
    matrix = np.asarray([[logit(raw_probability(row, outcome))] for row in rows], dtype=float)
    target = np.asarray([actual_outcome(row, outcome) for row in rows], dtype=int)
    model = LogisticRegression(C=c_value, solver="lbfgs", random_state=RANDOM_SEED)
    model.fit(matrix, target)
    return {
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def calibrated_probability(raw: float, parameters: dict[str, float]) -> float:
    return sigmoid(parameters["coefficient"] * logit(raw) + parameters["intercept"])


def brier(rows: Iterable[tuple[float, int]]) -> float:
    values = list(rows)
    return mean((probability - actual) ** 2 for probability, actual in values)


def select_calibration(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    races = sorted({row["race_id"] for row in rows})
    comparison: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for outcome in OUTCOMES:
        raw_score = brier(
            (raw_probability(row, outcome), actual_outcome(row, outcome)) for row in rows
        )
        candidates = []
        for c_value in CALIBRATION_C_VALUES:
            held_out_values: list[tuple[float, int]] = []
            for race_id in races:
                training = [row for row in rows if row["race_id"] != race_id]
                held_out = [row for row in rows if row["race_id"] == race_id]
                parameters = fit_sigmoid(training, outcome, c_value)
                held_out_values.extend(
                    (
                        calibrated_probability(raw_probability(row, outcome), parameters),
                        actual_outcome(row, outcome),
                    )
                    for row in held_out
                )
            candidates.append((c_value, brier(held_out_values)))
        chosen_c, calibrated_score = min(candidates, key=lambda value: (value[1], value[0]))
        use_calibration = calibrated_score < raw_score
        parameters = fit_sigmoid(rows, outcome, chosen_c)
        selected[outcome] = {
            "c": chosen_c,
            **parameters,
            "selection_raw_brier": raw_score,
            "selection_grouped_cv_brier": calibrated_score,
            "use_calibration": use_calibration,
        }
        comparison.append(
            {
                "outcome": outcome,
                "raw_brier": raw_score,
                "sigmoid_grouped_cv_brier": calibrated_score,
                "selected_c": chosen_c,
                "use_calibration": use_calibration,
            }
        )
    return selected, comparison


def add_calibrated_probabilities(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> None:
    for row in rows:
        for outcome in OUTCOMES:
            raw = raw_probability(row, outcome)
            calibrated = calibrated_probability(raw, calibration[outcome])
            row[f"calibrated_{outcome}_probability"] = calibrated
            row[f"preferred_{outcome}_probability"] = (
                calibrated if calibration[outcome]["use_calibration"] else raw
            )


def probability_metrics(
    rows: list[dict[str, Any]], outcome: str, probability_column: str
) -> dict[str, float]:
    values = [(float(row[probability_column]), actual_outcome(row, outcome)) for row in rows]
    brier_score = brier(values)
    log_loss = -mean(
        actual * math.log(min(max(probability, 1e-12), 1.0 - 1e-12))
        + (1 - actual) * math.log(min(max(1.0 - probability, 1e-12), 1.0 - 1e-12))
        for probability, actual in values
    )
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for probability, actual in values:
        bins[min(int(probability * 5), 4)].append((probability, actual))
    calibration_error = sum(
        (len(group) / len(values))
        * abs(mean(probability for probability, _ in group) - mean(actual for _, actual in group))
        for group in bins.values()
    )
    return {
        "brier_score": brier_score,
        "log_loss": log_loss,
        "expected_calibration_error_5_bins": calibration_error,
    }


def enforce_probability_order(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        win = float(row["preferred_win_probability"])
        podium = float(row["preferred_podium_probability"])
        top_10 = float(row["preferred_top_10_probability"])
        top_10 = max(top_10, podium, win)
        podium = min(max(podium, win), top_10)
        win = min(win, podium)
        row["preferred_win_probability"] = win
        row["preferred_podium_probability"] = podium
        row["preferred_top_10_probability"] = top_10


def fit_final_models(
    rows: list[dict[str, Any]], selected: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if selected["model_family"] == "pairwise_logistic":
        cutoff = next(group for group in selection_groups(rows) if group[0]["round"] == 6)
        scheme = selected["history_scheme"]
        c_value = float(str(selected["parameters"]).split("=")[1])
        training = history_rows(rows, cutoff[0]["race_date"], True, scheme)
        model: Any = fit_pairwise(training, scheme, c_value)
        predictions = []
        scores = []
        for group in test_groups(rows):
            race_predictions = rank_pairwise(model, group, scheme, selected["model_id"])
            predictions.extend(race_predictions)
            scores.append(score_race(race_predictions))
        return model, training, predictions, scores
    model, training, predictions, scores = frozen_test(rows, "practice", 0.70)
    return model, training, predictions, scores


def fit_pairwise_test(
    rows: list[dict[str, Any]], experiment: dict[str, Any]
) -> tuple[PairwiseRankingModel, list[dict[str, Any]], list[dict[str, Any]]]:
    cutoff = next(group for group in selection_groups(rows) if group[0]["round"] == 6)
    scheme = str(experiment["history_scheme"])
    c_value = float(str(experiment["parameters"]).split("=")[1])
    training = history_rows(rows, cutoff[0]["race_date"], True, scheme)
    model = fit_pairwise(training, scheme, c_value)
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for group in test_groups(rows):
        race_predictions = rank_pairwise(model, group, scheme, str(experiment["model_id"]))
        predictions.extend(race_predictions)
        scores.append(score_race(race_predictions))
    return model, predictions, scores


def probability_rows_for_groups(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    selected_simulation: dict[str, Any],
    simulations: int,
    include_cutoff: bool,
    fixed_history_cutoff: str | None = None,
) -> list[dict[str, Any]]:
    scores = {group[0]["race_id"]: scores_by_driver(group) for group in race_groups(predictions)}
    output = []
    for group in groups:
        race_id = group[0]["race_id"]
        output.extend(
            simulate_race(
                group,
                scores[race_id],
                float(selected_simulation["driver_sigma"]),
                float(selected_simulation["constructor_sigma"]),
                global_dnf_rate(
                    rows,
                    fixed_history_cutoff or group[0]["race_date"],
                    include_cutoff,
                ),
                simulations,
                stable_seed(race_id, simulations),
            )
        )
    return output


def raw_probability_signature(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["race_id"],
            row["driver_id"],
            row["raw_win_probability"],
            row["raw_podium_probability"],
            row["raw_top_10_probability"],
            row["raw_dnf_probability"],
            row["expected_position"],
        )
        for row in rows
    ]


def train_and_evaluate(feature_dir: Path, evaluation_dir: Path, model_dir: Path) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    rows = load_rows(feature_dir)
    ranking_experiments, selected_ranking, selection_predictions = select_ranking_model(rows)
    selected_selection_predictions = selection_predictions[selected_ranking["model_id"]]
    simulation_experiments, selected_simulation = choose_simulation(
        rows, selected_selection_predictions
    )
    selection_probability_rows = probability_rows_for_groups(
        rows,
        selected_selection_predictions,
        selection_groups(rows),
        selected_simulation,
        SELECTION_SIMULATIONS,
        False,
    )
    calibration, calibration_comparison = select_calibration(selection_probability_rows)
    model, training_rows, test_predictions, test_ranking_scores = fit_final_models(
        rows, selected_ranking
    )
    fixed_test_cutoff = max(row["race_date"] for row in training_rows)
    test_probability_rows = probability_rows_for_groups(
        rows,
        test_predictions,
        test_groups(rows),
        selected_simulation,
        TEST_SIMULATIONS,
        True,
        fixed_test_cutoff,
    )
    reproduced_probability_rows = probability_rows_for_groups(
        rows,
        test_predictions,
        test_groups(rows),
        selected_simulation,
        TEST_SIMULATIONS,
        True,
        fixed_test_cutoff,
    )
    fixed_seed_reproducible = raw_probability_signature(
        test_probability_rows
    ) == raw_probability_signature(reproduced_probability_rows)
    add_calibrated_probabilities(test_probability_rows, calibration)
    simulation_test_scores = [
        simulation_order_score(group, "race_simulation")
        for group in race_groups(test_probability_rows)
    ]
    current_model, _, _, current_test_scores = frozen_test(rows, "practice", 0.70)
    del current_model
    best_pairwise = sorted(
        (row for row in ranking_experiments if row["model_family"] == "pairwise_logistic"),
        key=lambda row: (
            -row["average_spearman_correlation"],
            row["average_mean_absolute_position_error"],
            row["model_id"],
        ),
    )[0]
    pairwise_model, _, pairwise_test_scores = fit_pairwise_test(rows, best_pairwise)
    probability_scores: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        raw_metrics = probability_metrics(
            test_probability_rows, outcome, f"raw_{outcome}_probability"
        )
        calibrated_metrics = probability_metrics(
            test_probability_rows, outcome, f"calibrated_{outcome}_probability"
        )
        retained = (
            calibration[outcome]["use_calibration"]
            and calibrated_metrics["brier_score"] < raw_metrics["brier_score"]
        )
        calibration[outcome]["retained_after_test"] = retained
        for row in test_probability_rows:
            row[f"preferred_{outcome}_probability"] = row[
                f"calibrated_{outcome}_probability" if retained else f"raw_{outcome}_probability"
            ]
    enforce_probability_order(test_probability_rows)
    for outcome in OUTCOMES:
        raw_metrics = probability_metrics(
            test_probability_rows, outcome, f"raw_{outcome}_probability"
        )
        calibrated_metrics = probability_metrics(
            test_probability_rows, outcome, f"calibrated_{outcome}_probability"
        )
        for method, metrics in (
            ("raw", raw_metrics),
            ("calibrated", calibrated_metrics),
            (
                "preferred",
                probability_metrics(
                    test_probability_rows,
                    outcome,
                    f"preferred_{outcome}_probability",
                ),
            ),
        ):
            probability_scores.append(
                {
                    "outcome": outcome,
                    "method": method,
                    **metrics,
                }
            )
    ranking_summary = summarize(test_ranking_scores)
    current_summary = summarize(current_test_scores)
    pairwise_summary = summarize(pairwise_test_scores)
    simulation_summary = summarize(simulation_test_scores)
    pairwise_by_race = {row["race_id"]: row for row in pairwise_test_scores}
    current_by_race = {row["race_id"]: row for row in current_test_scores}
    pairwise_test_wins = sum(
        pairwise_by_race[race_id]["spearman_correlation"]
        > current_by_race[race_id]["spearman_correlation"]
        for race_id in pairwise_by_race
    )
    pairwise_selected = selected_ranking["model_family"] == "pairwise_logistic"
    pairwise_test_improves = (
        pairwise_summary["average_spearman_correlation"]
        > current_summary["average_spearman_correlation"]
        and pairwise_test_wins >= 3
    )
    model_path = model_dir / "ranking_model.joblib"
    joblib.dump(model, model_path)
    pairwise_model_path = model_dir / "pairwise_candidate.joblib"
    joblib.dump(pairwise_model, pairwise_model_path)
    bundle = {
        "ranking_model": model,
        "model_family": selected_ranking["model_family"],
        "history_scheme": selected_ranking["history_scheme"],
        "feature_columns": (
            ranking_columns(selected_ranking["history_scheme"])
            if pairwise_selected
            else model_columns("practice")
        ),
        "simulation": {
            "driver_sigma": selected_simulation["driver_sigma"],
            "constructor_sigma": selected_simulation["constructor_sigma"],
            "simulations": TEST_SIMULATIONS,
            "random_seed": RANDOM_SEED,
            "dnf_probability": ("70% rolling constructor DNF rate and 30% earlier field DNF rate"),
        },
        "calibration": calibration,
    }
    probability_model_path = model_dir / "probability_model.joblib"
    joblib.dump(bundle, probability_model_path)
    ranking_columns_output = list(ranking_experiments[0])
    write_csv(evaluation_dir / "ranking_selection.csv", ranking_experiments, ranking_columns_output)
    write_csv(
        evaluation_dir / "ranking_test_scores.csv",
        test_ranking_scores,
        list(test_ranking_scores[0]),
    )
    write_csv(
        evaluation_dir / "current_model_test_scores.csv",
        current_test_scores,
        list(current_test_scores[0]),
    )
    write_csv(
        evaluation_dir / "pairwise_test_scores.csv",
        pairwise_test_scores,
        list(pairwise_test_scores[0]),
    )
    write_csv(
        evaluation_dir / "simulation_selection.csv",
        simulation_experiments,
        list(simulation_experiments[0]),
    )
    write_csv(
        evaluation_dir / "simulation_test_scores.csv",
        simulation_test_scores,
        list(simulation_test_scores[0]),
    )
    probability_columns = list(test_probability_rows[0])
    write_csv(
        evaluation_dir / "probability_predictions.csv",
        test_probability_rows,
        probability_columns,
    )
    write_csv(
        evaluation_dir / "probability_scores.csv",
        probability_scores,
        list(probability_scores[0]),
    )
    write_csv(
        evaluation_dir / "calibration_selection.csv",
        calibration_comparison,
        list(calibration_comparison[0]),
    )
    metadata = {
        "model_id": selected_ranking["model_id"],
        "model_family": selected_ranking["model_family"],
        "history_scheme": selected_ranking["history_scheme"],
        "feature_columns": bundle["feature_columns"],
        "parameters": selected_ranking["parameters"],
        "random_seed": RANDOM_SEED,
        "training_result_filter": "CLASSIFIED",
        "training_cutoff_season": 2026,
        "training_cutoff_round": 6,
        "training_cutoff_date": max(row["race_date"] for row in training_rows),
        "training_rows": len(training_rows),
        "test_rounds": list(TEST_ROUNDS),
        "ranking_test": ranking_summary,
        "current_model_test": current_summary,
        "best_pairwise_candidate": best_pairwise["model_id"],
        "pairwise_test": pairwise_summary,
        "simulation_test": simulation_summary,
        "pairwise_selected_on_selection": pairwise_selected,
        "pairwise_repeatable_test_improvement": pairwise_test_improves,
        "pairwise_test_races_beating_current": pairwise_test_wins,
        "weather_included": False,
        "dnf_affects_driver_pace": False,
    }
    write_json(model_dir / "model_metadata.json", metadata)
    write_json(model_dir / "calibration.json", calibration)
    checks = {
        "selection_rounds_are_one_to_six": sorted(
            {row["round"] for row in selected_selection_predictions}
        )
        == list(range(1, 7)),
        "test_rounds_are_seven_to_eleven": sorted({row["round"] for row in test_probability_rows})
        == list(TEST_ROUNDS),
        "training_contains_only_classified_results": all(
            row["actual_classification_status"] == "CLASSIFIED" for row in training_rows
        ),
        "one_simulated_winner_per_race": all(
            math.isclose(sum(row["raw_win_probability"] for row in group), 1.0, abs_tol=1e-9)
            for group in race_groups(test_probability_rows)
        ),
        "probabilities_are_bounded": all(
            0.0 <= float(row[column]) <= 1.0
            for row in test_probability_rows
            for column in probability_columns
            if "probability" in column
        ),
        "preferred_probabilities_are_ordered": all(
            row["preferred_win_probability"]
            <= row["preferred_podium_probability"]
            <= row["preferred_top_10_probability"]
            for row in test_probability_rows
        ),
        "simulation_orders_are_complete": all(
            len(group) == len({row["driver_id"] for row in group})
            for group in race_groups(test_probability_rows)
        ),
        "fixed_seed_is_reproducible": fixed_seed_reproducible,
        "saved_models_exist": all(
            path.is_file() for path in (model_path, pairwise_model_path, probability_model_path)
        ),
        "no_weather_inputs": all("weather" not in column.lower() for column in rows[0]),
        "dnf_is_separate_from_pace_training": all(
            row["actual_classification_status"] == "CLASSIFIED" for row in training_rows
        ),
    }
    errors = [f"Failed check: {name}" for name, passed in checks.items() if not passed]
    validation = {
        "status": "passed" if not errors else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "errors": errors,
    }
    write_json(evaluation_dir / "validation_report.json", validation)
    summary = {
        "selected_ranking_model": selected_ranking["model_id"],
        "ranking_model_retained": pairwise_selected and pairwise_test_improves,
        "ranking_test": ranking_summary,
        "current_model_test": current_summary,
        "best_pairwise_candidate": best_pairwise["model_id"],
        "pairwise_test": pairwise_summary,
        "simulation_test": simulation_summary,
        "pairwise_test_races_beating_current": pairwise_test_wins,
        "selected_simulation": {
            "driver_sigma": selected_simulation["driver_sigma"],
            "constructor_sigma": selected_simulation["constructor_sigma"],
            "simulations_per_race": TEST_SIMULATIONS,
        },
        "calibration": {
            outcome: {
                "use_calibration": calibration[outcome]["use_calibration"],
                "retained_after_test": calibration[outcome]["retained_after_test"],
                "selection_raw_brier": calibration[outcome]["selection_raw_brier"],
                "selection_grouped_cv_brier": calibration[outcome]["selection_grouped_cv_brier"],
            }
            for outcome in OUTCOMES
        },
    }
    write_json(evaluation_dir / "model_summary.json", summary)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_dataset": {
            "path": portable_path(feature_dir / "post_qualifying_features.csv", PROJECT_ROOT),
            "sha256": sha256_file(feature_dir / "post_qualifying_features.csv"),
        },
        "selection_rounds": list(range(1, 7)),
        "test_rounds": list(TEST_ROUNDS),
        "training_cutoff": {
            "season": 2026,
            "round": 6,
            "date": fixed_test_cutoff,
        },
        "random_seed": RANDOM_SEED,
        "weather_included": False,
        "simulation_interpretation": (
            "Probabilities express model uncertainty, constructor variation, reliability, and "
            "ordinary race incidents. They are not guarantees."
        ),
        "scikit_learn_version": sklearn.__version__,
        "evaluation_artifacts": {
            name: sha256_file(evaluation_dir / name)
            for name in (
                "ranking_selection.csv",
                "ranking_test_scores.csv",
                "current_model_test_scores.csv",
                "pairwise_test_scores.csv",
                "simulation_selection.csv",
                "simulation_test_scores.csv",
                "probability_predictions.csv",
                "probability_scores.csv",
                "calibration_selection.csv",
                "validation_report.json",
                "model_summary.json",
            )
        },
        "model_artifacts": {
            name: sha256_file(model_dir / name)
            for name in (
                "ranking_model.joblib",
                "pairwise_candidate.joblib",
                "probability_model.joblib",
                "model_metadata.json",
                "calibration.json",
            )
        },
    }
    write_json(evaluation_dir / "evaluation_manifest.json", manifest)
    if errors:
        raise ValueError("Probability evaluation validation failed.")
    print(f"Selected ranking model: {selected_ranking['model_id']}")
    print(f"Ranking average Spearman: {ranking_summary['average_spearman_correlation']:.3f}")
    print(f"Current model average Spearman: {current_summary['average_spearman_correlation']:.3f}")
    print(f"Validation status: {validation['status']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ranking models and evaluate calibrated race probabilities."
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        args.feature_dir.resolve(),
        args.evaluation_dir.resolve(),
        args.model_dir.resolve(),
    )


if __name__ == "__main__":
    main()
