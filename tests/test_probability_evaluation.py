from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from f1_race_predictor.probability_evaluation import (
    calibrated_probability,
    enforce_probability_order,
    fit_sigmoid,
    simulate_race,
)


def race_row(driver_id: str, constructor_id: str, position: int) -> dict[str, Any]:
    return {
        "race_id": "2026-11",
        "season": 2026,
        "round": 11,
        "race_date": "2026-07-26",
        "driver_id": driver_id,
        "driver_name": driver_id,
        "constructor_id": constructor_id,
        "constructor_name": constructor_id,
        "constructor_recent_dnf_rate_10_weekend": 0.10,
        "actual_position": position,
        "actual_classification_status": "CLASSIFIED",
    }


def test_simulation_is_reproducible_and_has_one_winner() -> None:
    rows = [race_row("driver_a", "team_a", 1), race_row("driver_b", "team_b", 2)]
    scores = {"driver_a": 1.0, "driver_b": 0.0}
    first = simulate_race(rows, scores, 0.5, 0.2, 0.1, 1000, 42)
    second = simulate_race(rows, scores, 0.5, 0.2, 0.1, 1000, 42)
    assert first == second
    assert sum(row["raw_win_probability"] for row in first) == pytest.approx(1.0)
    assert all(0.0 <= row["raw_dnf_probability"] <= 1.0 for row in first)


def test_sigmoid_calibration_returns_bounded_probabilities() -> None:
    rows: list[dict[str, Any]] = []
    for index, (probability, actual_position) in enumerate(
        ((0.1, 2), (0.3, 2), (0.7, 1), (0.9, 1))
    ):
        rows.append(
            {
                "race_id": f"2026-{index + 1:02d}",
                "raw_win_probability": probability,
                "actual_position": actual_position,
                "actual_classification_status": "CLASSIFIED",
            }
        )
    parameters = fit_sigmoid(rows, "win", 1.0)
    calibrated = np.asarray(
        [calibrated_probability(float(row["raw_win_probability"]), parameters) for row in rows]
    )
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert calibrated[-1] > calibrated[0]


def test_preferred_probabilities_follow_event_order() -> None:
    rows = [
        {
            "preferred_win_probability": 0.20,
            "preferred_podium_probability": 0.10,
            "preferred_top_10_probability": 0.15,
        }
    ]
    enforce_probability_order(rows)
    assert rows[0]["preferred_win_probability"] <= rows[0]["preferred_podium_probability"]
    assert rows[0]["preferred_podium_probability"] <= rows[0]["preferred_top_10_probability"]
