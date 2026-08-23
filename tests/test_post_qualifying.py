from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from f1_race_predictor.features import FEATURE_COLUMNS as PRE_WEEKEND_FEATURES
from f1_race_predictor.grid_snapshot import create_snapshot
from f1_race_predictor.post_qualifying_features import build_features
from f1_race_predictor.post_qualifying_model import BlendedRankingModel
from f1_race_predictor.weekend_ingestion import scale_profiles, seconds


class ConstantEstimator:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(len(matrix), 0.5)


def test_ergast_qualifying_time_is_parsed() -> None:
    assert seconds("1:29.708") == pytest.approx(89.708)


def pre_weekend_row(race_id: str, driver_id: str, constructor_id: str) -> dict[str, Any]:
    season, race_round = (int(value) for value in race_id.split("-"))
    row: dict[str, Any] = {
        "race_id": race_id,
        "season": season,
        "round": race_round,
        "race_date": f"{season}-03-{race_round + 1:02d}",
        "circuit_id": f"circuit_{race_round}",
        "driver_id": driver_id,
        "driver_name": driver_id.title(),
        "constructor_id": constructor_id,
        "constructor_name": constructor_id.title(),
    }
    row.update({column: 0.5 for column in PRE_WEEKEND_FEATURES})
    return row


def result_row(
    race_id: str,
    driver_id: str,
    constructor_id: str,
    position: int,
    status: str,
) -> dict[str, str]:
    season, race_round = (int(value) for value in race_id.split("-"))
    return {
        "race_id": race_id,
        "season": str(season),
        "round": str(race_round),
        "race_date": f"{season}-03-{race_round + 1:02d}",
        "driver_id": driver_id,
        "constructor_id": constructor_id,
        "normalized_position": str(position),
        "classification_status": status,
    }


def qualifying_row(race_id: str, driver_id: str, position: int) -> dict[str, str]:
    return {
        "race_id": race_id,
        "driver_id": driver_id,
        "official_grid_position": str(position),
        "qualifying_position": str(position),
        "qualifying_gap_percent": str(position - 1),
        "qualifying_teammate_delta_percent": "0",
        "grid_change_from_qualifying": "0",
        "qualifying_missing": "False",
        "qualifying_cutoff": "2026-03-01T12:00:00+00:00",
        "grid_recorded_at": "2026-03-02T10:00:00+00:00",
    }


def practice_row(race_id: str, driver_id: str) -> dict[str, str]:
    return {
        "race_id": race_id,
        "driver_id": driver_id,
        "practice_laps": "12",
        "practice_gap_percent": "1",
        "practice_sector_score": "90",
        "long_run_laps": "6",
        "long_run_gap_percent": "1",
        "degradation_percent_per_lap": "0.05",
        "compounds_used": "MEDIUM|SOFT",
        "practice_missing": "False",
    }


def profile_row(race_id: str) -> dict[str, str]:
    return {
        "race_id": race_id,
        "straight_demand": "0.7",
        "cornering_demand": "0.5",
        "braking_demand": "0.4",
        "tyre_stress": "0.3",
        "profile_missing": "False",
    }


def synthetic_inputs() -> tuple[list[dict[str, Any]], ...]:
    pre = []
    results = []
    qualifying = []
    practice = []
    profiles = []
    for race_id in ("2026-01", "2026-02"):
        profiles.append(profile_row(race_id))
        for position, (driver_id, constructor_id) in enumerate(
            (("driver_a", "team_a"), ("driver_b", "team_b")), start=1
        ):
            pre.append(pre_weekend_row(race_id, driver_id, constructor_id))
            status = "DNF" if race_id == "2026-01" and driver_id == "driver_a" else "CLASSIFIED"
            results.append(result_row(race_id, driver_id, constructor_id, position, status))
            qualifying.append(qualifying_row(race_id, driver_id, position))
            practice.append(practice_row(race_id, driver_id))
    return pre, results, qualifying, practice, profiles


def test_current_result_does_not_change_current_post_qualifying_features() -> None:
    inputs = synthetic_inputs()
    original = build_features(*inputs)
    changed_results = copy.deepcopy(inputs[1])
    for row in changed_results:
        if row["race_id"] == "2026-02":
            row["classification_status"] = "DNF"
            row["normalized_position"] = "1"
    changed = build_features(inputs[0], changed_results, inputs[2], inputs[3], inputs[4])
    original_race = [row for row in original if row["race_id"] == "2026-02"]
    changed_race = [row for row in changed if row["race_id"] == "2026-02"]
    assert original_race == changed_race
    driver_a = next(row for row in original_race if row["driver_id"] == "driver_a")
    assert driver_a["driver_recent_dnf_rate_10"] == 1.0


def test_circuit_scores_do_not_depend_on_other_races() -> None:
    first = {
        "high_speed_fraction": 0.6,
        "corner_density": 2.5,
        "brake_event_density": 1.5,
        "tyre_degradation_percent_per_lap": 0.1,
    }
    initial = [copy.deepcopy(first)]
    scale_profiles(initial)
    with_future = [
        copy.deepcopy(first),
        {
            "high_speed_fraction": 0.9,
            "corner_density": 4.5,
            "brake_event_density": 2.5,
            "tyre_degradation_percent_per_lap": 0.18,
        },
    ]
    scale_profiles(with_future)
    assert initial[0] == with_future[0]


def test_weekend_weight_controls_the_final_score() -> None:
    model = BlendedRankingModel(ConstantEstimator(), [0], 1, 0.70)
    scores = model.predict(np.asarray([[0.0, 0.8], [0.0, 0.2]], dtype=float))
    assert scores.tolist() == pytest.approx([0.71, 0.29])


def test_grid_snapshot_is_complete_and_cannot_be_replaced(tmp_path: Path) -> None:
    input_file = tmp_path / "grid.csv"
    with input_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["driver_id", "official_grid_position", "penalty_note"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "driver_id": "driver_a",
                    "official_grid_position": 1,
                    "penalty_note": "",
                },
                {
                    "driver_id": "driver_b",
                    "official_grid_position": 2,
                    "penalty_note": "Five-place penalty",
                },
            ]
        )
    output_file = tmp_path / "snapshot.json"
    create_snapshot(
        input_file,
        output_file,
        "2026-12",
        "2026-08-01T12:00:00+00:00",
        "2026-08-01T13:00:00+00:00",
    )
    with pytest.raises(FileExistsError):
        create_snapshot(
            input_file,
            output_file,
            "2026-12",
            "2026-08-01T12:00:00+00:00",
            "2026-08-01T13:00:00+00:00",
        )
