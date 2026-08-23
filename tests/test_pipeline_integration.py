from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from f1_race_predictor.features import generate_feature_dataset
from f1_race_predictor.normalization import normalize_dataset
from f1_race_predictor.training import (
    feature_columns,
    final_training_rows,
    fit_estimator,
    load_rows,
    rank_race,
)

DRIVERS = (
    ("driver_a", "Driver", "A", "team_a"),
    ("driver_b", "Driver", "B", "team_b"),
    ("hadjar", "Isack", "Hadjar", "team_c"),
)


def raw_race(season: int, race_round: int) -> dict[str, object]:
    results = []
    order = list(DRIVERS)
    if race_round % 2 == 0:
        order[0], order[1] = order[1], order[0]
    for position, (driver_id, given_name, family_name, constructor_id) in enumerate(order, start=1):
        status = "Finished"
        position_text = str(position)
        laps = "57"
        if season == 2025 and race_round == 1 and driver_id == "hadjar":
            status = "Finished"
        elif season == 2026 and race_round == 4 and driver_id == "driver_b":
            status = "Accident"
            position_text = "R"
            laps = "20"
        results.append(
            {
                "number": str(position),
                "position": str(position),
                "positionText": position_text,
                "points": "0",
                "grid": str(position),
                "laps": laps,
                "status": status,
                "Driver": {
                    "driverId": driver_id,
                    "code": driver_id[:3].upper(),
                    "givenName": given_name,
                    "familyName": family_name,
                    "dateOfBirth": "2000-01-01",
                    "nationality": "Test",
                },
                "Constructor": {
                    "constructorId": constructor_id,
                    "name": constructor_id.title(),
                    "nationality": "Test",
                },
            }
        )
    return {
        "season": str(season),
        "round": str(race_round),
        "raceName": f"Race {season}-{race_round}",
        "date": f"{season}-03-{race_round + 1:02d}",
        "Circuit": {
            "circuitId": f"circuit_{race_round % 3}",
            "circuitName": "Test Circuit",
            "Location": {"locality": "Test", "country": "Test"},
        },
        "Results": results,
    }


def write_raw_fixture(root: Path) -> None:
    rounds_by_season = {2023: 1, 2024: 1, 2025: 1, 2026: 11}
    for season, round_count in rounds_by_season.items():
        year_dir = root / str(season)
        year_dir.mkdir(parents=True)
        races = [raw_race(season, race_round) for race_round in range(1, round_count + 1)]
        (year_dir / "race_results.json").write_text(
            json.dumps(races),
            encoding="utf-8",
        )


def test_small_raw_fixture_reaches_valid_prediction(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    feature_dir = tmp_path / "features"
    write_raw_fixture(raw_dir)
    normalization_report = normalize_dataset(
        raw_dir,
        processed_dir,
        (2023, 2024, 2025, 2026),
    )
    assert normalization_report["status"] == "passed"

    feature_report = generate_feature_dataset(processed_dir / "race_results.csv", feature_dir)
    assert feature_report["status"] == "passed"
    rows = load_rows(feature_dir)
    config: dict[str, Any] = {
        "model_id": "integration_ridge",
        "model_family": "ridge",
        "feature_set": "full",
        "history_scheme": "unweighted",
        "feature_columns": feature_columns("full", "unweighted"),
        "parameters": {"alpha": 10.0},
    }
    training_rows = final_training_rows(rows, "2026-03-07", "unweighted")
    estimator = fit_estimator(training_rows, config)
    race_rows = [row for row in rows if row["race_id"] == "2026-07"]
    prediction = rank_race(estimator, race_rows, config, config["model_id"])
    assert len(prediction) == 3
    assert [row["predicted_position"] for row in prediction] == [1, 2, 3]
    assert len({row["driver_id"] for row in prediction}) == 3
