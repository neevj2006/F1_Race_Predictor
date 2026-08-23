from __future__ import annotations

import pytest

from f1_race_predictor.features import build_features, parse_season_weights
from f1_race_predictor.training import training_rows_before


def source_row(
    race_id: str,
    race_date: str,
    driver_id: str,
    position: int,
    status: str = "CLASSIFIED",
) -> dict[str, object]:
    season, race_round = (int(value) for value in race_id.split("-"))
    return {
        "race_id": race_id,
        "season": season,
        "round": race_round,
        "race_date": race_date,
        "circuit_id": "test_circuit",
        "driver_id": driver_id,
        "driver_name": driver_id.title(),
        "constructor_id": f"team_{driver_id}",
        "constructor_name": f"Team {driver_id.title()}",
        "normalized_position": position,
        "official_position": position,
        "classification_status": status,
    }


def test_season_weights_must_increase_toward_recent_seasons() -> None:
    assert parse_season_weights("2023=0.25,2024=0.5,2025=0.75,2026=1") == {
        2023: 0.25,
        2024: 0.5,
        2025: 0.75,
        2026: 1.0,
    }
    with pytest.raises(ValueError):
        parse_season_weights("2023=1,2024=0.5")


def test_rookie_uses_neutral_fallbacks() -> None:
    rows = [
        source_row("2023-01", "2023-03-05", "veteran", 1),
        source_row("2023-01", "2023-03-05", "other", 2),
        source_row("2024-01", "2024-03-03", "rookie", 1),
        source_row("2024-01", "2024-03-03", "veteran", 2),
    ]
    features = build_features(rows)
    rookie = next(row for row in features if row["driver_id"] == "rookie")
    assert rookie["driver_prior_starts"] == 0
    assert rookie["driver_recent_teammate_delta_5"] == 0.0
    assert rookie["driver_recent_finish_score_5"] == pytest.approx(0.5)


def test_training_cutoff_excludes_later_and_dnf_rows() -> None:
    rows = [
        {
            "race_id": "2026-01",
            "race_date": "2026-03-01",
            "actual_classification_status": "CLASSIFIED",
        },
        {
            "race_id": "2026-02",
            "race_date": "2026-03-08",
            "actual_classification_status": "DNF",
        },
        {
            "race_id": "2026-03",
            "race_date": "2026-03-15",
            "actual_classification_status": "CLASSIFIED",
        },
    ]
    selected = training_rows_before(rows, "2026-03-15", "unweighted")
    assert [row["race_id"] for row in selected] == ["2026-01"]
