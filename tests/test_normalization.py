from __future__ import annotations

import copy
from typing import Any

from f1_race_predictor.normalization import (
    assign_normalized_positions,
    classification_status,
    validate,
)


def result_row(
    driver_id: str,
    official_position: int,
    status: str,
    laps: int,
) -> dict[str, Any]:
    return {
        "race_id": "2025-01",
        "driver_id": driver_id,
        "driver_name": driver_id.title(),
        "driver_date_of_birth": "2000-01-01",
        "classification_status": status,
        "laps_completed": laps,
        "official_position": official_position,
        "normalized_position": 0,
        "status_corrected": driver_id == "hadjar",
    }


def test_exceptional_statuses_are_normalized() -> None:
    assert classification_status("R", "Accident", 30) == "DNF"
    assert classification_status("R", "Lapped", 55) == "NC"
    assert classification_status("R", "Did not start", 0) == "DNS"
    assert classification_status("R", "Disqualified", 55) == "DSQ"
    assert classification_status("4", "Finished", 57) == "CLASSIFIED"


def test_finishing_order_places_dns_and_dsq_last() -> None:
    rows = [
        result_row("classified", 4, "CLASSIFIED", 57),
        result_row("dnf", 2, "DNF", 40),
        result_row("hadjar", 1, "DNS", 0),
        result_row("dsq", 3, "DSQ", 57),
    ]
    assign_normalized_positions(rows)
    ordered = [
        row["driver_id"] for row in sorted(rows, key=lambda row: int(row["normalized_position"]))
    ]
    assert ordered == ["classified", "dnf", "hadjar", "dsq"]


def test_driver_identity_change_is_rejected() -> None:
    rows = [
        result_row("hadjar", 1, "DNS", 0),
        result_row("driver", 2, "CLASSIFIED", 57),
    ]
    second_race = copy.deepcopy(rows[1])
    second_race["race_id"] = "2025-02"
    second_race["driver_name"] = "Different Name"
    rows.append(second_race)
    assign_normalized_positions(rows)
    report = validate(rows)
    assert report["status"] == "failed"
    assert any("Driver identity changed" in error for error in report["errors"])
