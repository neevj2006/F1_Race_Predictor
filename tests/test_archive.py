from __future__ import annotations

import json
from pathlib import Path

import pytest

from f1_race_predictor.archive import archive_prediction


def test_archived_prediction_cannot_be_overwritten(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    model_dir = registry_root / "model-v1"
    model_dir.mkdir(parents=True)
    (model_dir / "registry.json").write_text("{}\n", encoding="utf-8")
    prediction = {
        "record_type": "historical_backtest",
        "created_at": "2026-08-23T00:00:00+00:00",
        "data_cutoff": "2026-07-25T00:00:00+00:00",
        "race": {"race_id": "2026-11"},
        "model_version": "model-v1",
        "drivers": [
            {"position": 1, "driver_id": "driver_a"},
            {"position": 2, "driver_id": "driver_b"},
        ],
    }
    archive_root = tmp_path / "archive"
    destination = archive_prediction(
        "backtest-2026-11",
        prediction,
        archive_root,
        registry_root,
        "2026-08-23T00:00:00+00:00",
    )
    archived = json.loads((destination / "prediction.json").read_text(encoding="utf-8"))
    assert archived["prediction_id"] == "backtest-2026-11"
    with pytest.raises(FileExistsError):
        archive_prediction(
            "backtest-2026-11",
            prediction,
            archive_root,
            registry_root,
        )
