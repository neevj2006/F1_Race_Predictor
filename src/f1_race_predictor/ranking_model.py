from __future__ import annotations

from typing import Any

import numpy as np


class PairwiseRankingModel:
    """Score drivers using a classifier trained on within-race driver pairs."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.decision_function(matrix), dtype=float)
