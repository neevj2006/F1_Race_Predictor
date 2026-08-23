from __future__ import annotations

from typing import Any

import numpy as np


class BlendedRankingModel:
    def __init__(
        self,
        support_estimator: Any,
        support_indices: list[int],
        weekend_score_index: int,
        weekend_weight: float,
    ) -> None:
        self.support_estimator = support_estimator
        self.support_indices = support_indices
        self.weekend_score_index = weekend_score_index
        self.weekend_weight = weekend_weight

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        support = np.clip(
            self.support_estimator.predict(matrix[:, self.support_indices]),
            0.0,
            1.0,
        )
        weekend = matrix[:, self.weekend_score_index]
        return self.weekend_weight * weekend + (1.0 - self.weekend_weight) * support
