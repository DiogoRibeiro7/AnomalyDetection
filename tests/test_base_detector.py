from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from analytics.base import BaseDetector, coerce_tabular_2d


class LifecycleDetector(BaseDetector):
    def get_name(self) -> str:
        return "Lifecycle Detector"

    def fit(self, data: object, **params: Any) -> LifecycleDetector:
        if params.get("fail"):
            raise ValueError("fit failed")
        return self

    def score(self, data: object) -> list[float]:
        return [1.0]


def test_detector_starts_unfitted_and_scores_only_after_fit() -> None:
    detector = LifecycleDetector()

    assert detector.is_fitted is False
    with pytest.raises(RuntimeError, match="must be fitted before calling score"):
        detector.score(object())

    assert detector.fit(object()) is detector
    assert detector.is_fitted is True
    assert detector.score(object()) == [1.0]


def test_failed_fit_does_not_mark_detector_as_fitted() -> None:
    detector = LifecycleDetector()

    with pytest.raises(ValueError, match="fit failed"):
        detector.fit(object(), fail=True)

    assert detector.is_fitted is False


def test_detect_anomalies_uses_fit_and_score_lifecycle() -> None:
    detector = LifecycleDetector()

    assert detector.detect_anomalies(object()) == [1.0]
    assert detector.is_fitted is True


def test_coerce_tabular_2d_validates_shape_and_empty_input() -> None:
    assert coerce_tabular_2d([[1, 2], [3, 4]]).shape == (2, 2)

    with pytest.raises(ValueError, match="2-D array"):
        coerce_tabular_2d([1, 2, 3])

    with pytest.raises(ValueError, match="at least one sample"):
        coerce_tabular_2d(np.empty((0, 2)))
