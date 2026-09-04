"""Regression tests for benchmark score and detector-selection contracts."""

from __future__ import annotations

import pytest

from anomalybench.analytics.base import BaseDetector, OrientedScores
from anomalybench.analytics.exceptions import UnknownDetectorError
from anomalybench.cli import _resolve_detector_entries


class LowerScoreDetector(BaseDetector):
    score_orientation = "lower_is_more_anomalous"

    def get_name(self) -> str:
        return "Lower Score Detector"

    def fit(self, data: object, **params: object) -> LowerScoreDetector:
        return self

    def score(self, data: object) -> list[float]:
        return [0.9, 0.1]


def test_detect_anomalies_preserves_score_orientation_metadata() -> None:
    scores = LowerScoreDetector().detect_anomalies(object())

    assert isinstance(scores, OrientedScores)
    assert scores.score_orientation == "lower_is_more_anomalous"
    assert scores.tolist() == [0.9, 0.1]


def test_unknown_detector_selection_fails_instead_of_expanding_to_all() -> None:
    with pytest.raises(
        UnknownDetectorError, match="Unknown detector 'isolation_forrest'"
    ):
        _resolve_detector_entries(["isolation_forrest"])
