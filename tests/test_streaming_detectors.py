from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("river")

from analytics.detectors.streaming import (  # noqa: E402
    HalfSpaceTreesDetector,
    OnlineIsolationForestDetector,
    RandomCutForestDetector,
)


def test_half_space_trees_scores_numpy_input() -> None:
    detector = HalfSpaceTreesDetector().fit(
        np.array([[0.0], [0.2], [0.4]], dtype=float),
        seed=42,
        window_size=2,
    )

    scores = detector.score(np.array([[0.1], [0.9]], dtype=float))

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)


def test_online_isolation_forest_alias_scores_dataframe_input() -> None:
    detector = OnlineIsolationForestDetector().fit(
        pd.DataFrame({"feature": [0.0, 0.2, 0.4]}),
        seed=42,
        window_size=2,
    )

    scores = detector.score(pd.DataFrame({"feature": [0.1, 0.9]}))

    assert len(scores) == 2
    assert detector.get_name() == "Online Isolation Forest"


def test_random_cut_forest_reports_unsupported_river_model() -> None:
    detector = RandomCutForestDetector()

    with pytest.raises(ImportError, match="RandomCutForest is not available"):
        detector.fit(np.array([[0.0], [1.0]], dtype=float))
