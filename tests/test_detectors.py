import numpy as np

from analytics.detectors import get_detector_class


def test_isolation_forest_scores_new_data():
    X = np.random.randn(20, 2)
    detector = get_detector_class("isolation_forest")().fit(X)
    scores = detector.score(X)
    assert len(scores) == 20
