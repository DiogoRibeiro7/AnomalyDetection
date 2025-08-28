import numpy as np

from analytics.detectors import get_detector_class


def test_isolation_forest_scores_new_data():
    X = np.random.randn(20, 2)
    detector = get_detector_class("isolation_forest")().fit(X)
    scores = detector.score(X)
    assert len(scores) == 20


def test_hbos_vectorized_score():
    X = np.random.rand(10, 3)
    det = get_detector_class("hbos")().fit(X)
    scores = det.score(X)
    assert len(scores) == 10


def test_knn_scores_new_data():
    X = np.random.rand(10, 2)
    Y = np.random.rand(5, 2)
    det = get_detector_class("knn")().fit(X)
    scores = det.score(Y)
    assert len(scores) == 5
