import numpy as np
import pandas as pd

from anomalybench.analytics.hyperparam import grid_search


def test_grid_search_returns_valid_result():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(100, 2))
    anomalies = rng.normal(loc=5, size=(10, 2))
    X = np.vstack([normal, anomalies])
    y = np.array([0] * len(normal) + [1] * len(anomalies))
    df = pd.DataFrame(X)
    labels = pd.Series(y)
    params, score = grid_search(
        "isolation_forest", {"n_estimators": [10, 20]}, df, labels, cv=2
    )
    assert params["n_estimators"] in [10, 20]
    assert 0.0 <= score <= 1.0


def test_grid_search_respects_score_orientation():
    """Lower-is-more-anomalous detectors must not be ranked upside down.

    grid_search previously scored raw ``score()`` output, so Isolation Forest
    produced an AUC near zero and the worst parameters won.
    """

    rng = np.random.default_rng(0)
    normal = rng.normal(size=(200, 2))
    anomalies = rng.normal(loc=6, scale=0.3, size=(20, 2))
    df = pd.DataFrame(np.vstack([normal, anomalies]))
    labels = pd.Series([0] * len(normal) + [1] * len(anomalies))

    _, score = grid_search(
        "isolation_forest", {"n_estimators": [10, 50]}, df, labels, cv=3
    )

    assert score > 0.8
