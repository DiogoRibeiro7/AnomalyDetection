import numpy as np
import pandas as pd

from analytics.hyperparam import grid_search


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
