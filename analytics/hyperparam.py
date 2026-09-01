"""Hyperparameter search utilities for anomaly detectors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from analytics.detectors import get_detector_class
from benchmarks.metrics import canonicalize_anomaly_scores


def grid_search(
    detector_name: str,
    param_grid: dict[str, Iterable],
    X,
    y,
    cv: int = 3,
):
    """Evaluate parameter combinations via stratified k-fold validation.

    Parameters
    ----------
    detector_name:
        Registered detector identifier.
    param_grid:
        Mapping of parameter names to a list of candidate values.
    X, y:
        Feature matrix and ground-truth labels. ``X`` should support ``iloc``
        indexing (e.g., :class:`pandas.DataFrame`).
    cv:
        Number of cross-validation folds.

    Returns
    -------
    tuple[dict, float]
        Best parameter set and corresponding mean ROC-AUC score.

    Scores are canonicalised to ``higher_is_more_anomalous`` before scoring,
    so detectors whose raw scores run the other way are ranked correctly.
    """

    best_params: dict[str, object] | None = None
    best_score = -np.inf
    DetectorClass = get_detector_class(detector_name)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, test_idx in splitter.split(X, y):
            detector = DetectorClass()
            detector.fit(X.iloc[train_idx], **params)
            scores = canonicalize_anomaly_scores(
                detector.detect_anomalies(X.iloc[test_idx], **params)
            )
            fold_scores.append(roc_auc_score(y.iloc[test_idx], scores))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score
