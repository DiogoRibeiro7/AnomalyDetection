"""Simple hyperparameter search utilities."""

from __future__ import annotations

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

from analytics.detectors import get_detector_class


def grid_search(detector_name: str, param_grid, X, y):
    """Perform a brute-force grid search over ``param_grid``.

    Parameters
    ----------
    detector_name:
        Name of the detector registered in :mod:`analytics.detectors`.
    param_grid:
        Dictionary defining the parameter grid.
    X, y:
        Data and ground-truth labels.
    Returns
    -------
    tuple[dict, float]
        Best parameter set and corresponding ROC-AUC score.
    """
    best_params = None
    best_auc = -1.0
    for params in ParameterGrid(param_grid):
        detector = get_detector_class(detector_name)()
        scores = detector.detect_anomalies(X, **params)
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_auc = auc
            best_params = params
    return best_params, best_auc
