"""Tests for modern tabular anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataexcept import DataValidationError, HyperparameterError

from analytics.detectors.modern_tabular import (
    ECODDetector,
    RandomFeatureIsolationForestDetector,
    RandomNetworkDistillationDetector,
)


def _tabular_data(seed: int = 123) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(rng.normal(size=(48, 4)), columns=list("abcd"))
    test = rng.normal(size=(12, 4))
    return train, test


@pytest.mark.parametrize(
    ("detector_cls", "fit_kwargs"),
    [
        (ECODDetector, {}),
        (
            RandomNetworkDistillationDetector,
            {
                "representation_dim": 8,
                "hidden_layer_sizes": (8,),
                "max_iter": 1000,
                "random_state": 0,
            },
        ),
        (
            RandomFeatureIsolationForestDetector,
            {
                "representation_dim": 12,
                "n_estimators": 16,
                "random_state": 0,
            },
        ),
    ],
)
def test_modern_tabular_detectors_score_new_samples(detector_cls, fit_kwargs) -> None:
    train, test = _tabular_data()

    detector = detector_cls().fit(train, **fit_kwargs)
    scores = detector.score(test)

    assert detector.is_fitted
    assert detector.score_orientation == "higher_is_more_anomalous"
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()


def test_random_feature_isolation_forest_is_deterministic_with_seed() -> None:
    train, test = _tabular_data()
    fit_kwargs = {
        "representation_dim": 10,
        "n_estimators": 16,
        "random_state": 42,
    }

    first = RandomFeatureIsolationForestDetector().fit(train, **fit_kwargs).score(test)
    second = RandomFeatureIsolationForestDetector().fit(train, **fit_kwargs).score(test)

    np.testing.assert_allclose(first, second)


def test_random_network_distillation_rejects_invalid_representation_dim() -> None:
    train, _test = _tabular_data()

    with pytest.raises(HyperparameterError, match="representation_dim"):
        RandomNetworkDistillationDetector().fit(train, representation_dim=0)


def test_modern_tabular_detectors_reject_1d_input() -> None:
    with pytest.raises(DataValidationError, match="2-D"):
        ECODDetector().fit(np.zeros(5))


@pytest.mark.parametrize(
    "detector_cls",
    [
        ECODDetector,
        RandomFeatureIsolationForestDetector,
        RandomNetworkDistillationDetector,
    ],
)
def test_modern_tabular_detectors_declare_provenance(detector_cls) -> None:
    assert detector_cls.method_status in {"native", "adapter"}
    assert detector_cls.dependency_extra == "base"
    assert detector_cls.implementation_provenance
    assert set(detector_cls.preset_configs) == {"smoke", "balanced", "research"}


def test_ecod_declares_upstream_provider() -> None:
    assert ECODDetector.upstream_provider == "pyod"
    assert ECODDetector.upstream_module == "pyod.models.ecod.ECOD"
