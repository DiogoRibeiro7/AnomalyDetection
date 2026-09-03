from __future__ import annotations

import pytest
from dataexcept import ConfigurationError

from analytics.base import BaseDetector
from analytics.detectors import registry
from analytics.exceptions import UnknownDetectorError


class DummyDetector:
    pass


class ReplacementDetector:
    pass


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "DETECTOR_REGISTRY", {})


def test_register_and_resolve_detector_class() -> None:
    registry.register_detector("dummy", __name__ + ":DummyDetector")
    resolved = registry.get_detector_class("dummy")
    assert resolved is DummyDetector


@pytest.mark.parametrize(
    "path",
    [
        "",
        "module_only",
        ":ClassOnly",
        "module:",
    ],
)
def test_register_rejects_invalid_path_format(path: str) -> None:
    with pytest.raises(ConfigurationError):
        registry.register_detector("dummy", path)


def test_register_rejects_duplicate_without_override() -> None:
    registry.register_detector("dummy", __name__ + ":DummyDetector")
    with pytest.raises(ConfigurationError):
        registry.register_detector("dummy", __name__ + ":ReplacementDetector")


def test_register_allows_duplicate_when_override_enabled() -> None:
    registry.register_detector("dummy", __name__ + ":DummyDetector")
    registry.register_detector(
        "dummy",
        __name__ + ":ReplacementDetector",
        allow_override=True,
    )
    assert registry.get_detector_class("dummy") is ReplacementDetector


def test_get_detector_class_raises_for_unknown_detector() -> None:
    with pytest.raises(UnknownDetectorError):
        registry.get_detector_class("missing")


def test_get_detector_class_raises_for_missing_target_class() -> None:
    registry.register_detector("dummy", __name__ + ":DoesNotExist")
    with pytest.raises(ConfigurationError):
        registry.get_detector_class("dummy")


def test_registered_detectors_declare_supported_score_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()
    supported = {
        "higher_is_more_anomalous",
        "lower_is_more_anomalous",
        "binary_anomaly",
        "estimator_defined",
    }

    for detector_name in registry.DETECTOR_REGISTRY:
        detector_cls = registry.get_detector_class(detector_name)
        assert issubclass(detector_cls, BaseDetector)
        assert detector_cls.score_orientation in supported


def test_legacy_detector_module_reexports_every_submodule() -> None:
    """``analytics.detector`` must re-export all detector submodules.

    The aggregator listed only five submodules, so the modern tabular
    detectors and InductiveDBSCANDetector were unreachable through the
    backwards-compatible import path it documents.
    """

    import importlib

    from analytics import detector as legacy

    exported = set(legacy.__all__)
    for submodule in (
        "classical",
        "correctness",
        "deep",
        "forecasting",
        "graph",
        "modern_tabular",
        "streaming",
    ):
        module = importlib.import_module(f"analytics.detectors.{submodule}")
        missing = sorted(set(module.__all__) - exported)
        assert not missing, f"{submodule} not re-exported: {missing}"
        for name in module.__all__:
            assert getattr(legacy, name) is getattr(module, name)
