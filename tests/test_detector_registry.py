from __future__ import annotations

import pytest

from analytics.detectors import registry


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
    with pytest.raises(ValueError):
        registry.register_detector("dummy", path)


def test_register_rejects_duplicate_without_override() -> None:
    registry.register_detector("dummy", __name__ + ":DummyDetector")
    with pytest.raises(ValueError):
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
    with pytest.raises(KeyError):
        registry.get_detector_class("missing")


def test_get_detector_class_raises_for_missing_target_class() -> None:
    registry.register_detector("dummy", __name__ + ":DoesNotExist")
    with pytest.raises(ImportError):
        registry.get_detector_class("dummy")
