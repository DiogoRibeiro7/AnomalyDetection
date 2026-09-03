"""Benchmark failure categories must survive the DataExcept migration.

``_classify_failure`` drives the ``failure_category`` column in reports and the
leaderboard. It used to switch on builtin exception types only, so moving this
package onto DataExcept would silently reclassify every failure it raises as a
generic detector error. Third-party detectors still raise builtins, so both
families have to map to the same categories.
"""

from __future__ import annotations

import pytest
from dataexcept import (
    ConfigurationError,
    DataFormatError,
    DataValidationError,
    DependencyError,
    HyperparameterError,
)

from analytics.exceptions import DetectorNotFittedError
from cli import _classify_failure


def test_no_exception_has_no_category() -> None:
    assert _classify_failure(None) == ""


@pytest.mark.parametrize(
    "exc",
    [
        DependencyError("PyTorch"),
        ImportError("no module named torch"),
        ModuleNotFoundError("no module named torch"),
    ],
)
def test_missing_dependency_covers_both_families(exc: Exception) -> None:
    assert _classify_failure(exc) == "missing_dependency"


@pytest.mark.parametrize(
    "exc",
    [
        DataValidationError("field", 1, "bad"),
        HyperparameterError("k", -1, "bad"),
        ConfigurationError("option", "bad"),
        DataFormatError(["DataFrame"], "list"),
        ValueError("raised by scikit-learn"),
        TypeError("raised by a third-party detector"),
    ],
)
def test_invalid_input_covers_both_families(exc: Exception) -> None:
    assert _classify_failure(exc) == "invalid_input_or_parameter"


@pytest.mark.parametrize(
    "exc",
    [
        DetectorNotFittedError("Isolation Forest"),
        RuntimeError("raised by a third-party detector"),
    ],
)
def test_runtime_errors_cover_both_families(exc: Exception) -> None:
    assert _classify_failure(exc) == "runtime_error"


def test_anything_else_falls_back_to_detector_error() -> None:
    assert _classify_failure(ZeroDivisionError("division by zero")) == "detector_error"
