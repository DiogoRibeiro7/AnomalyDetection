"""The package promises every error it raises inherits from DataExceptError.

That claim is made in the README and the CHANGELOG breaking-change note, and it
was already wrong once: the migration missed a ``raise FileNotFoundError`` in
``collect_dataset_integrity`` because the sweep grepped for a fixed list of
builtin names. This walks the source instead, so a raise of any type cannot be
added without either inheriting from DataExceptError or failing here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from dataexcept import DataExceptError

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ["anomalybench", "src"]
SOURCE_FILES: list[str] = []

# Raised deliberately as control flow for the test suite itself, not as a
# failure this package reports to callers.
ALLOWED_NON_DATAEXCEPT = {"AssertionError", "StopIteration", "NotImplementedError"}


def _python_files() -> list[Path]:
    files = [ROOT / name for name in SOURCE_FILES]
    for root in SOURCE_ROOTS:
        files.extend(sorted((ROOT / root).rglob("*.py")))
    return [f for f in files if f.exists()]


def _raised_names(path: Path) -> list[tuple[str, int]]:
    """Return (exception name, line) for every ``raise Name(...)`` statement."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name):
            found.append((exc.id, node.lineno))
        elif isinstance(exc, ast.Attribute):
            found.append((exc.attr, node.lineno))
    return found


def _project_exception_classes() -> list[type[BaseException]]:
    """Every exception class this package defines, wherever it is declared."""

    import anomalybench.analytics.exceptions as project
    from anomalybench.benchmarks.config_benchmark import ConfigValidationError

    classes = [getattr(project, name) for name in project.__all__]
    # Declared outside analytics.exceptions, so not covered by __all__ above.
    classes.append(ConfigValidationError)
    return classes


def _dataexcept_names() -> set[str]:
    import dataexcept

    names = {
        name
        for name in dir(dataexcept)
        if isinstance(getattr(dataexcept, name), type)
        and issubclass(getattr(dataexcept, name), DataExceptError)
    }
    # Derived from the hierarchy rather than from names: an exception that
    # stopped inheriting from DataExceptError drops out of the allowed set, so
    # its raise sites are reported instead of being whitelisted by name.
    names |= {
        cls.__name__
        for cls in _project_exception_classes()
        if issubclass(cls, DataExceptError)
    }
    return names


def test_every_raise_uses_the_dataexcept_hierarchy() -> None:
    allowed = _dataexcept_names() | ALLOWED_NON_DATAEXCEPT
    offenders: list[str] = []

    for path in _python_files():
        for name, line in _raised_names(path):
            if name not in allowed:
                rel = path.relative_to(ROOT).as_posix()
                offenders.append(f"{rel}:{line} raises {name}")

    assert not offenders, "raises outside the DataExcept hierarchy:\n" + "\n".join(
        offenders
    )


def test_project_exceptions_all_inherit_from_dataexcept_error() -> None:
    for cls in _project_exception_classes():
        assert issubclass(cls, DataExceptError), cls.__name__


def test_project_exceptions_do_not_inherit_from_builtins() -> None:
    """The breaking change is that these are no longer builtin subclasses."""

    for cls in _project_exception_classes():
        for builtin in (ValueError, KeyError, TypeError, RuntimeError, ImportError):
            assert not issubclass(
                cls, builtin
            ), f"{cls.__name__} is a {builtin.__name__}"


@pytest.mark.parametrize("root", SOURCE_ROOTS)
def test_the_scan_actually_finds_raises(root: str) -> None:
    """Guard against the walk silently matching nothing."""

    total = sum(len(_raised_names(p)) for p in (ROOT / root).rglob("*.py"))
    assert total > 0
