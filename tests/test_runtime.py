from dataexcept import DependencyError

from analytics.runtime import ensure_supported_python


def test_supported_python_passes() -> None:
    ensure_supported_python((3, 12, 0))


def test_unsupported_python_raises_actionable_error() -> None:
    try:
        ensure_supported_python((3, 13, 0))
    except DependencyError as exc:
        message = str(exc)
        assert "Unsupported Python runtime 3.13" in message
        assert "supports Python 3.12" in message
    else:
        raise AssertionError("Expected DependencyError for unsupported Python")
