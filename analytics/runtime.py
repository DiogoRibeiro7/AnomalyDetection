"""Runtime compatibility guards."""

from __future__ import annotations

import sys
from typing import Sequence


SUPPORTED_PYTHON: tuple[int, int] = (3, 12)


def ensure_supported_python(
    version_info: Sequence[int] | None = None,
) -> None:
    """Raise an actionable error when Python runtime is unsupported."""

    info = version_info if version_info is not None else sys.version_info
    major, minor = int(info[0]), int(info[1])
    if (major, minor) != SUPPORTED_PYTHON:
        expected = f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}"
        current = f"{major}.{minor}"
        raise RuntimeError(
            "Unsupported Python runtime "
            f"{current}. This project currently supports Python {expected}. "
            "Use a Python 3.12 virtual environment."
        )
