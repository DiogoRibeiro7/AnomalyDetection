"""Anomaly detection algorithms, benchmark datasets, and comparison tooling.

Subpackages live under this namespace so that installing the distribution does
not place generic names such as ``analytics`` or ``benchmarks`` on the import
path, where they would shadow unrelated modules.
"""

from __future__ import annotations

__all__ = ["analytics", "benchmarks", "cli"]
