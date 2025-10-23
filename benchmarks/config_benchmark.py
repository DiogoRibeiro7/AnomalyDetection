"""Utilities for running benchmarks from a YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from cli import run_benchmarks


def run_from_config(
    path: str | Path,
    leaderboard: str | None = None,
    n_jobs: int | None = None,
) -> None:
    """Execute benchmarks defined in a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file containing ``datasets`` and ``detectors`` lists.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    datasets: Iterable[str] | None = config.get("datasets")
    detectors: Iterable[str] | None = config.get("detectors")
    configured_jobs = config.get("n_jobs")
    effective_jobs = n_jobs if n_jobs is not None else configured_jobs
    run_benchmarks(
        datasets,
        detectors,
        leaderboard=leaderboard,
        n_jobs=effective_jobs,
    )
