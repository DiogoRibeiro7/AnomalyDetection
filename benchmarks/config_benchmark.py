"""Utilities for running benchmarks from a YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cli import load_plugins, run_benchmarks


def run_from_config(
    path: str | Path,
    leaderboard: str | None = None,
    n_jobs: int | None = None,
) -> None:
    """Execute benchmarks defined in a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file containing ``datasets`` and ``detectors`` selectors.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    datasets: Any = config.get("datasets")
    detectors: Any = config.get("detectors")
    configured_jobs = config.get("n_jobs")
    effective_jobs = n_jobs if n_jobs is not None else configured_jobs
    configured_leaderboard = config.get("leaderboard")
    effective_leaderboard = (
        leaderboard if leaderboard is not None else configured_leaderboard
    )

    config_plugins = config.get("plugins")
    if config_plugins:
        load_plugins(config_plugins)

    run_benchmarks(
        datasets,
        detectors,
        leaderboard=effective_leaderboard,
        n_jobs=effective_jobs,
    )
