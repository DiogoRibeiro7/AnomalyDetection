"""Utilities for running benchmarks from a YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when a benchmark YAML configuration is invalid."""


def _fail(path: str, expected: str, value: Any) -> None:
    value_type = type(value).__name__
    raise ConfigValidationError(
        f"Invalid config at '{path}': expected {expected}, got {value_type}."
    )


_load_plugins = None
_run_benchmarks = None


def _validate_dataset_selector(selector: Any, path: str) -> None:
    if selector is None or isinstance(selector, str):
        return
    if isinstance(selector, (list, tuple)):
        for idx, item in enumerate(selector):
            _validate_dataset_selector(item, f"{path}[{idx}]")
        return
    if isinstance(selector, dict):
        keys = set(selector)
        if {"include", "exclude", "limit"} & keys:
            if "include" in selector:
                _validate_dataset_selector(selector.get("include"), f"{path}.include")
            if "exclude" in selector:
                _validate_dataset_selector(selector.get("exclude"), f"{path}.exclude")
            if "limit" in selector:
                limit = selector.get("limit")
                if not isinstance(limit, int):
                    _fail(f"{path}.limit", "an integer", limit)
            return
        if "name" in selector:
            name = selector.get("name")
            if not isinstance(name, str):
                _fail(f"{path}.name", "a string", name)
            return
        if "tag" in selector:
            tag = selector.get("tag")
            if not isinstance(tag, str):
                _fail(f"{path}.tag", "a string", tag)
            return
        _fail(path, "a dataset selector dictionary", selector)
    _fail(path, "a dataset selector", selector)


def _validate_detector_selector(selector: Any, path: str) -> None:
    if selector is None or isinstance(selector, str):
        return
    if isinstance(selector, (list, tuple)):
        for idx, item in enumerate(selector):
            _validate_detector_selector(item, f"{path}[{idx}]")
        return
    if isinstance(selector, dict):
        keys = set(selector)
        if {"include", "exclude", "defaults"} & keys:
            if "include" in selector:
                _validate_detector_selector(selector.get("include"), f"{path}.include")
            if "exclude" in selector:
                _validate_detector_selector(selector.get("exclude"), f"{path}.exclude")
            if "defaults" in selector:
                defaults = selector.get("defaults")
                if not isinstance(defaults, dict):
                    _fail(f"{path}.defaults", "a mapping", defaults)
                params = defaults.get("params")
                if params is not None and not isinstance(params, dict):
                    _fail(f"{path}.defaults.params", "a mapping", params)
            return
        if "name" in selector:
            name = selector.get("name")
            if not isinstance(name, str):
                _fail(f"{path}.name", "a string", name)
            label = selector.get("label")
            if label is not None and not isinstance(label, str):
                _fail(f"{path}.label", "a string", label)
            params = selector.get("params")
            if params is not None and not isinstance(params, dict):
                _fail(f"{path}.params", "a mapping", params)
            return
        _fail(path, "a detector selector dictionary", selector)
    _fail(path, "a detector selector", selector)


def _validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        _fail("root", "a mapping", config)

    allowed_keys = {"datasets", "detectors", "n_jobs", "leaderboard", "plugins"}
    unknown = set(config) - allowed_keys
    if unknown:
        unknown_keys = ", ".join(sorted(unknown))
        raise ConfigValidationError(
            f"Invalid config at 'root': unknown key(s): {unknown_keys}."
        )

    _validate_dataset_selector(config.get("datasets"), "datasets")
    _validate_detector_selector(config.get("detectors"), "detectors")

    n_jobs = config.get("n_jobs")
    if n_jobs is not None and not isinstance(n_jobs, int):
        _fail("n_jobs", "an integer", n_jobs)

    leaderboard = config.get("leaderboard")
    if leaderboard is not None and not isinstance(leaderboard, str):
        _fail("leaderboard", "a string", leaderboard)

    plugins = config.get("plugins")
    if plugins is not None:
        if not isinstance(plugins, (list, tuple)):
            _fail("plugins", "a list of strings", plugins)
        for idx, plugin in enumerate(plugins):
            if not isinstance(plugin, str):
                _fail(f"plugins[{idx}]", "a string", plugin)


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
    _validate_config(config)
    global _load_plugins, _run_benchmarks
    if _load_plugins is None or _run_benchmarks is None:
        from cli import load_plugins as _cli_load_plugins, run_benchmarks as _cli_run_benchmarks

        _load_plugins = _cli_load_plugins
        _run_benchmarks = _cli_run_benchmarks

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
        _load_plugins(config_plugins)

    _run_benchmarks(
        datasets,
        detectors,
        leaderboard=effective_leaderboard,
        n_jobs=effective_jobs,
    )
