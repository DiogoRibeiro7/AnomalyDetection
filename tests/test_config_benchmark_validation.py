"""Validation tests for YAML-driven benchmark execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks import config_benchmark


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(content, encoding="utf-8")
    return path


def test_invalid_n_jobs_type_fails_fast(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "n_jobs: two\n")
    with pytest.raises(config_benchmark.ConfigValidationError, match="n_jobs"):
        config_benchmark.run_from_config(path)


def test_invalid_plugins_type_fails_fast(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "plugins: plugins.my_module\n")
    with pytest.raises(config_benchmark.ConfigValidationError, match="plugins"):
        config_benchmark.run_from_config(path)


def test_invalid_detector_defaults_params_type_fails_fast(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "detectors:\n"
        "  defaults:\n"
        "    params: 7\n",
    )
    with pytest.raises(
        config_benchmark.ConfigValidationError,
        match=r"detectors\.defaults\.params",
    ):
        config_benchmark.run_from_config(path)


def test_unknown_top_level_key_fails_fast(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "unexpected: true\n")
    with pytest.raises(
        config_benchmark.ConfigValidationError,
        match="unknown key",
    ):
        config_benchmark.run_from_config(path)


def test_valid_config_calls_plugins_and_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(
        tmp_path,
        "datasets:\n"
        "  - iris\n"
        "detectors:\n"
        "  include:\n"
        "    - name: isolation_forest\n"
        "      params:\n"
        "        n_estimators: 100\n"
        "plugins:\n"
        "  - plugins.example\n"
        "leaderboard: results.csv\n"
        "n_jobs: 2\n",
    )

    captured: dict[str, object] = {}

    def _fake_load_plugins(modules):
        captured["plugins"] = list(modules)

    def _fake_run_benchmarks(datasets, detectors, leaderboard=None, n_jobs=None):
        captured["datasets"] = datasets
        captured["detectors"] = detectors
        captured["leaderboard"] = leaderboard
        captured["n_jobs"] = n_jobs

    monkeypatch.setattr(config_benchmark, "_load_plugins", _fake_load_plugins)
    monkeypatch.setattr(config_benchmark, "_run_benchmarks", _fake_run_benchmarks)

    config_benchmark.run_from_config(path)

    assert captured["plugins"] == ["plugins.example"]
    assert captured["datasets"] == ["iris"]
    assert captured["leaderboard"] == "results.csv"
    assert captured["n_jobs"] == 2
