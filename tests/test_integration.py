"""Integration tests for the CLI benchmarking workflow."""

from __future__ import annotations

import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli


def _mock_dataset() -> dict[str, object]:
    """Return a minimal dataset dictionary matching the loader contract."""
    frame = pd.DataFrame({"feature": [0.1, 0.9], "label": [0, 1]})
    return {
        "name": "mock_dataset",
        "key": "mock_dataset_key",
        "dataframe": frame,
        "feature_cols": ["feature"],
        "label_col": "label",
    }


def _patch_dataset_loader(
    monkeypatch: pytest.MonkeyPatch, dataset: dict[str, object]
) -> None:
    """Patch :func:`cli.load_all_datasets` to return *dataset*."""

    def _fake_loader(selected: list[str] | None = None):
        if selected and dataset["name"] not in selected:
            return []
        return [dataset]

    def _fake_resolver(selectors):
        if selectors is None:
            return [dataset["name"]]
        if isinstance(selectors, dict):
            include = selectors.get("include")
            if include is None:
                return [dataset["name"]]
            return _fake_resolver(include)
        if isinstance(selectors, (list, tuple)):
            return [dataset["name"] for entry in selectors if entry == dataset["name"]]
        if selectors == dataset["name"]:
            return [dataset["name"]]
        return []

    monkeypatch.setattr(cli, "load_all_datasets", _fake_loader)
    monkeypatch.setattr(cli, "resolve_dataset_names", _fake_resolver)


def _install_stub_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a lightweight detector implementation for testing."""

    class StubDetector:
        """Detector that returns the first feature as the anomaly score."""

        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            return values.iloc[:, 0].to_numpy(dtype=float)

    registry = {"stub": "tests.test_integration:StubDetector"}
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", registry)

    def _get_detector(name: str) -> type[StubDetector]:
        assert name == "stub"
        return StubDetector

    monkeypatch.setattr(cli, "get_detector_class", _get_detector)


def _install_param_detector(
    monkeypatch: pytest.MonkeyPatch, scales: list[float]
) -> None:
    """Register a detector that records the initialization scale."""

    class ParamDetector:
        def __init__(self, scale: float = 1.0):
            self.scale = scale
            scales.append(scale)

        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            return values.iloc[:, 0].to_numpy(dtype=float) * self.scale

    registry = {"param_stub": "tests.test_integration:ParamDetector"}
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", registry)

    def _get_detector(name: str) -> type[ParamDetector]:
        assert name == "param_stub"
        return ParamDetector

    monkeypatch.setattr(cli, "get_detector_class", _get_detector)


def test_cli_runs_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI should execute the benchmark pipeline when driven by a YAML config."""

    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)
    _install_stub_detector(monkeypatch)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "n_jobs: 2\n" "datasets:\n  - mock_dataset\n" "detectors:\n  - stub\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(config_path)])
    cli.main()

    output = capsys.readouterr().out
    assert "Dataset: mock_dataset" in output
    assert "stub: AUC=1.000" in output


def test_cli_appends_leaderboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the CLI with --leaderboard should append benchmark results."""

    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)
    _install_stub_detector(monkeypatch)

    leaderboard_path = tmp_path / "leaderboard.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "mock_dataset",
            "--detectors",
            "stub",
            "--leaderboard",
            str(leaderboard_path),
        ],
    )
    cli.main()

    with leaderboard_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_name"] == "mock_dataset"
    assert row["dataset_key"] == "mock_dataset_key"
    assert row["detector_name"] == "stub"
    assert row["detector_label"] == "stub"
    assert row["auc"] == "1.0"
    assert row["error"] == ""


def test_cli_parallel_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI should distribute detector evaluation across worker threads."""

    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)

    class DetectorA:
        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            return values.iloc[:, 0].to_numpy(dtype=float)

    class DetectorB:
        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            return (1.0 - values.iloc[:, 0]).to_numpy(dtype=float)

    registry = {
        "parallel_a": "tests.test_integration:DetectorA",
        "parallel_b": "tests.test_integration:DetectorB",
    }
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", registry)

    def _get_detector(name: str):
        mapping = {"parallel_a": DetectorA, "parallel_b": DetectorB}
        return mapping[name]

    monkeypatch.setattr(cli, "get_detector_class", _get_detector)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            dataset["name"],
            "--detectors",
            "parallel_a",
            "parallel_b",
            "--n-jobs",
            "2",
        ],
    )
    cli.main()

    output = capsys.readouterr().out
    assert "parallel_a: AUC=1.000" in output
    assert "parallel_b" in output


def test_leaderboard_header_written_once_across_multiple_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)
    _install_stub_detector(monkeypatch)

    leaderboard_path = tmp_path / "leaderboard.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "mock_dataset",
            "--detectors",
            "stub",
            "--leaderboard",
            str(leaderboard_path),
        ],
    )
    cli.main()
    cli.main()

    lines = leaderboard_path.read_text(encoding="utf-8").strip().splitlines()
    header_count = sum(1 for line in lines if line.startswith("run_timestamp_utc,"))
    assert header_count == 1
    assert len(lines) == 3  # one header + two result rows


def test_cli_registers_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plugins should integrate with the CLI when provided through --plugins."""

    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)

    shared_registry: dict[str, str] = {}
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", shared_registry)
    monkeypatch.setattr("analytics.detectors.DETECTOR_REGISTRY", shared_registry)
    monkeypatch.setattr(
        "analytics.detectors.registry.DETECTOR_REGISTRY", shared_registry
    )

    plugin_pkg = tmp_path / "plugins"
    plugin_pkg.mkdir()
    (plugin_pkg / "__init__.py").write_text("", encoding="utf-8")
    plugin_module = plugin_pkg / "integration_plugin.py"
    plugin_module.write_text(
        "from analytics.detectors import register_detector\n"
        "\n"
        "class PluginDetector:\n"
        "    def detect_anomalies(self, values):\n"
        "        return values.iloc[:, 0].to_numpy(dtype=float)\n"
        "\n"
        'register_detector("plugin_stub", __name__ + ":PluginDetector")\n',
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["cli.py", "--plugins", "plugins.integration_plugin"]
    )
    cli.main()

    output = capsys.readouterr().out
    assert "plugin_stub" in shared_registry
    assert "plugin_stub" in output
    assert "AUC=1.000" in output


def test_leaderboard_persists_detector_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)

    class FailingDetector:
        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            raise RuntimeError("intentional failure")

    registry = {"failing_stub": "tests.test_integration:FailingDetector"}
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", registry)
    monkeypatch.setattr(cli, "get_detector_class", lambda name: FailingDetector)

    leaderboard_path = tmp_path / "leaderboard.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "mock_dataset",
            "--detectors",
            "failing_stub",
            "--leaderboard",
            str(leaderboard_path),
        ],
    )
    cli.main()

    with leaderboard_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["detector_name"] == "failing_stub"
    assert row["auc"] == ""
    assert "intentional failure" in row["error"]


def test_config_supports_defaults_and_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Config-driven runs should support detector defaults, labels, and plugins."""

    dataset = _mock_dataset()
    _patch_dataset_loader(monkeypatch, dataset)

    recorded_scales: list[float] = []
    _install_param_detector(monkeypatch, recorded_scales)

    config_path = tmp_path / "config.yaml"
    leaderboard_path = tmp_path / "leaderboard.csv"
    config_path.write_text(
        "leaderboard: " + str(leaderboard_path) + "\n"
        "plugins: []\n"
        "datasets:\n"
        "  include:\n"
        "    - mock_dataset\n"
        "detectors:\n"
        "  defaults:\n"
        "    params:\n"
        "      scale: 2.0\n"
        "  include:\n"
        "    - name: param_stub\n"
        "      label: scaled_stub\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", str(config_path)])
    cli.main()

    output = capsys.readouterr().out
    assert "scaled_stub (scale=2.0)" in output
    assert recorded_scales == [2.0]

    with leaderboard_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_name"] == "mock_dataset"
    assert row["detector_label"] == "scaled_stub"
    assert row["detector_params"] == '{"scale": 2.0}'
    assert row["auc"] == "1.0"
