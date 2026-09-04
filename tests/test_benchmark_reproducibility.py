"""Tests for reproducible benchmark report generation."""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cli
from benchmarks import reproducibility
from benchmarks.reproducibility import (
    MANIFEST_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    apply_seed_to_detector_entries,
    benchmark_config_hash,
    collect_dataset_integrity,
    package_version,
)


def test_reproducibility_hash_is_stable() -> None:
    detectors = [
        {"name": "isolation_forest", "label": "iforest", "params": {"seed": 7}}
    ]

    metric_config = {"names": ["roc_auc"], "positive_label": 1}
    first = benchmark_config_hash(
        ["iris"], detectors, random_seed=7, n_jobs=1, metric_config=metric_config
    )
    second = benchmark_config_hash(
        ["iris"], detectors, random_seed=7, n_jobs=1, metric_config=metric_config
    )

    assert first == second
    assert len(first) == 16


def test_package_version_matches_the_source_of_truth() -> None:
    """Read from pyproject rather than pinned, so a version bump is not a failure.

    This assertion used to hardcode the version, which meant every release broke
    it. Releases are automated now, so that would have failed on every one.
    """

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        expected = str(tomllib.load(fh)["project"]["version"])

    assert package_version() == expected


def test_package_version_prefers_source_over_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the function: report the tree's version, not the installed one.

    A checkout whose version has moved ahead of the installed distribution must
    report the checkout, or a benchmark manifest would record the wrong version.
    """

    monkeypatch.setattr(
        reproducibility.metadata,
        "version",
        lambda _name: "9.9.9-installed",
    )

    assert package_version() != "9.9.9-installed"


def test_seed_is_added_only_for_supported_detectors() -> None:
    entries = [
        {"name": "isolation_forest", "params": {}, "label": "iforest"},
        {"name": "random_feature_isolation_forest", "params": {}, "label": "rfif"},
        {"name": "random_network_distillation", "params": {}, "label": "rnd"},
        {"name": "knn", "params": {}, "label": "knn"},
    ]

    seeded = apply_seed_to_detector_entries(entries, seed=42)

    assert seeded[0]["params"] == {"random_state": 42}
    assert seeded[1]["params"] == {"random_state": 42}
    assert seeded[2]["params"] == {"random_state": 42}
    assert seeded[3]["params"] == {}


def test_dataset_integrity_collects_bundled_file_hashes() -> None:
    records = collect_dataset_integrity(["thyroid"])

    assert len(records) == 1
    assert records[0]["dataset_key"] == "thyroid"
    assert records[0]["file"] == "thyroid.csv"
    assert records[0]["size_bytes"] > 0
    assert len(records[0]["sha256"]) == 64


def test_run_benchmarks_writes_manifest_report_and_enriched_leaderboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = {
        "name": "mock_dataset",
        "key": "mock_dataset_key",
        "dataframe": pd.DataFrame({"feature": [0.1, 0.9], "label": [0, 1]}),
        "feature_cols": ["feature"],
        "label_col": "label",
        "metadata": {
            "modality": "tabular",
            "task": "classification",
            "label_type": "binary_class",
            "positive_label": 1,
        },
    }

    class StubDetector:
        def detect_anomalies(self, values: pd.DataFrame) -> np.ndarray:
            return values.iloc[:, 0].to_numpy(dtype=float)

    monkeypatch.setattr(cli, "resolve_dataset_names", lambda datasets: ["mock_key"])
    monkeypatch.setattr(cli, "load_all_datasets", lambda names: [dataset])
    monkeypatch.setattr(cli, "collect_dataset_integrity", lambda names: [])
    monkeypatch.setattr(cli, "DETECTOR_REGISTRY", {"stub": "tests:StubDetector"})
    monkeypatch.setattr(cli, "get_detector_class", lambda name: StubDetector)

    output_dir = tmp_path / "out"
    report_path = output_dir / "report.json"
    leaderboard_path = output_dir / "leaderboard.csv"

    report = cli.run_benchmarks(
        datasets=["mock_key"],
        detectors=["stub"],
        leaderboard=leaderboard_path,
        output_dir=output_dir,
        json_report=report_path,
        run_id="fixture-run",
        random_seed=123,
        n_jobs=1,
        metrics=["roc_auc", "average_precision", "runtime"],
    )

    manifest_path = output_dir / "fixture-run-manifest.json"
    assert manifest_path.exists()
    assert report_path.exists()
    assert report["schema_version"] == REPORT_SCHEMA_VERSION

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["run_id"] == "fixture-run"
    assert manifest["dataset_keys"] == ["mock_dataset_key"]
    assert manifest["dataset_metadata"][0]["modality"] == "tabular"
    assert manifest["dataset_metadata"][0]["label_type"] == "binary_class"
    assert persisted_report["schema_version"] == REPORT_SCHEMA_VERSION
    assert persisted_report["manifest"]["config_hash"] == manifest["config_hash"]

    result_row = persisted_report["results"][0]
    assert result_row["run_id"] == "fixture-run"
    assert result_row["detector_name"] == "stub"
    assert result_row["random_seed"] == 123
    assert result_row["runtime_seconds"] >= 0
    assert result_row["failure_category"] == ""
    # The stub returns a plain array, so evaluation applies the default
    # orientation and the report records the one actually used.
    assert result_row["score_orientation"] == "higher_is_more_anomalous"
    assert result_row["metrics"]["roc_auc"] == 1.0
    assert result_row["metrics"]["average_precision"] == 1.0
    assert result_row["metrics"]["runtime"] >= 0
    assert result_row["auc"] == 1.0
    assert persisted_report["manifest"]["metrics"]["names"] == [
        "roc_auc",
        "average_precision",
        "runtime",
    ]

    with leaderboard_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))

    assert csv_rows[0]["run_id"] == "fixture-run"
    assert csv_rows[0]["config_hash"] == manifest["config_hash"]
    assert csv_rows[0]["random_seed"] == "123"
    assert csv_rows[0]["failure_category"] == ""
    assert csv_rows[0]["score_orientation"] == "higher_is_more_anomalous"
    assert '"average_precision": 1.0' in csv_rows[0]["metrics"]
