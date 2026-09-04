"""Validation tests for the dataset, detector, and metric selectors.

``test_config_benchmark_validation`` covers the top-level scalar keys. The
selector grammars they delegate to were largely unexercised, so each rejection
path is pinned here through the same public entry point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anomalybench.benchmarks import config_benchmark

# (config body, fragment expected in the error message)
INVALID_CONFIGS: list[tuple[str, str]] = [
    # Root
    ("- one\n- two\n", "root"),
    ("datasets: iris\nnot_a_key: 1\n", "unknown key"),
    # Dataset selectors
    ("datasets: 5\n", "datasets"),
    ("datasets:\n  - 5\n", "datasets[0]"),
    ("datasets:\n  limit: three\n", "datasets.limit"),
    ("datasets:\n  modality: 5\n", "datasets.modality"),
    ("datasets:\n  task:\n    - 5\n", "datasets.task"),
    ("datasets:\n  name: 5\n", "datasets.name"),
    ("datasets:\n  tag: 5\n", "datasets.tag"),
    ("datasets:\n  mystery: 5\n", "datasets"),
    ("datasets:\n  include: 5\n", "datasets.include"),
    ("datasets:\n  exclude: 5\n", "datasets.exclude"),
    # Detector selectors
    ("detectors: 5\n", "detectors"),
    ("detectors:\n  - 5\n", "detectors[0]"),
    ("detectors:\n  defaults: 5\n", "detectors.defaults"),
    ("detectors:\n  name: 5\n", "detectors.name"),
    # Metric selectors
    ("metrics:\n  - 5\n", "metrics[0]"),
    ("metrics:\n  mystery: 1\n", "unknown key"),
    ("metrics:\n  k: five\n", "metrics.k"),
    ("metrics:\n  threshold: high\n", "metrics.threshold"),
    ("metrics:\n  positive_label:\n    - 1\n", "metrics.positive_label"),
    # Top-level scalars not already covered
    ("leaderboard: 5\n", "leaderboard"),
    ("output_dir: 5\n", "output_dir"),
    ("json_report: 5\n", "json_report"),
    ("run_id: 5\n", "run_id"),
    ("plugins:\n  - 5\n", "plugins[0]"),
]


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize(("body", "expected"), INVALID_CONFIGS)
def test_invalid_selectors_are_rejected_with_the_offending_path(
    tmp_path: Path, body: str, expected: str
) -> None:
    path = _write_config(tmp_path, body)

    with pytest.raises(config_benchmark.ConfigValidationError) as excinfo:
        config_benchmark.run_from_config(path)

    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "body",
    [
        "datasets:\n  - iris\n  - cardio\n",
        "datasets:\n  include:\n    - iris\n  exclude:\n    - cardio\n  limit: 1\n",
        "datasets:\n  modality: tabular\n  task:\n    - classification\n",
        "datasets:\n  name: iris\n",
        "datasets:\n  tag: tabular\n",
        "metrics: roc_auc\n",
        "metrics:\n  - roc_auc\n  - runtime\n",
        "metrics:\n  include:\n    - roc_auc\n  k: 5\n  threshold: 0.5\n"
        "  positive_label: 1\n",
        "metrics:\n  names:\n    - roc_auc\n  positive_label: anomaly\n",
    ],
)
def test_valid_selectors_pass_validation(body: str) -> None:
    """Accepted grammars must not raise; execution itself is covered elsewhere."""

    import yaml

    config = yaml.safe_load(body)
    config_benchmark._validate_config(config)
