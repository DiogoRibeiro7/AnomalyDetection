"""Tests for the benchmark dataset anomaly-label contract."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from benchmarks.catalog import (
    get_dataset_functions,
    get_dataset_metadata,
    list_available_datasets,
)
from benchmarks.load_all_datasets import load_all_datasets

EXPECTED_SOURCE_ANOMALY_LABELS: dict[str, int] = {
    "arrhythmia": 0,
    "cardio": 0,
    "digits": 0,
    "fashion_mnist_sample": 0,
    "iris": 0,
    "karate_club_graph": 1,
    "kddcup_sample": 1,
    "lympho": 0,
    "nab_art_daily_small_noise": 0,
    "nab_machine_temperature": 0,
    "synthetic_timeseries": 0,
    "thyroid": 0,
    "wisconsin_breast_cancer": 0,
}


def test_every_dataset_declares_audited_source_anomaly_label() -> None:
    available = set(list_available_datasets())

    assert available == set(EXPECTED_SOURCE_ANOMALY_LABELS)
    for dataset_key, expected_label in EXPECTED_SOURCE_ANOMALY_LABELS.items():
        metadata = get_dataset_metadata(dataset_key)
        assert metadata["source_anomaly_label"] == expected_label
        assert metadata["positive_label"] == 1


@pytest.mark.parametrize("dataset_key", sorted(EXPECTED_SOURCE_ANOMALY_LABELS))
def test_loaded_benchmarks_canonicalize_anomaly_class_to_one(dataset_key: str) -> None:
    dataset = load_all_datasets([dataset_key])[0]
    data = dataset["dataframe"]
    label_col = dataset["label_col"]

    if isinstance(data, pd.DataFrame):
        labels = set(data[label_col].unique())
    elif isinstance(data, nx.Graph):
        labels = set(nx.get_node_attributes(data, label_col).values())
    else:  # pragma: no cover - guarded by loader contract
        raise AssertionError(f"Unexpected benchmark data type: {type(data)!r}")

    assert labels == {0, 1}
    assert dataset["metadata"]["positive_label"] == 1


def test_cardio_raw_and_benchmark_labels_have_distinct_documented_semantics() -> None:
    """Pathological Cardio cases are raw 0 but canonical benchmark anomaly 1."""

    raw_loader = get_dataset_functions()["cardio"]
    raw_df, _, raw_label_col, _ = raw_loader()
    benchmark = load_all_datasets(["cardio"])[0]
    benchmark_df = benchmark["dataframe"]

    assert isinstance(benchmark_df, pd.DataFrame)
    raw_anomalies = (raw_df[raw_label_col] == 0).sum()
    raw_inliers = (raw_df[raw_label_col] == 1).sum()
    benchmark_anomalies = (benchmark_df[raw_label_col] == 1).sum()
    benchmark_inliers = (benchmark_df[raw_label_col] == 0).sum()
    assert raw_anomalies == benchmark_anomalies
    assert raw_inliers == benchmark_inliers
