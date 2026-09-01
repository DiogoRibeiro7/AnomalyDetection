"""Failure contracts for benchmark label canonicalization.

``test_dataset_label_contract`` covers the bundled datasets, which all satisfy
the contract. The rejections that protect it are exercised here, since a
mislabelled dataset would otherwise invert every metric computed from it.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd
import pytest

from benchmarks.load_all_datasets import _canonicalize_anomaly_label, load_all_datasets


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "Class": [0, 1, 0]})


def test_missing_label_column_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="is missing"):
        _canonicalize_anomaly_label(frame, label_col="absent", source_anomaly_label=1)


def test_empty_label_column_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one value"):
        _canonicalize_anomaly_label(
            frame.iloc[0:0], label_col="Class", source_anomaly_label=1
        )


def test_absent_source_label_is_rejected(frame: pd.DataFrame) -> None:
    """A declared anomaly label that never occurs means the audit is stale."""

    with pytest.raises(ValueError, match="is not present in dataset labels"):
        _canonicalize_anomaly_label(frame, label_col="Class", source_anomaly_label=9)


def test_frame_labels_are_mapped_to_one_for_anomalies(frame: pd.DataFrame) -> None:
    canonical = _canonicalize_anomaly_label(
        frame, label_col="Class", source_anomaly_label=0
    )

    assert isinstance(canonical, pd.DataFrame)
    # The source anomaly label was 0, so those rows become 1.
    assert canonical["Class"].tolist() == [1, 0, 1]
    # The input must not be mutated.
    assert frame["Class"].tolist() == [0, 1, 0]


def test_graph_without_the_label_attribute_is_rejected() -> None:
    graph = nx.Graph()
    graph.add_node(0)

    with pytest.raises(ValueError, match="must expose node attribute"):
        _canonicalize_anomaly_label(graph, label_col="label", source_anomaly_label=1)


def test_graph_with_an_absent_source_label_is_rejected() -> None:
    graph = nx.Graph()
    graph.add_node(0, label=0)

    with pytest.raises(ValueError, match="is not present in graph labels"):
        _canonicalize_anomaly_label(graph, label_col="label", source_anomaly_label=9)


def test_graph_labels_are_mapped_to_one_for_anomalies() -> None:
    graph = nx.Graph()
    graph.add_node(0, label="bad")
    graph.add_node(1, label="good")

    canonical = _canonicalize_anomaly_label(
        graph, label_col="label", source_anomaly_label="bad"
    )

    assert isinstance(canonical, nx.Graph)
    assert nx.get_node_attributes(canonical, "label") == {0: 1, 1: 0}


def test_unsupported_dataset_types_are_rejected() -> None:
    unsupported: Any = [[1, 2], [3, 4]]

    with pytest.raises(TypeError, match="DataFrames or NetworkX graphs"):
        _canonicalize_anomaly_label(
            unsupported, label_col="Class", source_anomaly_label=1
        )


def test_unknown_dataset_names_list_the_available_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        load_all_datasets(["not_a_dataset"])

    message = str(excinfo.value)
    assert "not_a_dataset" in message
    assert "Available datasets" in message
