"""Dataset loading helpers used throughout the benchmarking utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import List

import networkx as nx
import pandas as pd

from benchmarks.catalog import (
    DatasetSpec,
    get_dataset_functions,
    get_dataset_metadata,
)

logger = logging.getLogger(__name__)


def _canonicalize_anomaly_label(
    data: object,
    *,
    label_col: str,
    source_anomaly_label: object,
) -> object:
    """Return data whose anomaly class is encoded canonically as ``1``.

    Dataset loaders retain their historical/raw label semantics. The benchmark
    boundary normalizes those labels so every metric sees the same contract:
    ``1 = anomaly`` and ``0 = inlier``.
    """

    if isinstance(data, pd.DataFrame):
        if label_col not in data.columns:
            raise KeyError(f"Dataset label column '{label_col}' is missing.")
        labels = set(data[label_col].dropna().unique())
        if not labels:
            raise ValueError("Dataset label column must contain at least one value.")
        if source_anomaly_label not in labels:
            raise ValueError(
                f"Declared source anomaly label {source_anomaly_label!r} is not "
                f"present in dataset labels {sorted(labels, key=str)!r}."
            )
        canonical = data.copy()
        canonical[label_col] = (canonical[label_col] == source_anomaly_label).astype(int)
        return canonical

    if isinstance(data, nx.Graph):
        labels = nx.get_node_attributes(data, label_col)
        if not labels:
            raise ValueError(
                f"Graph dataset must expose node attribute '{label_col}' for labels."
            )
        observed = set(labels.values())
        if source_anomaly_label not in observed:
            raise ValueError(
                f"Declared source anomaly label {source_anomaly_label!r} is not "
                f"present in graph labels {sorted(observed, key=str)!r}."
            )
        canonical = data.copy()
        nx.set_node_attributes(
            canonical,
            {node: int(value == source_anomaly_label) for node, value in labels.items()},
            label_col,
        )
        return canonical

    raise TypeError(
        "Benchmark datasets must be pandas DataFrames or NetworkX graphs before "
        "label canonicalization."
    )


def load_all_datasets(names: Sequence[str] | None = None) -> List[DatasetSpec]:
    """Load benchmark datasets by canonical name.

    Raw loaders may use historical label conventions. Dataset metadata declares
    ``source_anomaly_label`` and this function converts every loaded benchmark
    to the public evaluation contract ``1 = anomaly`` before returning it.

    Parameters
    ----------
    names:
        Optional sequence of canonical dataset identifiers (for example,
        ``"iris"``). ``None`` loads every available dataset.
    """

    functions = get_dataset_functions()
    if names is None:
        selected = list(functions.items())
    else:
        missing = [name for name in names if name not in functions]
        if missing:
            available = ", ".join(sorted(functions))
            missing_str = ", ".join(missing)
            logger.error(
                "Unknown dataset selector(s): %s. Available datasets: %s",
                missing_str,
                available,
            )
            raise KeyError(
                "Unknown dataset selector(s): "
                f"{missing_str}. Available datasets: {available}"
            )
        selected = [(name, functions[name]) for name in names]

    datasets: List[DatasetSpec] = []
    for key, func in selected:
        data, feature_cols, label_col, display_name = func()
        metadata = get_dataset_metadata(key)
        if "source_anomaly_label" not in metadata:
            raise ValueError(
                f"Dataset '{key}' must declare source_anomaly_label in datasets.yml."
            )
        source_anomaly_label = metadata["source_anomaly_label"]
        canonical_data = _canonicalize_anomaly_label(
            data,
            label_col=label_col,
            source_anomaly_label=source_anomaly_label,
        )
        datasets.append(
            {
                "dataframe": canonical_data,
                "feature_cols": feature_cols,
                "label_col": label_col,
                "name": display_name,
                "key": key,
                "metadata": metadata,
            }
        )

    return datasets
