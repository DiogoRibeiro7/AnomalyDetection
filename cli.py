#!/usr/bin/env python3
"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""
import argparse
from collections import Counter
from importlib import import_module

import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

from benchmarks.load_all_datasets import load_all_datasets
from analytics.detectors import (
    DETECTOR_REGISTRY,
    get_detector_class,
)


def summarize_datasets(datasets=None):
    """Print basic information about the available datasets."""
    for ds in load_all_datasets(datasets):
        name = ds["name"]
        if datasets and name not in datasets:
            continue
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        print(f"Dataset: {name}")
        if isinstance(df, pd.DataFrame):
            counts = df[labels].value_counts().to_dict()
            print(f"  samples: {len(df)}")
            print(f"  features: {len(features)}")
            print(f"  label distribution: {counts}")
        else:
            counts = Counter(nx.get_node_attributes(df, labels).values())
            print(f"  nodes: {df.number_of_nodes()}")
            print(f"  edges: {df.number_of_edges()}")
            print(f"  label distribution: {dict(counts)}")
        print()


# Instantiate detectors lazily from the registry
DETECTORS = {name: get_detector_class(name)() for name in DETECTOR_REGISTRY}


ALLOWED_PLUGIN_PREFIX = "plugins."


def load_plugins(modules):
    """Import plugin modules while restricting the allowed namespace."""
    for mod in modules:
        if not mod.startswith(ALLOWED_PLUGIN_PREFIX):
            raise ValueError(
                f"Plugin '{mod}' is not allowed; must start with '{ALLOWED_PLUGIN_PREFIX}'"
            )
        import_module(mod)


def run_benchmarks(datasets=None, detectors=None, leaderboard=None):
    """Run benchmarks for the specified datasets and detectors.

    Parameters
    ----------
    datasets: list[str] | None
        Dataset names to evaluate. ``None`` evaluates all available datasets.
    detectors: list[str] | None
        Detector names to evaluate. ``None`` evaluates all registered detectors.
    leaderboard: str | None
        Optional path to a CSV file where results are appended.
    """
    if detectors:
        selected = [(n, DETECTORS[n]) for n in detectors if n in DETECTORS]
    else:
        selected = DETECTORS.items()

    for ds in load_all_datasets(datasets):
        name = ds["name"]
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        print(f"Dataset: {name}")
        for det_name, detector in selected:
            try:
                if isinstance(df, pd.DataFrame):
                    scores = detector.detect_anomalies(df[features])
                    y_true = df[labels]
                else:
                    scores = detector.detect_anomalies(df)
                    y_true = [data[labels] for _, data in df.nodes(data=True)]
                auc = roc_auc_score(y_true, scores)
                print(f"  {det_name}: AUC={auc:.3f}")
                if leaderboard:
                    import csv

                    with open(leaderboard, "a", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh)
                        writer.writerow([name, det_name, auc])
            except Exception as e:  # pragma: no cover - benchmarking utility
                print(f"  {det_name}: skipped ({e})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection benchmarks")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Names of datasets to benchmark. Defaults to all",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show dataset summaries instead of running benchmarks",
    )
    parser.add_argument(
        "--detectors",
        nargs="*",
        help="Detector names to run. Defaults to all",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration specifying datasets and detectors",
    )
    parser.add_argument(
        "--plugins",
        nargs="*",
        help="Plugin modules providing additional detectors",
    )
    parser.add_argument(
        "--leaderboard",
        help="CSV file path to append benchmark results",
    )
    args = parser.parse_args()
    if args.plugins:
        load_plugins(args.plugins)
    if args.config:
        from benchmarks.config_benchmark import run_from_config

        run_from_config(args.config, leaderboard=args.leaderboard)
    elif args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(args.datasets, args.detectors, leaderboard=args.leaderboard)


if __name__ == "__main__":
    main()
