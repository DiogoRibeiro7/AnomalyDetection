#!/usr/bin/env python3
"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""
import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Detector instances are created on demand to avoid importing optional
# dependencies unless required.


ALLOWED_PLUGIN_PREFIX = "plugins."


def load_plugins(modules):
    """Import plugin modules while restricting the allowed namespace."""
    for mod in modules:
        if not mod.startswith(ALLOWED_PLUGIN_PREFIX):
            raise ValueError(
                f"Plugin '{mod}' is not allowed; must start with '{ALLOWED_PLUGIN_PREFIX}'"
            )
        import_module(mod)


def _resolve_worker_count(total_tasks, requested):
    """Return the number of worker threads to use for the benchmark run."""

    if total_tasks <= 0:
        return 1
    if requested is None:
        return 1
    if requested <= 0:
        return min(total_tasks, os.cpu_count() or 1)
    return min(total_tasks, requested)


def run_benchmarks(datasets=None, detectors=None, leaderboard=None, n_jobs=None):
    """Run benchmarks for the specified datasets and detectors.

    Parameters
    ----------
    datasets: list[str] | None
        Dataset names to evaluate. ``None`` evaluates all available datasets.
    detectors: list[str] | None
        Detector names to evaluate. ``None`` evaluates all registered detectors.
    leaderboard: str | None
        Optional path to a CSV file where results are appended.
    n_jobs: int | None
        Number of worker threads to use. ``None`` or ``1`` runs sequentially.
        Non-positive values use the available CPU count.
    """
    if detectors:
        selected = [n for n in detectors if n in DETECTOR_REGISTRY]
    else:
        selected = list(DETECTOR_REGISTRY)

    datasets_to_run = list(load_all_datasets(datasets))

    if not selected:
        for ds in datasets_to_run:
            print(f"Dataset: {ds['name']}")
            print()
        return

    if not datasets_to_run:
        return

    total_tasks = len(datasets_to_run) * len(selected)
    worker_count = _resolve_worker_count(total_tasks, n_jobs)

    results = {ds["name"]: {} for ds in datasets_to_run}

    def _evaluate(dataset_spec, det_name):
        name = dataset_spec["name"]
        df = dataset_spec["dataframe"]
        features = dataset_spec["feature_cols"]
        labels = dataset_spec["label_col"]
        detector = get_detector_class(det_name)()
        try:
            if isinstance(df, pd.DataFrame):
                scores = detector.detect_anomalies(df[features])
                y_true = df[labels]
            else:
                scores = detector.detect_anomalies(df)
                y_true = [data[labels] for _, data in df.nodes(data=True)]
            auc = float(roc_auc_score(y_true, scores))
            return name, det_name, auc, None
        except Exception as exc:  # pragma: no cover - benchmarking utility
            return name, det_name, None, str(exc)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_evaluate, dataset_spec, det_name)
            for dataset_spec in datasets_to_run
            for det_name in selected
        ]
        for future in as_completed(futures):
            dataset_name, det_name, auc, error = future.result()
            results[dataset_name][det_name] = {"auc": auc, "error": error}

    for dataset_spec in datasets_to_run:
        name = dataset_spec["name"]
        print(f"Dataset: {name}")
        dataset_results = results.get(name, {})
        for det_name in selected:
            outcome = dataset_results.get(det_name)
            if not outcome:
                print(f"  {det_name}: skipped (not evaluated)")
                continue
            if outcome["error"] is None:
                print(f"  {det_name}: AUC={outcome['auc']:.3f}")
            else:
                print(f"  {det_name}: skipped ({outcome['error']})")
        print()

    if leaderboard:
        import csv

        with open(leaderboard, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for dataset_spec in datasets_to_run:
                name = dataset_spec["name"]
                for det_name in selected:
                    outcome = results[name].get(det_name)
                    if outcome and outcome["auc"] is not None:
                        writer.writerow([name, det_name, outcome["auc"]])


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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of worker threads for detector execution. "
            "Use a non-positive value to leverage all available CPUs."
        ),
    )
    args = parser.parse_args()
    if args.plugins:
        load_plugins(args.plugins)
    if args.config:
        from benchmarks.config_benchmark import run_from_config

        run_from_config(
            args.config,
            leaderboard=args.leaderboard,
            n_jobs=args.n_jobs,
        )
    elif args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(
            args.datasets,
            args.detectors,
            leaderboard=args.leaderboard,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
