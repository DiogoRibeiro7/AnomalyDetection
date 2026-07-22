#!/usr/bin/env python3
"""Command line interface to run anomaly detection benchmarks.

Use ``--summary`` to display information about the available datasets instead
of running the detectors.
"""
import argparse
import csv
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path

from analytics.runtime import ensure_supported_python

ensure_supported_python()

import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

from benchmarks.catalog import resolve_dataset_names
from benchmarks.load_all_datasets import load_all_datasets
from benchmarks.reproducibility import (
    apply_seed_to_detector_entries,
    benchmark_config_hash,
    build_manifest,
    build_report,
    collect_dataset_integrity,
    normalize_run_id,
    seed_runtime,
    utc_timestamp,
    write_json,
)
from analytics.detectors import (
    DETECTOR_REGISTRY,
    get_detector_class,
)


logger = logging.getLogger(__name__)
LEADERBOARD_HEADER = [
    "run_timestamp_utc",
    "run_id",
    "config_hash",
    "dataset_name",
    "dataset_key",
    "detector_name",
    "detector_label",
    "detector_params",
    "random_seed",
    "runtime_seconds",
    "failure_category",
    "auc",
    "error",
]


def summarize_datasets(datasets=None):
    """Print basic information about the available datasets."""

    resolved = resolve_dataset_names(datasets)
    try:
        dataset_entries = load_all_datasets(resolved)
    except KeyError as exc:  # pragma: no cover - defensive logging
        logger.error("%s", exc)
        raise

    for ds in dataset_entries:
        name = ds["name"]
        df = ds["dataframe"]
        features = ds["feature_cols"]
        labels = ds["label_col"]
        metadata = ds.get("metadata") or {}
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
        tags = metadata.get("tags")
        if tags:
            print(f"  tags: {', '.join(tags)}")
        source = metadata.get("source")
        if source:
            print(f"  source: {source}")
        task = metadata.get("task")
        if task:
            print(f"  task: {task}")
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


def _format_detector_display(entry: dict[str, object]) -> str:
    label = entry.get("label") or entry["name"]
    params = entry.get("params") or {}
    if not params:
        return str(label)
    param_bits = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{label} ({param_bits})"


def _resolve_detector_entries(selection):
    if selection is None:
        return [
            {"name": name, "params": {}, "label": name} for name in DETECTOR_REGISTRY
        ]

    if isinstance(selection, dict) and {"include", "exclude", "defaults"} & set(
        selection
    ):
        defaults = selection.get("defaults", {})
        default_params = {}
        if isinstance(defaults, dict):
            default_params = dict(defaults.get("params", {}))
        include_spec = selection.get("include")
        if include_spec is None:
            include_entries = [
                {"name": name, "params": {}, "label": name}
                for name in DETECTOR_REGISTRY
            ]
        else:
            include_entries = _resolve_detector_entries(include_spec)
        exclude_spec = selection.get("exclude")
        excluded = (
            {entry["name"] for entry in _resolve_detector_entries(exclude_spec)}
            if exclude_spec
            else set()
        )
        resolved: list[dict[str, object]] = []
        for entry in include_entries:
            name = entry["name"]
            if name not in DETECTOR_REGISTRY or name in excluded:
                continue
            params = dict(default_params)
            params.update(entry.get("params") or {})
            resolved.append(
                {
                    "name": name,
                    "params": params,
                    "label": entry.get("label") or entry.get("name", name),
                }
            )
        return resolved

    if isinstance(selection, (list, tuple)):
        resolved: list[dict[str, object]] = []
        for item in selection:
            resolved.extend(_resolve_detector_entries(item))
        return resolved

    if isinstance(selection, str):
        if selection not in DETECTOR_REGISTRY:
            return []
        return [{"name": selection, "params": {}, "label": selection}]

    if isinstance(selection, dict) and "name" in selection:
        name = selection.get("name")
        if name not in DETECTOR_REGISTRY:
            return []
        params = dict(selection.get("params", {}))
        label = selection.get("label") or name
        return [{"name": name, "params": params, "label": label}]

    return []


def _classify_failure(exc: Exception | None) -> str:
    if exc is None:
        return ""
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    if isinstance(exc, ValueError):
        return "invalid_input_or_parameter"
    if isinstance(exc, RuntimeError):
        return "runtime_error"
    return "detector_error"


def _result_rows(
    *,
    timestamp: str,
    run_id: str,
    config_hash: str,
    random_seed: int | None,
    datasets_to_run,
    detector_entries,
    results,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_spec in datasets_to_run:
        dataset_name = dataset_spec["name"]
        dataset_key = dataset_spec.get("key") or dataset_name
        for det_spec in detector_entries:
            label = det_spec.get("label") or det_spec["name"]
            outcome = results.get(dataset_name, {}).get(label)
            if not outcome:
                continue
            params = det_spec.get("params") or {}
            rows.append(
                {
                    "run_timestamp_utc": timestamp,
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "dataset_name": dataset_name,
                    "dataset_key": dataset_key,
                    "detector_name": det_spec["name"],
                    "detector_label": label,
                    "detector_params": params,
                    "random_seed": random_seed,
                    "runtime_seconds": outcome["runtime_seconds"],
                    "failure_category": outcome["failure_category"],
                    "auc": outcome["auc"],
                    "error": outcome["error"] or "",
                }
            )
    return rows


def _append_leaderboard_rows(leaderboard_path, rows):
    leaderboard_parent = os.path.dirname(os.fspath(leaderboard_path))
    if leaderboard_parent:
        os.makedirs(leaderboard_parent, exist_ok=True)
    write_header = (
        not os.path.exists(leaderboard_path) or os.path.getsize(leaderboard_path) == 0
    )

    with open(leaderboard_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_HEADER)
        if write_header:
            writer.writeheader()

        for row in rows:
            csv_row = dict(row)
            csv_row["detector_params"] = json.dumps(
                csv_row["detector_params"], sort_keys=True
            )
            csv_row["random_seed"] = (
                "" if csv_row["random_seed"] is None else csv_row["random_seed"]
            )
            csv_row["auc"] = "" if csv_row["auc"] is None else csv_row["auc"]
            writer.writerow(csv_row)


def run_benchmarks(
    datasets=None,
    detectors=None,
    leaderboard=None,
    n_jobs=None,
    output_dir=None,
    json_report=None,
    run_id=None,
    random_seed=None,
):
    """Run benchmarks for the specified datasets and detectors.

    Parameters
    ----------
    datasets: Any
        Dataset selectors to evaluate. ``None`` evaluates all available datasets.
        Selectors may be dataset names, ``tag:<name>`` expressions, or dictionaries
        with ``include``/``exclude``/``limit`` keys.
    detectors: Any
        Detector selectors to evaluate. ``None`` evaluates all registered
        detectors. Detector selectors may be detector names, dictionaries with a
        ``name``/``params`` pair, or configuration dictionaries containing
        ``include``/``exclude``/``defaults``.
    leaderboard: str | None
        Optional path to a CSV file where results are appended.
    n_jobs: int | None
        Number of worker threads to use. ``None`` or ``1`` runs sequentially.
        Non-positive values use the available CPU count.
    output_dir: str | None
        Optional directory where manifest and default report JSON files are
        written.
    json_report: str | None
        Optional explicit path for the JSON benchmark report.
    run_id: str | None
        Optional deterministic run identifier.
    random_seed: int | None
        Optional random seed applied to supported detectors and common RNGs.
    """
    detector_entries = _resolve_detector_entries(detectors)
    if not detector_entries:
        detector_entries = [
            {"name": name, "params": {}, "label": name} for name in DETECTOR_REGISTRY
        ]
    detector_entries = apply_seed_to_detector_entries(detector_entries, random_seed)

    resolved_datasets = resolve_dataset_names(datasets)
    dataset_integrity = collect_dataset_integrity(resolved_datasets)
    try:
        datasets_to_run = list(load_all_datasets(resolved_datasets))
    except KeyError as exc:
        logger.error("%s", exc)
        raise

    if not datasets_to_run:
        return

    seed_runtime(random_seed)
    dataset_keys = [str(ds.get("key") or ds["name"]) for ds in datasets_to_run]
    config_hash = benchmark_config_hash(
        dataset_keys,
        detector_entries,
        random_seed,
        n_jobs,
    )
    timestamp = utc_timestamp()
    effective_run_id = normalize_run_id(run_id, timestamp, config_hash)

    total_tasks = len(datasets_to_run) * len(detector_entries)
    worker_count = _resolve_worker_count(total_tasks, n_jobs)

    results = {ds["name"]: {} for ds in datasets_to_run}

    def _evaluate(dataset_spec, det_spec):
        name = dataset_spec["name"]
        df = dataset_spec["dataframe"]
        features = dataset_spec["feature_cols"]
        labels = dataset_spec["label_col"]
        detector_cls = get_detector_class(det_spec["name"])
        params = dict(det_spec.get("params") or {})
        fit_params: dict[str, object] = {}
        try:
            detector = detector_cls(**params)
        except TypeError:
            detector = detector_cls()
            fit_params = params
        started = time.perf_counter()
        try:
            if isinstance(df, pd.DataFrame):
                scores = detector.detect_anomalies(df[features], **fit_params)
                y_true = df[labels]
            else:
                scores = detector.detect_anomalies(df, **fit_params)
                y_true = [data[labels] for _, data in df.nodes(data=True)]
            auc = float(roc_auc_score(y_true, scores))
            runtime_seconds = round(time.perf_counter() - started, 6)
            return name, det_spec, auc, None, "", runtime_seconds
        except Exception as exc:  # pragma: no cover - benchmarking utility
            runtime_seconds = round(time.perf_counter() - started, 6)
            return (
                name,
                det_spec,
                None,
                str(exc),
                _classify_failure(exc),
                runtime_seconds,
            )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_evaluate, dataset_spec, det_spec)
            for dataset_spec in datasets_to_run
            for det_spec in detector_entries
        ]
        for future in as_completed(futures):
            (
                dataset_name,
                det_spec,
                auc,
                error,
                failure_category,
                runtime_seconds,
            ) = future.result()
            key = det_spec.get("label") or det_spec["name"]
            results[dataset_name][key] = {
                "auc": auc,
                "error": error,
                "failure_category": failure_category,
                "runtime_seconds": runtime_seconds,
                "detector": det_spec,
            }

    for dataset_spec in datasets_to_run:
        name = dataset_spec["name"]
        print(f"Dataset: {name}")
        dataset_results = results.get(name, {})
        for det_spec in detector_entries:
            label = det_spec.get("label") or det_spec["name"]
            outcome = dataset_results.get(label)
            if not outcome:
                print(f"  {label}: skipped (not evaluated)")
                continue
            if outcome["error"] is None:
                display = _format_detector_display(det_spec)
                print(f"  {display}: AUC={outcome['auc']:.3f}")
            else:
                print(f"  {label}: skipped ({outcome['error']})")
        print()

    rows = _result_rows(
        timestamp=timestamp,
        run_id=effective_run_id,
        config_hash=config_hash,
        random_seed=random_seed,
        datasets_to_run=datasets_to_run,
        detector_entries=detector_entries,
        results=results,
    )
    manifest = build_manifest(
        run_id=effective_run_id,
        timestamp=timestamp,
        config_hash=config_hash,
        dataset_keys=dataset_keys,
        detector_entries=detector_entries,
        random_seed=random_seed,
        n_jobs=n_jobs,
        output_directory=str(output_dir) if output_dir else None,
        dataset_integrity=dataset_integrity,
    )
    report = build_report(manifest, rows)

    if leaderboard:
        _append_leaderboard_rows(leaderboard, rows)

    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        write_json(output_path / f"{effective_run_id}-manifest.json", manifest)
    if json_report:
        write_json(json_report, report)
    elif output_path:
        write_json(output_path / f"{effective_run_id}-report.json", report)

    return report


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
        "--output-dir",
        help="Directory for reproducibility manifest and default JSON report files",
    )
    parser.add_argument(
        "--json-report",
        help="Path to write a versioned JSON benchmark report",
    )
    parser.add_argument(
        "--run-id",
        help="Stable identifier to store in manifests, reports, and leaderboards",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Seed supported detectors and common Python/NumPy RNGs",
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
            output_dir=args.output_dir,
            json_report=args.json_report,
            run_id=args.run_id,
            random_seed=args.random_seed,
        )
    elif args.summary:
        summarize_datasets(args.datasets)
    else:
        run_benchmarks(
            args.datasets,
            args.detectors,
            leaderboard=args.leaderboard,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            json_report=args.json_report,
            run_id=args.run_id,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
