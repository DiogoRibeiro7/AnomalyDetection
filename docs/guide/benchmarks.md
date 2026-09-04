# Running benchmarks

## The CLI

```bash
benchmark-cli                                     # every dataset, every detector
benchmark-cli iris digits                         # specific datasets
benchmark-cli iris --detectors isolation_forest knn
benchmark-cli --summary                           # describe datasets, run nothing
```

`--summary` prints sample counts, feature counts, label distribution, and the
catalog metadata (source, task, modality, label type) without fitting anything.

### Options

| Option | Effect |
| --- | --- |
| `--config PATH` | YAML configuration (see below) |
| `--detectors NAME ...` | restrict the detector set |
| `--plugins MODULE ...` | import plugin modules before running |
| `--metrics NAME ...` | metrics to compute; default `roc_auc` |
| `--metric-k K` | top-k for `precision_at_k` / `recall_at_k` |
| `--metric-threshold T` | threshold for `f1_at_threshold` |
| `--positive-label V` | label value treated as the anomaly class |
| `--random-seed N` | seed detectors and the Python/NumPy RNGs |
| `--n-jobs N` | worker threads for detector execution |
| `--run-id ID` | stable identifier recorded in every artifact |
| `--output-dir DIR` | where the manifest and default report are written |
| `--json-report PATH` | versioned JSON report |
| `--leaderboard PATH` | CSV to append results to |

## Configuration files

Everything the flags express can be written down instead, which is what makes a
run repeatable:

```yaml
run_id: v0.4.0-metrics-smoke
random_seed: 42
output_dir: benchmark-results
json_report: benchmark-results/report.json
leaderboard: benchmark-results/leaderboard.csv
n_jobs: 1

datasets:
  modality: tabular
  task: classification
  limit: 1

metrics:
  include: [roc_auc, average_precision, precision_at_k, best_f1, runtime]
  positive_label: 1
  k: 10

detectors:
  - name: isolation_forest
    params:
      n_estimators: 16
      contamination: 0.1
```

`datasets` accepts either an explicit list of names or a selector — `modality`,
`task`, `limit` — resolved against the catalog. Detectors are either bare names
or `{name, params}` mappings. A name the catalog or registry does not hold
raises `UnknownDatasetError` / `UnknownDetectorError`, and the message lists
what *is* available rather than making you go looking.

```bash
benchmark-cli --config benchmark_config.yml
```

## Run manifests

Every run writes a manifest recording what produced the numbers:

- `schema_version`, `run_id`, `run_timestamp_utc`
- `package_version`, `python_version`, `platform`, `executable`
- `dataset_keys` and per-dataset `dataset_integrity` checksums
- resolved `detectors` with their parameters
- `random_seed`, `n_jobs`, `config_hash`, `metrics`

The `config_hash` is derived from the resolved configuration, so two runs that
share a hash ran the same benchmark. Publish the manifest alongside any result
you report — the version DOI pins the code, the manifest pins the run.

## Leaderboards

`--leaderboard results.csv` appends one row per detector-dataset pair, with
metrics, runtime, orientation, seed, and the run identifiers. Rows accumulate
across runs, so a leaderboard is a history rather than a snapshot; the run id
and config hash are what let you separate one run's rows from another's.

A detector that fails does not abort the run — it records a `failure_category`
and `error`, and the remaining detectors continue.
