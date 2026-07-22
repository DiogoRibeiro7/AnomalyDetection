# Contributing

Thank you for improving the anomaly detection library. This guide explains the
expected local workflow and the conventions for adding detectors, datasets, and
benchmark tooling.

## Getting Started

1. Fork the repository and clone your fork.
2. Use Python `3.12`. Python `3.13` is currently unsupported.
3. Install dependencies and pre-commit hooks:

   ```bash
   poetry install
   poetry run pre-commit install
   ```

4. Run the core validation checks:

   ```bash
   poetry check
   poetry build -f wheel
   poetry run python -m pytest -q
   ```

## Adding A New Detector

1. Create a detector class inheriting from `BaseDetector` in the appropriate
   module under `analytics/detectors/`.
2. Implement `fit` and `score`. Use existing detectors as references.
3. Register the detector so it can be selected with `--detectors`.
4. Add focused unit tests that cover expected behavior and important edge cases.
5. Update `README.md` when the detector changes user-facing capabilities,
   dependencies, or usage.

## Expanding Benchmark Datasets

1. Add compact dataset assets under `benchmarks/`. Prefer small,
   redistributable excerpts that keep tests fast.
2. Implement a loader in `benchmarks/load_datasets.py` that returns
   `(dataframe, feature_columns, label_column, display_name)`.
3. Describe the dataset in `benchmarks/datasets.yml`, including tags such as
   `tabular`, `graph`, or `time_series`, plus source metadata.
4. Cover new loaders with tests. `tests/test_benchmark_catalog.py` contains
   examples that assert catalog registration and metadata exposure.

## Benchmark Configuration

The CLI accepts YAML configuration via `--config`. See
`benchmarks/config_benchmark.py` for supported keys. Configuration can:

- Select datasets by name, by tag, or through `include`, `exclude`, and `limit`
  selectors.
- Provide detector defaults with `detectors.defaults.params`.
- Provide per-detector overrides with `detectors.include`.
- Assign readable labels for leaderboard exports.
- Declare plugin modules that register community detectors.

Keep configuration-focused tests alongside `tests/test_integration.py` when
changing benchmark orchestration behavior.

## Coding Standards

- Follow PEP 8 and run pre-commit before submitting changes.
- Include type hints for public functions.
- Document new public functions and classes.
- Keep changes focused on the behavior being added or fixed.

## Submitting Changes

Before opening a pull request, run:

```bash
poetry check
poetry build -f wheel
poetry run pre-commit run --all-files
poetry run python -m pytest -q
```

Open a pull request that describes the motivation, implementation details, and
validation performed. Open an issue first for larger design changes.
