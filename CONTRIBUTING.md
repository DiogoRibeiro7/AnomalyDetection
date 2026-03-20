# Contributing

Thank you for your interest in improving the anomaly detection library. This guide explains how to add new algorithms and contribute changes.

## Getting Started
1. Fork the repository and clone your fork.
2. Use Python `3.12` (for example `3.12.x`). Python `3.13` is currently unsupported.
3. Install dependencies and pre-commit hooks:
   ```bash
   poetry install
   pre-commit install
   ```
4. Ensure tests run:
   ```bash
   pytest -q
   ```

## Adding a New Detector
1. Create a detector class inheriting from `BaseDetector` within the appropriate submodule under `analytics/detectors/` (e.g., `classical.py`, `deep.py`).
2. Implement the `fit` and `score` methods. Use existing detectors as references.
3. Register the detector in the CLI registry so it can be selected with `--detectors`.
4. Add unit tests covering the new detector’s behaviour.
5. Update the README with a brief description of the detector and any usage notes.

## Expanding Benchmark Datasets
1. Add compact dataset assets under `benchmarks/` (for example CSV files). Prefer small, redistributable excerpts that keep tests fast.
2. Implement a loader in `benchmarks/load_datasets.py` that returns `(dataframe, feature_columns, label_column, display_name)`.
3. Describe the dataset in `benchmarks/datasets.yml`, including tags (e.g., `tabular`, `graph`, `time_series`) and source metadata so others can find the original reference.
4. Cover new loaders with tests—`tests/test_benchmark_catalog.py` contains examples that assert catalog registration and metadata exposure.

## Benchmark Configuration & Community Tooling
- The CLI accepts YAML configurations via `--config`. See `benchmarks/config_benchmark.py` for supported keys. You can:
  - Select datasets by name, by tag (`tag:tabular`), or via `include`/`exclude`/`limit` selectors.
  - Provide detector defaults (`detectors.defaults.params`) and per-detector overrides (`detectors.include`).
  - Assign human-friendly labels (`label`) so leaderboard exports stay readable.
  - Declare plugins under `plugins` to auto-load community detectors.
- Keep configuration-focused tests alongside `tests/test_integration.py` to ensure new capabilities remain stable.
- When proposing new tooling (dataset generators, reporting utilities, dashboards), document the workflow in the README and add usage notes or templates that the community can extend.

## Coding Standards
- Follow PEP 8 style and run `pre-commit run --files <changed-files>` before committing.
- Include type hints for public functions.
- Document new functions and classes.

## Submitting Changes
1. Ensure all tests pass:
   ```bash
   pre-commit run --files <changed-files>
   pytest
   ```
2. Commit your changes with a descriptive message.
3. Open a pull request describing the motivation and implementation details.

We welcome suggestions and improvements to these guidelines—feel free to open an issue to discuss enhancements.
