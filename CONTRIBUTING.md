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

## Branch Naming

Use descriptive branch names with one of these prefixes:

- `feature/` for new functionality.
- `fix/` for bug fixes.
- `docs/` for documentation-only changes.
- `ci/` for workflow and automation changes.
- `release/` for release preparation.

Avoid tool- or author-specific prefixes in shared repository branches.

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
## Releasing

`develop` is the integration branch. `main` points at the last released commit
and is never a place where work is authored.

Promote by **fast-forward**, never through a pull request:

```bash
git push origin develop:main
```

or run the **Promote Develop To Main** workflow, which performs the same
fast-forward and refuses to run if one is not possible.

Then run the **Manual Release** workflow with the version and `main` as the
target. It validates that the version agrees across `pyproject.toml`,
`.zenodo.json`, and `CITATION.cff`, and that `CHANGELOG.md` has a dated section
for that version, before tagging and publishing.

### Why promotion must be a fast-forward

Both branches require a linear history, so a pull request into `main` can only
be squashed or rebased, and each of those writes a **new commit onto `main`**
with a new SHA. `develop` never receives it, so the branches diverge
immediately and the next promotion conflicts on the files every release
touches: the changelog and the citation metadata.

A fast-forward creates no commit, so the two branches stay identical and the
next promotion is a fast-forward again.

If `main` ever does end up ahead, merge it back into `develop` and **keep the
merge commit**. Squashing that merge discards its second parent, so the
histories stay diverged and the conflict returns.
