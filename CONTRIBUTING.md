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

Releases are automated. Nothing about a version is typed by hand.

[Release Please](https://github.com/googleapis/release-please) reads the
Conventional Commit messages on `main` and keeps a release pull request open
that carries the next version and its changelog. Merging that pull request tags
the release and publishes it; Zenodo archives the published release and mints
the DOI.

So the whole release is: **merge the release pull request.**

The version is derived from the commits, which is why the prefix matters:

| Commit prefix | Effect while pre-1.0 |
| --- | --- |
| `fix:` | patch bump |
| `feat:` | minor bump |
| `feat!:` or a `BREAKING CHANGE:` footer | minor bump, listed under Breaking |
| `chore:`, `ci:`, `docs:`, `test:`, `refactor:`, `style:` | no bump; grouped in the notes |

The release pull request updates the version in `pyproject.toml`,
`.zenodo.json`, and `CITATION.cff` together, and
`tests/test_citation_metadata.py` asserts the three agree, so a missed file
fails CI rather than shipping.

### Manual releases

`Manual Release` remains as a fallback for when the automated path is
unavailable. It validates that the version agrees across the three files and
that `CHANGELOG.md` has a dated section for it before tagging. Prefer the
release pull request.

### Branches

`main` is the only long-lived branch. Work happens on short-lived branches that
merge into it, and releases are cut from it.

There is deliberately no second long-lived branch. The release commit lands on
whichever branch is released from and never reaches the other, so a two-branch
flow diverges on exactly the files every release touches. That cost several
rounds of conflict resolution before v0.5.0.
