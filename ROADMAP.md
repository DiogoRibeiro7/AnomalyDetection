# Roadmap

This roadmap translates the project direction into versioned milestones. It is
intended to help contributors choose work that fits the current release stage
and to keep releases scoped enough to validate properly.

The project is currently in the `0.x` series. Minor versions may still adjust
APIs when needed, but each release should preserve documented workflows or
provide a migration note.

## v0.1.x - Foundation And Release Hygiene

Status: released as `v0.1.0`; patch releases only.

The `0.1` line establishes the repository, packaging, release, and citation
baseline.

- Maintain Poetry packaging and the installable `benchmark-cli` command.
- Keep GitHub Actions green for core tests and streaming-extra smoke tests.
- Keep Zenodo and GitHub citation metadata current.
- Patch documentation, packaging, and CI issues found while preparing later
  releases.
- Avoid broad feature work unless it directly unblocks `v0.2.0`.

Exit criteria for `v0.1.x` patches:

- `poetry check` passes.
- `poetry build -f wheel` passes.
- `poetry run python -m pytest -q` passes.
- GitHub CI passes on `main`.

## v0.2.0 - Detector API Stabilization

Goal: make detector implementations predictable enough for users and plugins.

Planned work:

- Define the supported detector lifecycle clearly: `fit`, `score`, and
  `detect_anomalies`.
- Normalize score orientation across detectors, documenting whether larger
  scores always mean more anomalous.
- Add fitted-state checks so `score` fails clearly before `fit`.
- Add common input validation for NumPy arrays, pandas dataframes, graph inputs,
  and time-series matrices.
- Replace duplicated input conversion helpers with shared utilities where it
  reduces maintenance risk.
- Audit detector registry keys and labels for consistency.
- Document optional dependency behavior for deep, streaming, and forecasting
  detectors.

Deliverables:

- Detector API reference in documentation.
- Focused tests for all registered detector classes.
- Backward-compatible aliases for renamed detectors where practical.
- Migration notes for any detector behavior that changes.

## v0.3.0 - Benchmark Reproducibility

Goal: make benchmark results reproducible, comparable, and easier to share.

Planned work:

- Add a benchmark run manifest that records package version, Python version,
  dataset keys, detector keys, parameters, random seeds, and timestamp.
- Standardize random seed handling across supported detectors.
- Expand leaderboard output with runtime, failure category, and configuration
  hash fields.
- Add CLI options for output directory, JSON report output, and deterministic
  run IDs.
- Add dataset integrity checks for bundled benchmark files.
- Add small smoke benchmark configurations for CI.
- Document dataset provenance, license notes, and intended task type.

Deliverables:

- Reproducible benchmark report format.
- Versioned example benchmark configuration files.
- Tests that verify leaderboard/report schema stability.

## v0.4.0 - Dataset And Evaluation Expansion

Goal: broaden evaluation coverage without making the base install heavy.

Planned work:

- Add more compact tabular anomaly datasets with clear metadata.
- Add lightweight time-series benchmark examples for point anomalies and
  contextual anomalies.
- Add graph benchmark examples with node-level anomaly labels.
- Add evaluation metrics beyond ROC AUC, such as average precision and
  precision at k.
- Add dataset selectors for task type, modality, size, and dependency needs.
- Add clearer CLI output for skipped detectors and unsupported dataset-detector
  combinations.

Deliverables:

- Expanded `benchmarks/datasets.yml` metadata.
- Metric selection in benchmark configuration.
- Documentation for choosing datasets and metrics.

## v0.5.0 - Plugin And Extension Ecosystem

Goal: make external detector and dataset extensions practical.

Planned work:

- Formalize plugin loading contracts for detectors.
- Add plugin metadata validation and collision reporting.
- Support external dataset loader registration.
- Add examples for a minimal detector plugin and a dataset plugin.
- Add safer plugin import diagnostics in the CLI.
- Document compatibility expectations for plugin authors.

Deliverables:

- Plugin author guide.
- Example plugin package under documentation or examples.
- Tests for plugin registration, override behavior, and failure messages.

## v0.6.0 - Visualization And Reporting

Goal: turn benchmark outputs into useful reports for inspection and comparison.

Planned work:

- Add report generation from leaderboard or JSON benchmark outputs.
- Add plots for ROC curves, precision-recall curves, score distributions, and
  runtime comparisons.
- Add dataset summary exports in Markdown and JSON.
- Improve visualization tests so plotting code is stable in headless CI.
- Keep plotting optional enough that non-report workflows remain lightweight.

Deliverables:

- CLI report command or report mode.
- Documented output examples.
- Tested visualization helpers for common benchmark outputs.

## v0.7.0 - Optional Deep Learning Maturity

Goal: make deep detector support more reliable without bloating the base
install.

Planned work:

- Separate PyTorch and TensorFlow extras if dependency conflicts make combined
  installation fragile.
- Add deterministic training controls for deep detectors where feasible.
- Add checkpoint loading/saving documentation.
- Add small CPU-friendly deep detector smoke tests for optional CI jobs.
- Improve early stopping and validation split behavior.
- Document expected input shapes for sequence and reconstruction models.

Deliverables:

- Deep detector usage guide.
- Optional CI coverage for at least one PyTorch-backed detector.
- Clear error messages when deep dependencies are missing.

## v1.0.0 - Stable Public Release

Goal: declare the public API and benchmark formats stable enough for downstream
users to depend on.

Requirements:

- Detector API is documented and covered by compatibility tests.
- CLI configuration schema is documented and versioned.
- Leaderboard and JSON report schemas are documented and versioned.
- Plugin contracts are documented and covered by tests.
- Supported Python versions are clearly declared and validated in CI.
- Release workflow is repeatable from the GitHub Actions dashboard.
- Zenodo citation metadata is current for the release.

Non-goals for `v1.0.0`:

- Supporting every anomaly detection algorithm.
- Bundling large datasets directly in the repository.
- Making all optional detector stacks part of the base install.

## Backlog

These items are useful but not yet assigned to a target version.

- Add documentation site generation.
- Add API docs from docstrings.
- Add benchmark result comparison across GitHub releases.
- Add model persistence helpers for detectors that support serialization.
- Evaluate Python `3.13` support once the dependency stack is stable.
- Add automated release note generation from merged pull requests.
- Add DOI badges once the preferred badge target is finalized.

## Contribution Guidance

When proposing roadmap work, include:

- Target version.
- User-facing behavior.
- Dependency impact.
- Testing plan.
- Documentation impact.
- Migration risk, if any.

Small, well-tested pull requests are preferred over broad rewrites. Roadmap
items can move between versions when implementation risk or dependency support
changes.
