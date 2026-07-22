# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning.

## v0.3.0 - Unreleased

- Added reproducible benchmark manifests and versioned JSON reports.
- Added CLI and YAML config options for output directories, JSON reports,
  stable run IDs, and random seeds.
- Expanded leaderboard rows with run IDs, configuration hashes, runtime, random
  seed, and failure category fields.
- Added metadata-driven integrity checks for bundled benchmark dataset files.
- Added a versioned smoke benchmark configuration for `v0.3.0`.

## v0.2.0 - 2026-07-22

- Added detector lifecycle enforcement through `BaseDetector.is_fitted`.
- Added clear `score`-before-`fit` errors across built-in detectors.
- Added shared tabular input coercion for detector implementations.
- Added detector score-orientation metadata.
- Added detector API documentation and lifecycle tests.

## v0.1.0 - 2026-07-22

- Modernized project metadata and packaging configuration.
- Added repository hygiene files for licensing, editor defaults, and releases.
- Added citation metadata for GitHub and Zenodo archiving.
