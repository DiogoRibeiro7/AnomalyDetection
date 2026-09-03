# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning.

## v0.5.0 - Unreleased

### Breaking

- Replaced the builtin exceptions raised by this package with the structured
  hierarchy from [DataExcept](https://pypi.org/project/DataExcept/). Every error
  raised here now inherits from `dataexcept.DataExceptError` and **no longer
  inherits from `ValueError`, `KeyError`, `TypeError`, `RuntimeError`, or
  `ImportError`**. Code that caught those from this package must catch the
  DataExcept types, or `DataExceptError` to catch all of them.
- `benchmarks.config_benchmark.ConfigValidationError` now inherits from
  `dataexcept.ConfigurationError` instead of `ValueError`. Its message is
  unchanged and the offending key is available as `option`.

  | Previously | Now |
  | --- | --- |
  | `ValueError` on detector/field validation | `DataValidationError` |
  | `ValueError` on a parameter value | `HyperparameterError` |
  | `ValueError` on configuration | `ConfigurationError` |
  | `ValueError` on input shape | `DataValidationError` |
  | `TypeError` on an unsupported container type | `DataFormatError` |
  | `KeyError` on a missing dataframe column | `MissingColumnError` |
  | `KeyError`/`ValueError` on an unknown detector | `UnknownDetectorError` |
  | `KeyError` on an unknown dataset | `UnknownDatasetError` |
  | `RuntimeError` on scoring before fit | `DetectorNotFittedError` |
  | `ImportError` on an optional dependency | `DependencyError` |

- Added modern tabular detectors: ECOD, Random Feature Isolation Forest, and
  Random Network Distillation.
- Added seed support for modern tabular detectors in benchmark manifests and
  CLI runs.
- Added a versioned `v0.5.0` modern tabular smoke benchmark configuration.
- Added lifecycle and deterministic-behavior tests for modern tabular
  detectors.
- Expanded Zenodo archival metadata with a structured description, related
  identifiers, references, and a broader keyword set.
- Synchronized `CITATION.cff` with `.zenodo.json` and added keywords, a
  repository URL, and the concept DOI identifier.
- Added citation metadata tests and a release workflow check that fail when
  `.zenodo.json`, `CITATION.cff`, and the project version drift apart.
- Documented concept and version DOI usage, added a BibTeX entry, and added a
  DOI badge to the README.

## v0.4.0 - 2026-07-22

- Added configurable benchmark metrics, including ROC AUC, average precision,
  precision at k, recall at k, F1 at threshold, best F1, and runtime.
- Added metric configuration to benchmark manifests, JSON reports, and
  leaderboard rows.
- Added metadata-based dataset selectors for modality, task, and label type.
- Expanded dataset metadata with modality, label type, positive label, and
  label semantics fields.
- Added a versioned `v0.4.0` metrics smoke benchmark configuration.

## v0.3.0 - 2026-07-22

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
