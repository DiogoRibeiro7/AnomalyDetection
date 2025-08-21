# Contributing

Thank you for your interest in improving the anomaly detection library. This guide explains how to add new algorithms and contribute changes.

## Getting Started
1. Fork the repository and clone your fork.
2. Install dependencies and pre-commit hooks:
   ```bash
   poetry install
   pre-commit install
   ```
3. Ensure tests run:
   ```bash
   pytest -q
   ```

## Adding a New Detector
1. Create a detector class inheriting from `BaseDetector` in `analytics/detector.py` or a dedicated module.
2. Implement the `fit` and `score` methods. Use existing detectors as references.
3. Register the detector in the CLI registry so it can be selected with `--detectors`.
4. Add unit tests covering the new detector’s behaviour.
5. Update the README with a brief description of the detector and any usage notes.

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
