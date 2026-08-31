"""Regression tests for the explicit time-series windowing contract."""

from __future__ import annotations

import numpy as np
import pytest

from analytics.detectors.registry import get_detector_class
from analytics.time_series import (
    WindowSpec,
    align_point_labels,
    coerce_sequence_batch,
    window_label_indices,
)


def test_pre_windowed_rows_become_univariate_sequences() -> None:
    data = np.arange(24, dtype=float).reshape(6, 4)

    sequences = coerce_sequence_batch(data)

    assert sequences.shape == (6, 4, 1)
    np.testing.assert_array_equal(sequences[:, :, 0], data)


def test_multivariate_point_stream_uses_rolling_windows() -> None:
    data = np.arange(30, dtype=float).reshape(10, 3)
    spec = WindowSpec(window_length=4, stride=2)

    sequences = coerce_sequence_batch(data, window_spec=spec)

    assert sequences.shape == (4, 4, 3)
    np.testing.assert_array_equal(sequences[0], data[0:4])
    np.testing.assert_array_equal(sequences[-1], data[6:10])


def test_window_labels_align_to_window_end_plus_horizon() -> None:
    labels = np.arange(12)
    spec = WindowSpec(window_length=4, stride=2, horizon=1)

    indices = window_label_indices(len(labels), spec)
    aligned = align_point_labels(labels, spec)

    np.testing.assert_array_equal(indices, np.array([4, 6, 8, 10]))
    np.testing.assert_array_equal(aligned, labels[indices])


def test_window_contract_rejects_sequence_length_one() -> None:
    with pytest.raises(ValueError, match="at least two time steps"):
        coerce_sequence_batch(np.ones((5, 1)))


def test_registry_uses_sequence_aware_temporal_detectors() -> None:
    lstm = get_detector_class("lstm_autoencoder")
    transformer = get_detector_class("transformer")

    assert lstm.__module__ == "analytics.detectors.temporal"
    assert transformer.__module__ == "analytics.detectors.temporal"


@pytest.mark.parametrize(
    "detector_key, fit_kwargs",
    [
        ("lstm_autoencoder", {"epochs": 1, "hidden_size": 3}),
        ("transformer", {"epochs": 1, "d_model": 4, "nhead": 2}),
    ],
)
def test_temporal_detectors_never_collapse_sequence_axis(
    detector_key: str,
    fit_kwargs: dict[str, int],
) -> None:
    pytest.importorskip("torch", reason="PyTorch is required for temporal models")
    rng = np.random.default_rng(7)
    rows_are_series = rng.normal(size=(8, 6)).astype(np.float32)
    detector_cls = get_detector_class(detector_key)
    detector = detector_cls()

    detector.fit(
        rows_are_series,
        validation_split=0.25,
        patience=1,
        **fit_kwargs,
    )

    assert detector.sequence_length == 6
    assert detector.input_dim == 1
    scores = detector.score(rows_are_series)
    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
