"""Tests for the rolling-window primitives in ``analytics.time_series``.

The contract tests exercise the happy paths through the temporal detectors,
most of which need PyTorch and skip without it. The validation and index
arithmetic below need neither, so they are pinned directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataexcept import DataValidationError, HyperparameterError

from anomalybench.analytics.time_series import (
    WindowedScores,
    WindowSpec,
    align_point_labels,
    coerce_sequence_batch,
    window_label_indices,
    window_start_indices,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_length": 1}, "must be at least 2"),
        ({"window_length": 4, "stride": 0}, "must be at least 1"),
        ({"window_length": 4, "horizon": -1}, "must be non-negative"),
        (
            {"window_length": 4, "aggregation": "mean"},
            "only window aggregation is currently supported",
        ),
        (
            {"window_length": 4, "label_alignment": "window_start"},
            "only window_end label alignment is currently supported",
        ),
    ],
)
def test_window_spec_rejects_unsupported_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(HyperparameterError, match=message):
        WindowSpec(**kwargs)  # type: ignore[arg-type]


def test_window_spec_round_trips_through_as_dict() -> None:
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    assert spec.as_dict()["window_length"] == 3
    assert spec.as_dict()["stride"] == 2


def test_window_start_indices_respect_stride_and_horizon() -> None:
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    # last valid start is 10 - 3 - 1 = 6, stepping by 2 from 0
    np.testing.assert_array_equal(window_start_indices(10, spec), [0, 2, 4, 6])


def test_window_start_indices_are_empty_when_the_series_is_too_short() -> None:
    spec = WindowSpec(window_length=8, horizon=2)
    assert window_start_indices(5, spec).size == 0


def test_window_label_indices_point_at_the_window_end_plus_horizon() -> None:
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    # start + window_length - 1 + horizon
    np.testing.assert_array_equal(window_label_indices(10, spec), [3, 5, 7, 9])


def test_align_point_labels_selects_the_aligned_positions() -> None:
    spec = WindowSpec(window_length=3, stride=2, horizon=1)
    labels = np.arange(10) * 10

    np.testing.assert_array_equal(align_point_labels(labels, spec), [30, 50, 70, 90])


def test_align_point_labels_requires_one_dimensional_labels() -> None:
    spec = WindowSpec(window_length=3)
    with pytest.raises(DataValidationError, match="one-dimensional"):
        align_point_labels(np.zeros((4, 2)), spec)


def test_windowed_scores_requires_one_dimensional_values() -> None:
    spec = WindowSpec(window_length=3)
    with pytest.raises(DataValidationError, match="one-dimensional"):
        WindowedScores(np.zeros((2, 2)), label_indices=[0, 1, 2, 3], window_spec=spec)


def test_windowed_scores_requires_matching_label_indices() -> None:
    spec = WindowSpec(window_length=3)
    with pytest.raises(DataValidationError, match="equal length"):
        WindowedScores([0.1, 0.2], label_indices=[0], window_spec=spec)


def test_windowed_scores_survive_numpy_views() -> None:
    """``__array_finalize__`` must carry the metadata onto derived arrays."""

    spec = WindowSpec(window_length=3)
    scores = WindowedScores([0.1, 0.2, 0.3], label_indices=[2, 3, 4], window_spec=spec)

    view = scores[1:]

    assert isinstance(view, WindowedScores)
    np.testing.assert_array_equal(view.label_indices, [2, 3, 4])
    assert view.window_spec == spec.as_dict()


def test_coerce_accepts_a_three_dimensional_batch_unchanged() -> None:
    batch = np.zeros((2, 4, 3))
    np.testing.assert_array_equal(coerce_sequence_batch(batch), batch)


@pytest.mark.parametrize(
    "batch",
    [
        np.zeros((0, 4, 3)),
        np.zeros((2, 1, 3)),
        np.zeros((2, 4, 0)),
    ],
)
def test_coerce_rejects_degenerate_three_dimensional_batches(
    batch: np.ndarray,
) -> None:
    with pytest.raises(DataValidationError, match="non-empty sequences"):
        coerce_sequence_batch(batch)


def test_coerce_rejects_one_dimensional_input() -> None:
    with pytest.raises(DataValidationError, match="2-D or 3-D"):
        coerce_sequence_batch(np.zeros(5))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_coerce_rejects_empty_two_dimensional_input(shape: tuple[int, int]) -> None:
    with pytest.raises(DataValidationError, match="must not be empty"):
        coerce_sequence_batch(np.zeros(shape))


def test_coerce_without_a_spec_treats_rows_as_series() -> None:
    rows = np.arange(6, dtype=float).reshape(2, 3)
    result = coerce_sequence_batch(rows)

    assert result.shape == (2, 3, 1)
    np.testing.assert_array_equal(result[:, :, 0], rows)


def test_coerce_without_a_spec_requires_two_time_steps() -> None:
    with pytest.raises(DataValidationError, match="at least two time steps"):
        coerce_sequence_batch(np.zeros((4, 1)))


def test_coerce_with_a_spec_builds_rolling_windows() -> None:
    series = np.arange(6, dtype=float).reshape(6, 1)
    spec = WindowSpec(window_length=3, stride=1)

    windows = coerce_sequence_batch(series, window_spec=spec)

    assert windows.shape == (4, 3, 1)
    np.testing.assert_array_equal(windows[0, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(windows[-1, :, 0], [3.0, 4.0, 5.0])


def test_coerce_with_a_spec_rejects_a_series_that_is_too_short() -> None:
    spec = WindowSpec(window_length=8, horizon=2)
    with pytest.raises(DataValidationError, match="too short"):
        coerce_sequence_batch(np.zeros((5, 1)), window_spec=spec)


def test_coerce_accepts_a_dataframe() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = coerce_sequence_batch(frame)

    assert result.shape == (3, 2, 1)
