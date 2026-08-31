"""Shared time-series input and windowing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SequenceArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class WindowSpec:
    """Describe a deterministic rolling-window transformation."""

    window_length: int
    stride: int = 1
    horizon: int = 0
    aggregation: str = "window"
    label_alignment: str = "window_end"

    def __post_init__(self) -> None:
        if self.window_length < 2:
            raise ValueError("window_length must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        if self.aggregation != "window":
            raise ValueError("Only window aggregation is currently supported")
        if self.label_alignment != "window_end":
            raise ValueError("Only window_end label alignment is currently supported")

    def as_dict(self) -> dict[str, int | str]:
        return {
            "window_length": self.window_length,
            "stride": self.stride,
            "horizon": self.horizon,
            "aggregation": self.aggregation,
            "label_alignment": self.label_alignment,
        }


def _as_float_array(data: pd.DataFrame | Any) -> SequenceArray:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=float, copy=False)
    return np.asarray(data, dtype=float)


def coerce_sequence_batch(
    data: pd.DataFrame | Any,
    *,
    window_spec: WindowSpec | None = None,
) -> SequenceArray:
    """Return data with shape ``(batch, sequence_length, channels)``.

    Three-dimensional inputs are already interpreted as explicit batches of
    multivariate sequences. Two-dimensional inputs have two supported meanings:

    - without ``window_spec`` each row is a complete univariate sequence;
    - with ``window_spec`` rows are ordered time points and rolling windows are
      constructed across the first axis, preserving columns as channels.
    """

    array = _as_float_array(data)
    if array.ndim == 3:
        if array.shape[0] == 0 or array.shape[1] < 2 or array.shape[2] == 0:
            raise ValueError(
                "Sequence input must contain non-empty sequences of length >= 2"
            )
        return array

    if array.ndim != 2:
        raise ValueError("Time-series input must be a 2-D or 3-D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Time-series input must not be empty")

    if window_spec is None:
        if array.shape[1] < 2:
            raise ValueError(
                "Pre-windowed series input must contain at least two time steps per row"
            )
        return array[:, :, np.newaxis]

    starts = window_start_indices(array.shape[0], window_spec)
    if starts.size == 0:
        raise ValueError(
            "Time series is too short for the requested window_length and horizon"
        )
    windows = [
        array[start : start + window_spec.window_length]
        for start in starts.tolist()
    ]
    return np.stack(windows, axis=0)


def window_start_indices(n_points: int, spec: WindowSpec) -> NDArray[np.int_]:
    """Return deterministic rolling-window start positions."""

    last_start = n_points - spec.window_length - spec.horizon
    if last_start < 0:
        return np.asarray([], dtype=int)
    return np.arange(0, last_start + 1, spec.stride, dtype=int)


def window_label_indices(n_points: int, spec: WindowSpec) -> NDArray[np.int_]:
    """Return point-label indices aligned to each produced window."""

    starts = window_start_indices(n_points, spec)
    return starts + spec.window_length - 1 + spec.horizon


def align_point_labels(labels: Any, spec: WindowSpec) -> NDArray[Any]:
    """Align point labels to rolling windows using the configured window end."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("Point labels must be one-dimensional")
    indices = window_label_indices(label_array.shape[0], spec)
    return label_array[indices]
