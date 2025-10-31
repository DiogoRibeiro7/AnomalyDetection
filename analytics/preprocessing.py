"""Preprocessing utilities for anomaly detection workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ArrayLike = NDArray[np.floating[Any]]


@dataclass
class PreprocessingPipeline:
    """Data preprocessing pipeline with sensible defaults.

    The pipeline performs the following steps in order:

    1. Clip numeric features to configurable quantile bounds.
    2. Impute missing numeric values with the median and categorical values
       with the most frequent category.
    3. Scale numeric features using :class:`~sklearn.preprocessing.StandardScaler`.
    4. One-hot encode categorical features while ignoring unknown categories.

    Parameters
    ----------
    clip_quantile : float, default=0.01
        Lower quantile used to compute symmetric clipping bounds. Values must be
        in the half-open interval ``[0, 0.5)``. The upper bound is calculated as
        ``1 - clip_quantile``. A value of ``0`` disables clipping.
    numeric_impute_strategy : str, default="median"
        Strategy passed to :class:`~sklearn.impute.SimpleImputer` for numeric
        columns.
    categorical_impute_strategy : str, default="most_frequent"
        Strategy passed to :class:`~sklearn.impute.SimpleImputer` for categorical
        columns.
    one_hot_sparse : bool, default=False
        Whether the :class:`~sklearn.preprocessing.OneHotEncoder` should return a
        sparse matrix.

    Notes
    -----
    All intermediate estimators are stored so the same pipeline instance can be
    reused across training and inference datasets.
    """

    clip_quantile: float = 0.01
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    one_hot_sparse: bool = False
    _transformer: ColumnTransformer | None = field(init=False, default=None)
    _numeric_bounds: dict[str, tuple[float, float]] | None = field(
        init=False, default=None
    )
    _numeric_columns: list[str] | None = field(init=False, default=None)
    _categorical_columns: list[str] | None = field(init=False, default=None)
    feature_names_out_: list[str] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 0 <= self.clip_quantile < 0.5:
            raise ValueError("clip_quantile must fall within [0, 0.5)")

    def fit(self, data: pd.DataFrame | ArrayLike) -> PreprocessingPipeline:
        """Fit preprocessing steps on ``data``."""

        frame = self._ensure_frame(data)
        numeric_index = frame.select_dtypes(include=[np.number]).columns
        categorical_index = frame.columns.difference(numeric_index)
        self._numeric_columns = list(numeric_index)
        self._categorical_columns = list(categorical_index)

        if len(self._numeric_columns) > 0 and self.clip_quantile > 0:
            lower = frame[self._numeric_columns].quantile(self.clip_quantile)
            upper = frame[self._numeric_columns].quantile(1 - self.clip_quantile)
            self._numeric_bounds = {
                col: (float(lower[col]), float(upper[col]))
                for col in self._numeric_columns
            }
        else:
            self._numeric_bounds = {}

        transformers: list[tuple[str, Pipeline, Iterable[str]]] = []
        if len(self._numeric_columns) > 0:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_transformer, self._numeric_columns))
        if len(self._categorical_columns) > 0:
            encoder_kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
            encoder_params = signature(OneHotEncoder).parameters
            if "sparse_output" in encoder_params:
                encoder_kwargs["sparse_output"] = self.one_hot_sparse
            else:
                encoder_kwargs["sparse"] = self.one_hot_sparse
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy=self.categorical_impute_strategy),
                    ),
                    ("encoder", OneHotEncoder(**encoder_kwargs)),
                ]
            )
            transformers.append(
                ("categorical", categorical_transformer, self._categorical_columns)
            )

        if transformers:
            self._transformer = ColumnTransformer(transformers=transformers)
            self._transformer.fit(frame)
            names_out = self._transformer.get_feature_names_out()
            self.feature_names_out_ = list(names_out)
        else:
            self._transformer = None
            self.feature_names_out_ = list(frame.columns)
        return self

    def transform(self, data: pd.DataFrame | ArrayLike) -> NDArray[np.float64]:
        """Transform ``data`` using the fitted preprocessing steps."""

        frame = self._ensure_frame(data)
        clipped = self._clip_numeric(frame)
        if self._transformer is None:
            return clipped.to_numpy(dtype=float)
        transformed = self._transformer.transform(clipped)
        if hasattr(transformed, "toarray") and not self.one_hot_sparse:
            transformed = transformed.toarray()  # type: ignore[assignment]
        return np.asarray(transformed, dtype=float)

    def fit_transform(self, data: pd.DataFrame | ArrayLike) -> NDArray[np.float64]:
        """Convenience method equivalent to calling :meth:`fit` then ``transform``."""

        self.fit(data)
        return self.transform(data)

    @property
    def numeric_clip_bounds_(self) -> dict[str, tuple[float, float]] | None:
        """Return the learned numeric clipping bounds if available."""

        if self._numeric_bounds is None:
            return None
        return dict(self._numeric_bounds)

    def _ensure_frame(self, data: pd.DataFrame | ArrayLike) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "PreprocessingPipeline expects a pandas DataFrame or numpy array"
            )
        columns = [f"feature_{idx}" for idx in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)

    def _clip_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._numeric_bounds:
            return frame
        clipped = frame.copy()
        for column, (lower, upper) in self._numeric_bounds.items():
            if column in clipped:
                clipped[column] = clipped[column].clip(lower=lower, upper=upper)
        return clipped
