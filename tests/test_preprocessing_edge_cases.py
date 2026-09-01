"""Edge cases for the preprocessing pipeline and its detector integration.

``test_preprocessing_pipeline`` covers the ordinary mixed-column path. The
input coercion, clipping bounds, all-categorical and sparse variants, and the
``BaseDetector`` hooks that drive the pipeline are pinned here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from analytics.base import OrientedScores
from analytics.detectors import get_detector_class
from analytics.preprocessing import PreprocessingPipeline


@pytest.fixture
def mixed() -> pd.DataFrame:
    return pd.DataFrame({"num": [1.0, 2.0, 3.0, 100.0], "cat": ["a", "b", "a", "c"]})


def test_clip_quantile_must_stay_below_one_half() -> None:
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        PreprocessingPipeline(clip_quantile=0.5)


def test_clip_quantile_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        PreprocessingPipeline(clip_quantile=-0.1)


def test_bounds_are_unavailable_before_fitting() -> None:
    assert PreprocessingPipeline().numeric_clip_bounds_ is None


def test_mixed_columns_are_scaled_and_one_hot_encoded(mixed: pd.DataFrame) -> None:
    pipeline = PreprocessingPipeline()
    transformed = pipeline.fit_transform(mixed)

    assert transformed.shape == (4, 4)
    assert pipeline.feature_names_out_ == [
        "numeric__num",
        "categorical__cat_a",
        "categorical__cat_b",
        "categorical__cat_c",
    ]


def test_all_categorical_input_records_no_numeric_bounds() -> None:
    pipeline = PreprocessingPipeline()
    transformed = pipeline.fit_transform(pd.DataFrame({"cat": ["a", "b", "a", "c"]}))

    assert transformed.shape == (4, 3)
    assert pipeline.numeric_clip_bounds_ == {}


def test_sparse_one_hot_output_is_still_returned_as_a_dense_array(
    mixed: pd.DataFrame,
) -> None:
    transformed = PreprocessingPipeline(one_hot_sparse=True).fit_transform(mixed)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (4, 4)


def test_clipping_narrows_the_learned_bounds(mixed: pd.DataFrame) -> None:
    unclipped = PreprocessingPipeline().fit(mixed).numeric_clip_bounds_
    clipped = PreprocessingPipeline(clip_quantile=0.1).fit(mixed).numeric_clip_bounds_

    assert unclipped is not None and clipped is not None
    assert clipped["num"][1] < unclipped["num"][1]


def test_clipping_bounds_are_applied_to_new_data(mixed: pd.DataFrame) -> None:
    pipeline = PreprocessingPipeline(clip_quantile=0.1)
    pipeline.fit(mixed)
    bounds = pipeline.numeric_clip_bounds_
    assert bounds is not None

    # An extreme value must be pulled back inside the learned upper bound.
    extreme = pd.DataFrame({"num": [10_000.0], "cat": ["a"]})
    transformed = pipeline.transform(extreme)

    reference = pipeline.transform(
        pd.DataFrame({"num": [bounds["num"][1]], "cat": ["a"]})
    )
    np.testing.assert_allclose(transformed, reference)


def test_numpy_input_is_given_positional_feature_names() -> None:
    pipeline = PreprocessingPipeline().fit(np.arange(8.0).reshape(4, 2))

    assert pipeline.feature_names_out_ == [
        "numeric__feature_0",
        "numeric__feature_1",
    ]


def test_unsupported_input_types_are_rejected() -> None:
    # Held as Any so the deliberate misuse type-checks the same everywhere.
    unsupported: Any = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(TypeError, match="DataFrame or numpy array"):
        PreprocessingPipeline().fit(unsupported)


def test_detectors_have_no_preprocessing_pipeline_by_default() -> None:
    assert get_detector_class("isolation_forest")().preprocessing_pipeline is None


def test_preprocessed_fit_and_score_round_trip(mixed: pd.DataFrame) -> None:
    detector = get_detector_class("isolation_forest")()
    pipeline = PreprocessingPipeline()
    detector.set_preprocessing_pipeline(pipeline)

    assert detector.preprocessing_pipeline is pipeline

    detector.fit_preprocessed(mixed, n_estimators=8, random_state=0)
    scores = np.asarray(detector.score_preprocessed(mixed))

    assert scores.shape == (len(mixed),)


def test_scoring_before_the_pipeline_is_fitted_is_refused(
    mixed: pd.DataFrame,
) -> None:
    detector = get_detector_class("isolation_forest")()
    detector.set_preprocessing_pipeline(PreprocessingPipeline())

    with pytest.raises(RuntimeError, match="fit_preprocessed"):
        detector.score_preprocessed(mixed)


def test_clearing_the_pipeline_restores_passthrough(mixed: pd.DataFrame) -> None:
    detector = get_detector_class("isolation_forest")()
    detector.set_preprocessing_pipeline(PreprocessingPipeline())
    detector.set_preprocessing_pipeline(None)

    assert detector.preprocessing_pipeline is None
    # With no pipeline the data reaches fit untouched, so numeric input works.
    numeric = mixed[["num"]]
    detector.fit_preprocessed(numeric, n_estimators=8, random_state=0)
    assert np.asarray(detector.score_preprocessed(numeric)).shape == (len(numeric),)


def test_oriented_scores_keep_their_orientation_across_views() -> None:
    scores = OrientedScores([1.0, 2.0, 3.0], "lower_is_more_anomalous")

    view = scores[1:]

    assert isinstance(view, OrientedScores)
    assert view.score_orientation == "lower_is_more_anomalous"


def test_oriented_scores_default_to_estimator_defined_for_bare_views() -> None:
    """A plain array cast into the subclass carries no declared orientation."""

    bare = np.asarray([1.0, 2.0]).view(OrientedScores)

    assert bare.score_orientation == "estimator_defined"
