"""Tests for the standalone Local Outlier Factor implementation.

``analytics.lof`` backs ``LOFDetector`` but had no dedicated tests, so the
distance semantics, normalization, and neighbour collection below are pinned
here rather than inferred from the detector that wraps them.
"""

from __future__ import annotations

import math

import pytest
from dataexcept import DataValidationError

from anomalybench.analytics.lof import (
    LOF,
    clear_distance_cache,
    distance_euclidean,
    k_distance,
    local_outlier_factor,
    outliers,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Keep cached pairwise distances from leaking between tests."""

    clear_distance_cache()


def test_distance_is_root_mean_squared_not_plain_euclidean() -> None:
    """The name says Euclidean but the implementation divides by dimension."""

    assert distance_euclidean((0, 0), (3, 4)) == pytest.approx(math.sqrt(12.5))


def test_distance_is_symmetric() -> None:
    forward = distance_euclidean((1.0, 2.0), (4.0, 6.0))
    reverse = distance_euclidean((4.0, 6.0), (1.0, 2.0))
    assert forward == pytest.approx(reverse)


def test_distance_accepts_lists_and_tuples_alike() -> None:
    assert distance_euclidean([0, 0], (3, 4)) == pytest.approx(
        distance_euclidean((0, 0), (3, 4))
    )


def test_distance_rejects_mismatched_lengths() -> None:
    with pytest.raises(DataValidationError, match="different number of arguments"):
        distance_euclidean((1.0, 2.0), (1.0,))


def test_distance_rejects_mixed_attribute_types() -> None:
    with pytest.raises(DataValidationError, match="different data types"):
        distance_euclidean((1.0,), ("a",))


def test_unequal_strings_count_as_a_unit_difference() -> None:
    """String attributes contribute 0 when equal and 1 when they differ."""

    assert distance_euclidean(("a", "b"), ("a", "c")) == pytest.approx(math.sqrt(0.5))
    assert distance_euclidean(("a", "b"), ("a", "b")) == pytest.approx(0.0)


def test_cached_distance_survives_until_cleared() -> None:
    first = distance_euclidean((0.0, 0.0), (1.0, 1.0))
    cached = distance_euclidean((0.0, 0.0), (1.0, 1.0))
    assert cached == first

    clear_distance_cache()
    assert distance_euclidean((0.0, 0.0), (1.0, 1.0)) == pytest.approx(first)


def test_k_distance_groups_neighbours_by_equal_distance() -> None:
    """Ties share a rank, so k groups can yield more than k neighbours."""

    grid = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    value, neighbours = k_distance(2, (0.0, 0.0), grid)

    assert value == pytest.approx(math.sqrt(0.5))
    # The point itself at distance 0, then both points tied at sqrt(0.5).
    assert neighbours == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]


def test_k_distance_falls_back_to_the_farthest_group() -> None:
    """Asking for more groups than exist uses the largest distance."""

    points = [(0.0, 0.0), (1.0, 1.0)]
    value, neighbours = k_distance(10, (0.0, 0.0), points)

    assert value == pytest.approx(distance_euclidean((0.0, 0.0), (1.0, 1.0)))
    assert neighbours == points


def test_normalize_instances_maps_each_dimension_onto_the_unit_interval() -> None:
    lof = LOF([(0.0, 10.0), (2.0, 20.0), (4.0, 30.0)], normalize=True)
    assert lof.instances == [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]


def test_normalize_instance_rescales_unseen_points_with_stored_bounds() -> None:
    lof = LOF([(0.0, 10.0), (4.0, 30.0)], normalize=True)
    assert lof.normalize_instance((2.0, 20.0)) == pytest.approx((0.5, 0.5))


def test_constant_dimension_warns_and_normalizes_to_zero() -> None:
    """A dimension with no spread cannot be scaled, so it collapses to zero."""

    with pytest.warns(UserWarning, match="No data variation in dimensions: 2"):
        lof = LOF([(0.0, 5.0), (2.0, 5.0), (4.0, 5.0)], normalize=True)

    assert [instance[1] for instance in lof.instances] == [0, 0, 0]


def test_local_outlier_factor_separates_an_isolated_point_from_the_cluster() -> None:
    cluster = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]

    isolated = local_outlier_factor(2, (9.0, 9.0), cluster)
    inside = local_outlier_factor(2, (0.05, 0.05), cluster)

    assert isolated > 1.0
    assert inside < 1.0
    assert isolated > inside


def test_outliers_reports_index_instance_and_score_ranked_by_score() -> None:
    points = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (9.0, 9.0)]

    found = outliers(2, points, normalize=False)

    assert found, "the isolated point should be reported"
    assert sorted(found[0]) == ["index", "instance", "lof"]
    assert found[0]["index"] == 4
    assert found[0]["instance"] == (9.0, 9.0)
    assert [entry["lof"] for entry in found] == sorted(
        (entry["lof"] for entry in found), reverse=True
    )


def test_lof_class_scores_new_points_against_normalized_training_data() -> None:
    cluster = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (5.0, 5.0)]
    lof = LOF(cluster, normalize=True)

    isolated = lof.local_outlier_factor(2, (5.0, 5.0))
    inside = lof.local_outlier_factor(2, (0.05, 0.05))

    assert isolated > inside
