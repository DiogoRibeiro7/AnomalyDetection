import pytest

pytest.importorskip("networkx")

from benchmarks.load_datasets import (
    load_iris,
    load_digits,
    load_karate_club_graph,
)


def test_iris_loader():
    df, features, label, name = load_iris()
    assert name == "iris"
    assert len(features) > 0
    assert label == "Class"
    assert not df.empty


def test_digits_loader():
    df, features, label, name = load_digits()
    assert name == "digits"
    assert len(features) > 0
    assert label == "Class"
    assert not df.empty


def test_graph_loader():
    pytest.importorskip("networkx")
    G, features, label, name = load_karate_club_graph()
    assert name == "karateClubGraph"
    assert label == "label"
    assert G.number_of_nodes() > 0
