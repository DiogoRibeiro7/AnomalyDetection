import pytest

pytest.importorskip("networkx")

from anomalybench.benchmarks.load_datasets import (
    load_cardio,
    load_digits,
    load_iris,
    load_karate_club_graph,
    load_lympho,
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


def test_cardio_loader_feature_columns_are_deterministic_and_label_free():
    df1, features1, label1, name1 = load_cardio()
    df2, features2, label2, name2 = load_cardio()

    assert name1 == "cardio"
    assert label1 == "Class"
    assert name2 == "cardio"
    assert label2 == "Class"
    assert "Class" not in features1
    assert "Class" not in features2
    assert features1 == features2
    assert features1 == [column for column in df1.columns if column != "Class"]


def test_lympho_loader_feature_columns_are_deterministic_and_label_free():
    df1, features1, label1, name1 = load_lympho()
    df2, features2, label2, name2 = load_lympho()

    assert name1 == "lympho"
    assert label1 == "Class"
    assert name2 == "lympho"
    assert label2 == "Class"
    assert "Class" not in features1
    assert "Class" not in features2
    assert features1 == features2
    assert features1 == [column for column in df1.columns if column != "Class"]
