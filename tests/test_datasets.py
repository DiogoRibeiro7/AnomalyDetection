from benchmarks.load_datasets import load_iris


def test_iris_loader():
    df, features, label, name = load_iris()
    assert name == "iris"
    assert len(features) > 0
    assert label == "Class"
    assert not df.empty
