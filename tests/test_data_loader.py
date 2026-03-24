import pandas as pd

from data_loader import load_dataset


def test_load_dataset(tmp_path):
    (tmp_path / "still").mkdir()
    csv_path = tmp_path / "still" / "sample.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_path, index=False, sep=';')

    data, labels = load_dataset(str(tmp_path), ["still"])

    assert len(data) == 1
    assert labels == ["still"]
    assert data[0].shape == (2, 2)
