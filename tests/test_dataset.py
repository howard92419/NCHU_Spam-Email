import os
import pandas as pd
import pytest

from src.data.dataset import SpamDataset


def create_csv(path, rows, header=False):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, header=header, encoding="utf-8")


def test_load_data_from_local_csv(tmp_path):
    # Create a headerless CSV matching the expected dataset format
    data = [("spam", "Win money now!!!"), ("ham", "Hello friend")]
    csv_path = tmp_path / "spam_local.csv"
    create_csv(csv_path, data, header=False)

    ds = SpamDataset(data_path=str(csv_path))
    X, y = ds.load_data()

    assert len(X) == 2
    assert list(y) == [1, 0]
    assert "Win money now" in X[0]


def test_load_data_invalid_format_raises(tmp_path):
    # Create a CSV with only one column which should trigger ValueError
    df = pd.DataFrame({"only": ["a", "b"]})
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False, header=False)

    ds = SpamDataset(data_path=str(bad_csv))
    with pytest.raises(ValueError):
        ds.load_data()
