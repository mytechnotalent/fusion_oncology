"""Tests for the artifact cache layer."""

import numpy as np
import pandas as pd

from fusion_oncology.data.cache import ArtifactCache


def test_dataframe_roundtrip(tmp_path):
    """Verify DataFrame save/load round-trips through Parquet.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cache = ArtifactCache(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cache.save_dataframe("test_df", df)
    assert cache.has_dataframe("test_df")
    loaded = cache.load_dataframe("test_df")
    pd.testing.assert_frame_equal(df, loaded)


def test_array_roundtrip(tmp_path):
    """Verify NumPy array save/load round-trips through ``.npy``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cache = ArtifactCache(tmp_path)
    arr = np.array([1.0, 2.0, 3.0])
    cache.save_array("test_arr", arr)
    assert cache.has_array("test_arr")
    loaded = cache.load_array("test_arr")
    np.testing.assert_array_equal(arr, loaded)


def test_bytes_roundtrip(tmp_path):
    """Verify raw-bytes save/load round-trips through ``.bin``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cache = ArtifactCache(tmp_path)
    data = b"hello world"
    cache.save_bytes("test_bytes", data)
    assert cache.has_bytes("test_bytes")
    assert cache.load_bytes("test_bytes") == data


def test_json_roundtrip(tmp_path):
    """Verify JSON dict save/load round-trips through ``.json``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cache = ArtifactCache(tmp_path)
    obj = {"key": "value", "num": 42}
    cache.save_json("test_json", obj)
    assert cache.has_json("test_json")
    assert cache.load_json("test_json") == obj


def test_clear(tmp_path):
    """Verify :meth:`clear` removes all cached artefacts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cache = ArtifactCache(tmp_path)
    cache.save_bytes("a", b"x")
    cache.save_bytes("b", b"y")
    n = cache.clear()
    assert n == 2
    assert not cache.has_bytes("a")
