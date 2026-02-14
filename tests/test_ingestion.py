"""Tests for the data ingestion module.

All network requests and ZIP I/O are mocked so tests run
without downloading from UCI.
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig
from fusion_oncology.data.ingestion import DataIngestion


@pytest.fixture()
def cfg(tmp_path):
    """Return a minimal ProjectConfig for ingestion tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    ProjectConfig
        Config with temp cache and output directories.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


# ── ZIP creation helpers ────────────────────────────────────────────────


def _csv_bytes(name: str, content: str) -> tuple[str, bytes]:
    """Encode a CSV string as bytes with a filename.

    Parameters
    ----------
    name : str
        Filename for the CSV entry.
    content : str
        CSV text content.

    Returns
    -------
    tuple[str, bytes]
        ``(filename, encoded_bytes)``.
    """
    return name, content.encode()


def _make_zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """Build an in-memory ZIP archive from entry pairs.

    Parameters
    ----------
    entries : list[tuple[str, bytes]]
        List of ``(filename, content_bytes)`` pairs.

    Returns
    -------
    bytes
        Raw ZIP archive bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _sample_data_csv() -> str:
    """Return a minimal TCGA-style data CSV string.

    Returns
    -------
    str
        CSV with two genes and three samples.
    """
    return "GENE1,GENE2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n"


def _sample_labels_csv() -> str:
    """Return a minimal labels CSV string.

    Returns
    -------
    str
        CSV with a Class column and three labels.
    """
    return "Class\nBRCA\nLUAD\nKIRC\n"


def _build_simple_zip() -> bytes:
    """Build a flat ZIP containing data and labels CSVs.

    Returns
    -------
    bytes
        ZIP archive with ``pancan_data.csv`` and ``pancan_labels.csv``.
    """
    entries = [
        _csv_bytes("pancan_data.csv", _sample_data_csv()),
        _csv_bytes("pancan_labels.csv", _sample_labels_csv()),
    ]
    return _make_zip_bytes(entries)


# ── __init__ tests ──────────────────────────────────────────────────────


class TestInit:
    """Tests for ``DataIngestion.__init__``."""

    def test_default_config(self) -> None:
        """Verify omitting config uses default ProjectConfig.

        Asserts that the ``cfg`` attribute is not ``None``
        after construction without arguments.
        """
        ing = DataIngestion()
        assert ing.cfg is not None

    def test_custom_config(self, cfg) -> None:
        """Verify a custom config is stored on the instance.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        assert ing.cfg is cfg


# ── _drop_unnamed tests ─────────────────────────────────────────────────


class TestDropUnnamed:
    """Tests for ``DataIngestion._drop_unnamed``."""

    def test_removes_unnamed_columns(self) -> None:
        """Verify columns matching ``Unnamed:*`` are dropped.

        Creates a DataFrame with an unnamed column and asserts
        it is removed after calling ``_drop_unnamed``.
        """
        df = pd.DataFrame({"A": [1], "Unnamed: 0": [2]})
        result = DataIngestion._drop_unnamed(df)
        assert "A" in result.columns
        assert "Unnamed: 0" not in result.columns

    def test_no_unnamed_unchanged(self) -> None:
        """Verify DataFrames without unnamed columns pass through.

        Asserts the column list is identical before and after
        the ``_drop_unnamed`` call.
        """
        df = pd.DataFrame({"A": [1], "B": [2]})
        result = DataIngestion._drop_unnamed(df)
        assert list(result.columns) == ["A", "B"]


# ── _find_csv tests ─────────────────────────────────────────────────────


class TestFindCsv:
    """Tests for ``DataIngestion._find_csv``."""

    def test_finds_matching_csv(self) -> None:
        """Verify a CSV whose name contains the tag is located.

        Builds a ZIP archive and asserts that ``_find_csv``
        returns a DataFrame with three rows.
        """
        raw = _build_simple_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        df = DataIngestion._find_csv(zf, "data")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_raises_on_missing_tag(self) -> None:
        """Verify FileNotFoundError is raised when no CSV matches.

        Passes a nonexistent tag and asserts the expected
        exception is raised.
        """
        raw = _build_simple_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        with pytest.raises(FileNotFoundError):
            DataIngestion._find_csv(zf, "nonexistent")


# ── _match_csv_entry tests ──────────────────────────────────────────────


class TestMatchCsvEntry:
    """Tests for ``DataIngestion._match_csv_entry``."""

    def test_matches_csv_directly(self) -> None:
        """Verify a ``.csv`` entry whose name contains tag matches.

        Opens a ZIP and asserts ``_match_csv_entry`` returns a
        non-None DataFrame for a matching filename.
        """
        raw = _build_simple_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        result = DataIngestion._match_csv_entry(zf, "pancan_data.csv", "data")
        assert result is not None

    def test_no_match_returns_none(self) -> None:
        """Verify a non-matching entry returns None.

        Passes a tag that does not appear in the filename
        and asserts the result is ``None``.
        """
        raw = _build_simple_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        result = DataIngestion._match_csv_entry(zf, "pancan_data.csv", "xyz")
        assert result is None


# ── _match_csv_in_inner_zip tests ───────────────────────────────────────


class TestMatchCsvInInnerZip:
    """Tests for ``DataIngestion._match_csv_in_inner_zip``."""

    def _build_nested_zip(self) -> bytes:
        """Build a ZIP containing a nested ZIP with a CSV inside.

        Returns
        -------
        bytes
            Outer ZIP with an inner ``inner.zip`` containing a CSV.
        """
        inner_buf = io.BytesIO()
        with zipfile.ZipFile(inner_buf, "w") as inner_zf:
            inner_zf.writestr("nested_data.csv", _sample_data_csv())
        outer_buf = io.BytesIO()
        with zipfile.ZipFile(outer_buf, "w") as outer_zf:
            outer_zf.writestr("inner.zip", inner_buf.getvalue())
        return outer_buf.getvalue()

    def test_finds_csv_in_inner_zip(self) -> None:
        """Verify a CSV is located inside a nested ZIP.

        Builds a nested archive and asserts the inner CSV
        is found with the correct row count.
        """
        raw = self._build_nested_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        result = DataIngestion._match_csv_in_inner_zip(zf, "inner.zip", "data")
        assert result is not None
        assert len(result) == 3

    def test_no_match_returns_none(self) -> None:
        """Verify None is returned when inner ZIP has no match.

        Searches for a tag absent from the nested CSV and
        asserts the result is ``None``.
        """
        raw = self._build_nested_zip()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        result = DataIngestion._match_csv_in_inner_zip(zf, "inner.zip", "labels")
        assert result is None


# ── _extract_labels tests ───────────────────────────────────────────────


class TestExtractLabels:
    """Tests for ``DataIngestion._extract_labels``."""

    def test_uses_class_column(self, cfg) -> None:
        """Verify the ``Class`` column is extracted when present.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        y_df = pd.DataFrame({"Class": ["BRCA", "LUAD"]})
        result = ing._extract_labels(y_df)
        assert list(result) == ["BRCA", "LUAD"]

    def test_falls_back_to_first_column(self, cfg) -> None:
        """Verify the first column is used when ``Class`` is absent.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        y_df = pd.DataFrame({"Label": ["A", "B"]})
        result = ing._extract_labels(y_df)
        assert list(result) == ["A", "B"]


# ── get_patient_data tests ──────────────────────────────────────────────


class TestGetPatientData:
    """Tests for ``DataIngestion.get_patient_data``."""

    @patch.object(DataIngestion, "_download_zip")
    def test_returns_tuple(self, mock_dl, cfg) -> None:
        """Verify (X, y) tuple is returned from downloaded ZIP.

        Parameters
        ----------
        mock_dl : MagicMock
            Patch for ``_download_zip``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        raw = _build_simple_zip()
        mock_dl.return_value = zipfile.ZipFile(io.BytesIO(raw))
        ing = DataIngestion(cfg)
        X, y = ing.get_patient_data()
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 3
        assert len(y) == 3

    @patch.object(DataIngestion, "_load_cached_patient_data")
    def test_uses_cache(self, mock_cache, cfg) -> None:
        """Verify cached data is returned when available.

        Parameters
        ----------
        mock_cache : MagicMock
            Patch for ``_load_cached_patient_data``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        X = pd.DataFrame({"G1": [1, 2]})
        y = pd.Series(["A", "B"])
        mock_cache.return_value = (X, y)
        ing = DataIngestion(cfg)
        result_X, result_y = ing.get_patient_data()
        assert len(result_X) == 2


# ── _persist_and_log tests ──────────────────────────────────────────────


class TestPersistAndLog:
    """Tests for ``DataIngestion._persist_and_log``."""

    def test_calls_cache_save(self, cfg) -> None:
        """Verify cache.save_dataframe is called for both X and y.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        X = pd.DataFrame({"G1": [1]})
        y = pd.Series(["A"])
        ing._persist_and_log(X, y)
        assert ing.cache.save_dataframe.call_count == 2


# ── _fetch_remote_zip tests ─────────────────────────────────────────────


class TestFetchRemoteZip:
    """Tests for ``DataIngestion._fetch_remote_zip``."""

    @patch("fusion_oncology.data.ingestion.requests.get")
    def test_fetches_and_caches(self, mock_get, cfg) -> None:
        """Should download ZIP bytes and cache them.

        Parameters
        ----------
        mock_get : MagicMock
            Patch for ``requests.get``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        mock_resp = MagicMock()
        mock_resp.content = _build_simple_zip()
        mock_get.return_value = mock_resp
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        result = ing._fetch_remote_zip()
        assert isinstance(result, bytes)
        ing.cache.save_bytes.assert_called_once()


# ── _download_zip tests ─────────────────────────────────────────────────


class TestDownloadZip:
    """Tests for ``DataIngestion._download_zip``."""

    def test_uses_cache_when_available(self, cfg) -> None:
        """Should return a ZipFile from cached bytes.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        raw = _build_simple_zip()
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        ing.cache.has_bytes.return_value = True
        ing.cache.load_bytes.return_value = raw
        zf = ing._download_zip()
        assert isinstance(zf, zipfile.ZipFile)

    @patch("fusion_oncology.data.ingestion.requests.get")
    def test_downloads_when_not_cached(self, mock_get, cfg) -> None:
        """Should download when cache is empty.

        Parameters
        ----------
        mock_get : MagicMock
            Patch for ``requests.get``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        mock_resp = MagicMock()
        mock_resp.content = _build_simple_zip()
        mock_get.return_value = mock_resp
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        ing.cache.has_bytes.return_value = False
        ing.cache.save_bytes.return_value = None
        zf = ing._download_zip()
        assert isinstance(zf, zipfile.ZipFile)


# ── _load_cached_patient_data tests ─────────────────────────────────────


class TestLoadCachedPatientData:
    """Tests for ``DataIngestion._load_cached_patient_data``."""

    def test_returns_none_when_not_cached(self, cfg) -> None:
        """Should return None when cache does not have both frames.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        ing.cache.has_dataframe.return_value = False
        assert ing._load_cached_patient_data() is None

    def test_returns_cached_data(self, cfg) -> None:
        """Should return (X, y) when both frames are cached.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        ing = DataIngestion(cfg)
        ing.cache = MagicMock()
        ing.cache.has_dataframe.return_value = True
        ing.cache.load_dataframe.side_effect = [
            pd.DataFrame({"G1": [1, 2]}),
            pd.DataFrame({"Label": ["A", "B"]}),
        ]
        result = ing._load_cached_patient_data()
        assert result is not None
        assert len(result[0]) == 2
