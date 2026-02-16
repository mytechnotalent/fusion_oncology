"""
Dataset retrieval from the UCI ML Repository (TCGA PANCAN HiSeq).

Downloads the ZIP archive once, caches it locally, and exposes clean
DataFrames for downstream analysis.
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from fusion_oncology.config import ProjectConfig
from fusion_oncology.data.cache import ArtifactCache

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Downloads and parses the TCGA Pan-Cancer gene-expression dataset.

    Parameters
    ----------
    config : ProjectConfig
        Runtime configuration (data URL, cache path, …).
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the data ingestion pipeline.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration supplying the data URL, cache
            directory, and variance-filter quantile.  Falls back to
            default settings when omitted.
        """
        self.cfg = config or ProjectConfig()
        self.cache = ArtifactCache(self.cfg.cache_dir)

    # ── private helpers ──────────────────────────────────────────────────

    def _fetch_remote_zip(self) -> bytes:
        """Stream the ZIP archive from the remote URL and cache it.

        Downloads the full archive from ``config.data_url``, persists
        the raw bytes in the artefact cache, and returns them.

        Returns
        -------
        bytes
            Raw bytes of the downloaded ZIP archive.
        """
        logger.info("Downloading dataset from %s …", self.cfg.data_url)
        resp = requests.get(self.cfg.data_url, timeout=120)
        resp.raise_for_status()
        raw = resp.content
        self.cache.save_bytes("tcga_pancan_zip", raw)
        return raw

    def _download_zip(self) -> zipfile.ZipFile:
        """Download (or load from cache) the remote ZIP archive.

        Checks the local artefact cache first.  On a cache miss the
        full archive is streamed from ``config.data_url`` and persisted
        for subsequent runs.

        Returns
        -------
        zipfile.ZipFile
            The in-memory ZIP file ready for extraction.
        """
        cache_key = "tcga_pancan_zip"
        raw = (
            self.cache.load_bytes(cache_key)
            if self.cache.has_bytes(cache_key)
            else self._fetch_remote_zip()
        )
        return zipfile.ZipFile(io.BytesIO(raw))

    @staticmethod
    def _match_csv_in_inner_zip(
        z: zipfile.ZipFile,
        name: str,
        tag: str,
    ) -> pd.DataFrame | None:
        """Search for a matching CSV inside a nested ZIP entry.

        UCI sometimes double-zips its datasets, so this helper opens
        an inner ZIP archive and scans for a CSV whose name contains
        *tag*.

        Parameters
        ----------
        z : zipfile.ZipFile
            The outer ZIP archive containing the nested ZIP.
        name : str
            Filename of the inner ZIP entry within *z*.
        tag : str
            Substring that must appear in the target CSV filename.

        Returns
        -------
        pd.DataFrame or None
            The parsed CSV as a DataFrame, or ``None`` if no match is
            found inside the inner archive.
        """
        inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
        for inner_name in inner.namelist():
            if tag in inner_name and inner_name.endswith(".csv"):
                return pd.read_csv(inner.open(inner_name))
        return None

    @staticmethod
    def _match_csv_entry(
        z: zipfile.ZipFile,
        name: str,
        tag: str,
    ) -> pd.DataFrame | None:
        """Check a single ZIP entry for a CSV matching *tag*.

        If the entry is itself a ``.csv`` whose name contains *tag* it
        is parsed directly.  If the entry is a nested ``.zip`` archive
        it is delegated to ``_match_csv_in_inner_zip``.

        Parameters
        ----------
        z : zipfile.ZipFile
            The (outer) ZIP archive being searched.
        name : str
            Filename of the current entry within *z*.
        tag : str
            Substring that must appear in the target CSV filename.

        Returns
        -------
        pd.DataFrame or None
            The parsed CSV as a DataFrame, or ``None`` if the entry
            does not match.
        """
        if tag in name and name.endswith(".csv"):
            return pd.read_csv(z.open(name))
        if name.endswith(".zip"):
            return DataIngestion._match_csv_in_inner_zip(z, name, tag)
        return None

    @staticmethod
    def _find_csv(z: zipfile.ZipFile, tag: str) -> pd.DataFrame:
        """Locate a CSV inside a (possibly nested) ZIP by substring match.

        Iterates through every entry in the archive and delegates
        matching logic to ``_match_csv_entry``.

        Parameters
        ----------
        z : zipfile.ZipFile
            The (outer) ZIP archive to search.
        tag : str
            Substring that must appear in the target filename
            (e.g. ``"data"`` or ``"labels"``).

        Returns
        -------
        pd.DataFrame
            The parsed CSV loaded into a DataFrame.

        Raises
        ------
        FileNotFoundError
            If no CSV matching *tag* is found in the archive.
        """
        for name in z.namelist():
            result = DataIngestion._match_csv_entry(z, name, tag)
            if result is not None:
                return result
        raise FileNotFoundError(f"No CSV matching '{tag}' found in archive")

    @staticmethod
    def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
        """Remove auto-generated ``Unnamed:`` columns from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame potentially containing unnamed index columns.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with unnamed columns removed.
        """
        return df.loc[:, ~df.columns.str.contains("^Unnamed")]

    def _load_cached_patient_data(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """Return cached patient data if both artefacts exist.

        Looks for previously persisted ``X_clean`` and ``y_clean``
        DataFrames in the artefact cache and returns them when
        available.

        Returns
        -------
        tuple of (pd.DataFrame, pd.Series) or None
            The cached feature matrix and label series, or ``None``
            when the cache does not contain both artefacts.
        """
        if not (self.cache.has_dataframe("X_clean") and self.cache.has_dataframe("y_clean")):
            return None
        X = self.cache.load_dataframe("X_clean")
        y_df = self.cache.load_dataframe("y_clean")
        return X, y_df.iloc[:, 0]

    def _extract_labels(self, y_df: pd.DataFrame) -> pd.Series:
        """Extract the target label series from a labels DataFrame.

        Uses the ``Class`` column when present; otherwise falls back
        to the first column.

        Parameters
        ----------
        y_df : pd.DataFrame
            Labels DataFrame returned by the archive.

        Returns
        -------
        pd.Series
            One-dimensional cancer-type labels.
        """
        return y_df["Class"] if "Class" in y_df.columns else y_df.iloc[:, 0]

    def _persist_and_log(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Cache cleaned data and log ingestion summary.

        Saves the feature matrix and label series as parquet artefacts
        and emits an informational log message summarising the dataset
        dimensions.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned gene-expression feature matrix.
        y : pd.Series
            Cancer-type labels aligned with *X*.
        """
        self.cache.save_dataframe("X_clean", X)
        self.cache.save_dataframe("y_clean", y.to_frame())
        logger.info(
            "Ingested %d samples x %d genes, %d cancer types",
            X.shape[0],
            X.shape[1],
            y.nunique(),
        )

    # ── public API ───────────────────────────────────────────────────────

    def get_patient_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Full ingestion pipeline: download → extract → clean.

        Returns
        -------
        X : pd.DataFrame
            Gene-expression feature matrix  (samples x genes).
        y : pd.Series
            Cancer-type labels aligned with X.
        """
        cached = self._load_cached_patient_data()
        if cached is not None:
            return cached
        z = self._download_zip()
        X = self._drop_unnamed(self._find_csv(z, "data"))
        y = self._extract_labels(self._drop_unnamed(self._find_csv(z, "labels")))
        self._persist_and_log(X, y)
        return X, y
