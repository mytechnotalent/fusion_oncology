"""
Local disk cache for downloaded datasets and computed embeddings.

Avoids re-downloading the ~80 MB TCGA archive on every run and allows
intermediate artefacts (e.g. DNABERT embeddings) to be persisted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ArtifactCache:
    """Simple file-system cache keyed on content hashes.

    Stores DataFrames (Parquet), NumPy arrays (``.npy``), raw bytes
    (``.bin``), and JSON metadata under a single directory, using
    truncated SHA-256 hashes as filenames.

    Parameters
    ----------
    cache_dir : Path
        Root directory for cached artefacts.  Created if it does not
        exist.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialise the cache and ensure the root directory exists.

        Parameters
        ----------
        cache_dir : Path
            Root directory for cached artefacts.
        """
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _hash(key: str) -> str:
        """Return a 16-character hex digest of *key* (SHA-256).

        Parameters
        ----------
        key : str
            Arbitrary cache key string.

        Returns
        -------
        str
            First 16 hex characters of the SHA-256 hash.
        """
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _path(self, key: str, suffix: str = ".parquet") -> Path:
        """Build the full filesystem path for a cached artefact.

        Parameters
        ----------
        key : str
            Logical cache key.
        suffix : str
            File extension (e.g. ``".parquet"``, ``".npy"``, ``".bin"``).

        Returns
        -------
        Path
            Absolute path where the artefact would be stored.
        """
        return self.root / f"{self._hash(key)}{suffix}"

    # ── dataframe cache ──────────────────────────────────────────────────

    def has_dataframe(self, key: str) -> bool:
        """Check whether a DataFrame is cached under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        bool
            ``True`` if a Parquet file exists for this key.
        """
        return self._path(key).exists()

    def load_dataframe(self, key: str) -> pd.DataFrame:
        """Load a previously cached DataFrame from Parquet.

        Parameters
        ----------
        key : str
            Logical cache key used when the DataFrame was saved.

        Returns
        -------
        pd.DataFrame
            The deserialised DataFrame.
        """
        path = self._path(key)
        logger.info("Cache HIT  – loading %s", path.name)
        return pd.read_parquet(path)

    def save_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame as Parquet under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.
        df : pd.DataFrame
            DataFrame to serialise.
        """
        path = self._path(key)
        df.to_parquet(path, index=False)
        logger.info("Cache SAVE – %s  (%d rows)", path.name, len(df))

    # ── numpy array cache ────────────────────────────────────────────────

    def has_array(self, key: str) -> bool:
        """Check whether a NumPy array is cached under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        bool
            ``True`` if a ``.npy`` file exists for this key.
        """
        return self._path(key, ".npy").exists()

    def load_array(self, key: str) -> np.ndarray:
        """Load a previously cached NumPy array.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        np.ndarray
            The deserialised array.
        """
        path = self._path(key, ".npy")
        logger.info("Cache HIT  – loading %s", path.name)
        return np.load(path)

    def save_array(self, key: str, arr: np.ndarray) -> None:
        """Persist a NumPy array as ``.npy`` under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.
        arr : np.ndarray
            Array to serialise.
        """
        path = self._path(key, ".npy")
        np.save(path, arr)
        logger.info("Cache SAVE – %s", path.name)

    # ── raw bytes cache (zip files, etc.) ────────────────────────────────

    def has_bytes(self, key: str) -> bool:
        """Check whether raw bytes are cached under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        bool
            ``True`` if a ``.bin`` file exists for this key.
        """
        return self._path(key, ".bin").exists()

    def load_bytes(self, key: str) -> bytes:
        """Load previously cached raw bytes.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        bytes
            The raw byte content.
        """
        path = self._path(key, ".bin")
        logger.info("Cache HIT  – loading %s", path.name)
        return path.read_bytes()

    def save_bytes(self, key: str, data: bytes) -> None:
        """Persist raw bytes under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.
        data : bytes
            Raw bytes to store.
        """
        path = self._path(key, ".bin")
        path.write_bytes(data)
        logger.info("Cache SAVE – %s  (%d bytes)", path.name, len(data))

    # ── JSON metadata cache ──────────────────────────────────────────────

    def has_json(self, key: str) -> bool:
        """Check whether a JSON object is cached under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        bool
            ``True`` if a ``.json`` file exists for this key.
        """
        return self._path(key, ".json").exists()

    def load_json(self, key: str) -> Any:
        """Load a previously cached JSON-serialisable object.

        Parameters
        ----------
        key : str
            Logical cache key.

        Returns
        -------
        Any
            The deserialised Python object (dict, list, etc.).
        """
        path = self._path(key, ".json")
        return json.loads(path.read_text())

    def save_json(self, key: str, obj: Any) -> None:
        """Persist a JSON-serialisable object under *key*.

        Parameters
        ----------
        key : str
            Logical cache key.
        obj : Any
            Object to serialise (must be JSON-compatible).
        """
        path = self._path(key, ".json")
        path.write_text(json.dumps(obj, indent=2, default=str))

    # ── housekeeping ─────────────────────────────────────────────────────

    def clear(self) -> int:
        """Remove all cached artefacts from the cache directory.

        Returns
        -------
        int
            The number of files deleted.
        """
        count = 0
        for p in self.root.iterdir():
            if p.is_file():
                p.unlink()
                count += 1
        logger.info("Cache CLEAR – removed %d files", count)
        return count
