"""
DNABERT-2 sequence embedding engine.

Loads the transformer model once and exposes methods for embedding
arbitrary DNA sequences and computing pairwise similarity.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


class DNABERTEngine:
    """
    Wraps the DNABERT-2 transformer for DNA sequence embedding.

    The model is loaded lazily on first use and kept on the best
    available device (CUDA → MPS → CPU).

    Parameters
    ----------
    config : ProjectConfig
        Supplies ``model_path`` and ``max_seq_len``.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the DNABERT engine (model loaded lazily).

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self.device = self._select_device()
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None

    # ── device selection ─────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> str:
        """Choose the best available compute device.

        Preference order: CUDA → Apple MPS → CPU.

        Returns
        -------
        str
            Device identifier string (``"cuda"``, ``"mps"``, or ``"cpu"``).
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── lazy loading ─────────────────────────────────────────────────────

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazily load and return the DNABERT-2 tokenizer.

        The tokenizer is downloaded from Hugging Face Hub on first
        access and cached in memory for subsequent calls.

        Returns
        -------
        AutoTokenizer
            The loaded tokenizer instance.
        """
        if self._tokenizer is None:
            logger.info("Loading tokenizer from %s …", self.cfg.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_path, trust_remote_code=True
            )
        return self._tokenizer

    @property
    def model(self) -> AutoModel:
        """Lazily load and return the DNABERT-2 model.

        The model is downloaded from Hugging Face Hub on first access,
        moved to the selected device, set to eval mode, and cached in
        memory for subsequent calls.

        Returns
        -------
        AutoModel
            The loaded transformer model in evaluation mode.
        """
        if self._model is None:
            logger.info("Loading DNABERT-2 on %s …", self.device)
            self._model = AutoModel.from_pretrained(
                self.cfg.model_path, trust_remote_code=True
            ).to(self.device)
            self._model.eval()
        return self._model

    # ── embedding ────────────────────────────────────────────────────────

    def embed(self, sequence: str) -> np.ndarray:
        """
        Generate a fixed-size embedding for a DNA sequence.

        The sequence is truncated to ``max_seq_len`` tokens, mean-pooled
        across the sequence dimension, and returned as a 1-D numpy vector.

        Parameters
        ----------
        sequence : str
            Raw DNA string (characters A / C / G / T).

        Returns
        -------
        np.ndarray
            Shape ``(hidden_dim,)`` — typically 768 for DNABERT-2-117M.
        """
        inputs = self.tokenizer(
            sequence[: self.cfg.max_seq_len],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean-pool the last hidden state over the sequence axis
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        return embedding

    def batch_embed(self, sequences: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Embed multiple sequences efficiently in batches.

        Parameters
        ----------
        sequences : list[str]
            List of DNA sequences.
        batch_size : int
            Number of sequences per forward pass.

        Returns
        -------
        np.ndarray
            Shape ``(len(sequences), hidden_dim)``.
        """
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            truncated = [s[: self.cfg.max_seq_len] for s in batch]
            inputs = self.tokenizer(
                truncated,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.max_seq_len,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embs)
        return np.vstack(all_embeddings)
