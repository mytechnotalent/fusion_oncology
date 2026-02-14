"""
DNABERT-2 sequence embedding engine.

Loads the transformer model once and exposes methods for embedding
arbitrary DNA sequences and computing pairwise similarity.
"""

from __future__ import annotations

import logging
from functools import lru_cache
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
            config = AutoConfig.from_pretrained(self.cfg.model_path, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.cfg.model_path,
                config=config,
                trust_remote_code=True,
            ).to(self.device)
            # DNABERT-2 bundles a Triton flash-attention kernel that is
            # incompatible with Triton >= 3.0 (removed ``trans_b`` kwarg
            # from ``tl.dot``).  The attention branch in
            # ``BertUnpadSelfAttention.forward()`` is:
            #
            #     if self.p_dropout or flash_attn_qkvpacked_func is None:
            #         <standard PyTorch attention>   <-- we want this
            #     else:
            #         <broken Triton flash attention>
            #
            # Monkey-patch ``flash_attn_qkvpacked_func`` to ``None`` in
            # the dynamically-loaded ``bert_layers`` module so the model
            # always falls back to standard PyTorch attention.
            for mod_name, mod in sys.modules.items():
                if "bert_layers" in mod_name and hasattr(mod, "flash_attn_qkvpacked_func"):
                    mod.flash_attn_qkvpacked_func = None
                    logger.info(
                        "Patched %s: flash_attn disabled (Triton compat)",
                        mod_name,
                    )
            self._model.eval()
        return self._model

    # ── embedding helpers ───────────────────────────────────────────────

    def _tokenize_sequence(
        self,
        sequence: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a single DNA sequence and move tensors to device.

        Parameters
        ----------
        sequence : str
            Raw DNA string (characters A / C / G / T).

        Returns
        -------
        dict[str, torch.Tensor]
            Token tensors on the compute device.
        """
        tok = self.tokenizer(
            sequence[: self.cfg.max_seq_len],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_seq_len,
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    def _mean_pool(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Run a forward pass and mean-pool the last hidden state.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Tokenized inputs already on the target device.

        Returns
        -------
        np.ndarray
            Shape ``(hidden_dim,)`` — single-sequence embedding.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

    # ── public embedding API ─────────────────────────────────────────────

    def embed(self, sequence: str) -> np.ndarray:
        """Generate a fixed-size embedding for a DNA sequence.

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
        inputs = self._tokenize_sequence(sequence)
        return self._mean_pool(inputs)

    # ── batch embedding helpers ──────────────────────────────────────────

    def _tokenize_batch(
        self,
        sequences: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of DNA sequences and move to device.

        Parameters
        ----------
        sequences : list[str]
            Batch of raw DNA strings.

        Returns
        -------
        dict[str, torch.Tensor]
            Token tensors on the compute device.
        """
        truncated = [s[: self.cfg.max_seq_len] for s in sequences]
        tok = self.tokenizer(
            truncated,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_seq_len,
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    def _embed_batch(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Forward-pass and mean-pool for a token batch.

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Tokenized batch inputs on the target device.

        Returns
        -------
        np.ndarray
            Shape ``(batch, hidden_dim)``.
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # ── public batch API ─────────────────────────────────────────────────

    def batch_embed(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed multiple sequences efficiently in batches.

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
            inputs = self._tokenize_batch(batch)
            all_embeddings.append(self._embed_batch(inputs))
        return np.vstack(all_embeddings)
