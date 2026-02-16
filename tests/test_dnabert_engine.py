"""Tests for the DNABERT-2 sequence embedding engine.

All transformer and torch operations are mocked so that tests
run instantly without a GPU or model download.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Return a minimal ProjectConfig for DNABERT tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    ProjectConfig
        Config with default model settings.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


# ── _select_device tests ────────────────────────────────────────────────


class TestSelectDevice:
    """Tests for ``DNABERTEngine._select_device``."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_prefers_cuda(self, _mock) -> None:
        """Verify CUDA is preferred when available.

        Parameters
        ----------
        _mock : MagicMock
            Patch for ``torch.cuda.is_available``.
        """
        from fusion_oncology.models.dnabert_engine import DNABERTEngine

        assert DNABERTEngine._select_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_falls_back_to_mps(self, _mps, _cuda) -> None:
        """Verify MPS is used when CUDA is unavailable.

        Parameters
        ----------
        _mps : MagicMock
            Patch for ``torch.backends.mps.is_available``.
        _cuda : MagicMock
            Patch for ``torch.cuda.is_available``.
        """
        from fusion_oncology.models.dnabert_engine import DNABERTEngine

        assert DNABERTEngine._select_device() == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_falls_back_to_cpu(self, _mps, _cuda) -> None:
        """Verify CPU is the last-resort device.

        Parameters
        ----------
        _mps : MagicMock
            Patch for ``torch.backends.mps.is_available``.
        _cuda : MagicMock
            Patch for ``torch.cuda.is_available``.
        """
        from fusion_oncology.models.dnabert_engine import DNABERTEngine

        assert DNABERTEngine._select_device() == "cpu"


# ── init tests ───────────────────────────────────────────────────────────


class TestInit:
    """Tests for ``DNABERTEngine.__init__``."""

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_default_config(self, _mps, _cuda) -> None:
        """Verify omitting config uses a default ProjectConfig.

        Parameters
        ----------
        _mps : MagicMock
            Patch for ``torch.backends.mps.is_available``.
        _cuda : MagicMock
            Patch for ``torch.cuda.is_available``.
        """
        from fusion_oncology.models.dnabert_engine import DNABERTEngine

        engine = DNABERTEngine()
        assert engine.cfg is not None
        assert engine.device == "cpu"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_lazy_model_none(self, _mps, _cuda, cfg) -> None:
        """Verify model and tokenizer start as ``None``.

        Parameters
        ----------
        _mps : MagicMock
            Patch for ``torch.backends.mps.is_available``.
        _cuda : MagicMock
            Patch for ``torch.cuda.is_available``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.models.dnabert_engine import DNABERTEngine

        engine = DNABERTEngine(cfg)
        assert engine._model is None
        assert engine._tokenizer is None


# ── helper to build a fully-mocked engine ────────────────────────────────


def _make_mock_engine(cfg):
    """Create a DNABERTEngine with mocked tokenizer and model.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    DNABERTEngine
        Engine with ``_tokenizer`` and ``_model`` pre-set to mocks
        that return deterministic tensors.
    """
    from fusion_oncology.models.dnabert_engine import DNABERTEngine

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        engine = DNABERTEngine(cfg)
    engine._tokenizer = _build_mock_tokenizer(cfg)
    engine._model = _build_mock_model()
    engine.device = "cpu"  # Force CPU regardless of detection
    return engine


def _build_mock_tokenizer(cfg):
    """Build a mock tokenizer that returns deterministic tensors.

    Parameters
    ----------
    cfg : ProjectConfig
        Supplies ``max_seq_len``.

    Returns
    -------
    MagicMock
        A tokenizer mock whose __call__ returns token dicts.
    """
    tok = MagicMock()
    tok.side_effect = lambda *a, **kw: {
        "input_ids": torch.ones(1, cfg.max_seq_len, dtype=torch.long, device="cpu"),
        "attention_mask": torch.ones(1, cfg.max_seq_len, dtype=torch.long, device="cpu"),
    }
    return tok


def _build_mock_model():
    """Build a mock transformer model returning fake hidden states.

    Returns
    -------
    MagicMock
        A model mock whose __call__ returns a tuple of (hidden, pooler).
    """
    model = MagicMock()
    hidden = torch.randn(1, 512, 768, device="cpu")
    model.side_effect = lambda **kw: (hidden, None)
    return model


# ── _tokenize_sequence tests ────────────────────────────────────────────


class TestTokenizeSequence:
    """Tests for ``DNABERTEngine._tokenize_sequence``."""

    def test_returns_dict_with_expected_keys(self, cfg) -> None:
        """Verify token dict contains input_ids and attention_mask.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_mock_engine(cfg)
        tokens = engine._tokenize_sequence("ACGTACGT")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens


# ── _mean_pool tests ────────────────────────────────────────────────────


class TestMeanPool:
    """Tests for ``DNABERTEngine._mean_pool``."""

    def test_returns_1d_numpy(self, cfg) -> None:
        """Verify mean-pooling returns a 1-D numpy array.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_mock_engine(cfg)
        tokens = engine._tokenize_sequence("ACGTACGT")
        result = engine._mean_pool(tokens)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1


# ── embed tests ──────────────────────────────────────────────────────────


class TestEmbed:
    """Tests for ``DNABERTEngine.embed``."""

    def test_returns_768d_vector(self, cfg) -> None:
        """Verify embedding is a 768-dimensional vector.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_mock_engine(cfg)
        result = engine.embed("ACGTACGTACGT")
        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)


# ── _tokenize_batch tests ───────────────────────────────────────────────


def _build_batch_mock_tokenizer(cfg):
    """Build a mock tokenizer for batch inputs.

    Parameters
    ----------
    cfg : ProjectConfig
        Supplies ``max_seq_len``.

    Returns
    -------
    MagicMock
        A tokenizer mock returning batch-shaped tensors.
    """
    tok = MagicMock()
    tok.side_effect = lambda *a, **kw: {
        "input_ids": torch.ones(2, cfg.max_seq_len, dtype=torch.long, device="cpu"),
        "attention_mask": torch.ones(2, cfg.max_seq_len, dtype=torch.long, device="cpu"),
    }
    return tok


def _build_batch_mock_model():
    """Build a mock model returning batch-shaped hidden states.

    Returns
    -------
    MagicMock
        Model mock whose __call__ returns ``(hidden, None)``
        with batch dimension 2.
    """
    model = MagicMock()
    hidden = torch.randn(2, 512, 768, device="cpu")
    model.side_effect = lambda **kw: (hidden, None)
    return model


def _make_batch_engine(cfg):
    """Create a DNABERTEngine mocked for batch operations.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    DNABERTEngine
        Engine with batch-sized mock tokenizer and model.
    """
    from fusion_oncology.models.dnabert_engine import DNABERTEngine

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        engine = DNABERTEngine(cfg)
    engine._tokenizer = _build_batch_mock_tokenizer(cfg)
    engine._model = _build_batch_mock_model()
    engine.device = "cpu"
    return engine


class TestTokenizeBatch:
    """Tests for ``DNABERTEngine._tokenize_batch``."""

    def test_returns_batch_tensors(self, cfg) -> None:
        """Verify batch tokenization returns batch-sized tensors.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_batch_engine(cfg)
        tokens = engine._tokenize_batch(["ACGT", "TGCA"])
        assert tokens["input_ids"].shape[0] == 2


# ── _embed_batch tests ──────────────────────────────────────────────────


class TestEmbedBatch:
    """Tests for ``DNABERTEngine._embed_batch``."""

    def test_returns_2d_numpy(self, cfg) -> None:
        """Verify batch embedding returns a 2-D numpy array.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_batch_engine(cfg)
        tokens = engine._tokenize_batch(["ACGT", "TGCA"])
        result = engine._embed_batch(tokens)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2


# ── batch_embed tests ───────────────────────────────────────────────────


class TestBatchEmbed:
    """Tests for ``DNABERTEngine.batch_embed``."""

    def test_returns_correct_shape(self, cfg) -> None:
        """Verify output rows match input count with 768 dimensions.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _make_batch_engine(cfg)
        result = engine.batch_embed(["ACGT", "TGCA"], batch_size=2)
        assert result.shape[0] == 2
        assert result.shape[1] == 768
