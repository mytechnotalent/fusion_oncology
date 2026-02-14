"""
Mutational instability scoring via embedding drift.

For a given DNA sequence we:
  1.  Compute the reference DNABERT embedding.
  2.  Introduce *N* random single-nucleotide mutations.
  3.  Re-embed each mutant and measure cosine distance to the reference.
  4.  Report the mean drift as the *instability score*.

A high score implies that even small mutations cause large shifts in
the learned representation, suggesting structural fragility — a
promising property for therapeutic targeting.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.dnabert_engine import DNABERTEngine

logger = logging.getLogger(__name__)

NUCLEOTIDES = np.array(["A", "C", "G", "T"])


class InstabilityAnalyzer:
    """
    Measures how sensitive a gene's embedding is to point mutations.

    Parameters
    ----------
    bert : DNABERTEngine
        A pre-initialised DNABERT engine.
    config : ProjectConfig
        Supplies ``fuzz_iterations``.
    """

    def __init__(self, bert: DNABERTEngine, config: ProjectConfig | None = None) -> None:
        """Initialise the instability analyser.

        Parameters
        ----------
        bert : DNABERTEngine
            A pre-initialised DNABERT engine used to compute embeddings.
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.
        """
        self.bert = bert
        self.cfg = config or ProjectConfig()
        self._rng = np.random.default_rng(seed=42)

    # ── core logic ───────────────────────────────────────────────────────

    def _mutate(self, seq: str) -> str:
        """Introduce a single random point mutation into *seq*.

        A random position is selected and its nucleotide is replaced
        with a uniformly chosen *different* base.

        Parameters
        ----------
        seq : str
            Original DNA sequence.

        Returns
        -------
        str
            Mutated sequence (exactly one base changed).
        """
        bases = list(seq)
        idx = self._rng.integers(0, len(bases))
        original = bases[idx]
        choices = [b for b in NUCLEOTIDES if b != original]
        bases[idx] = self._rng.choice(choices)
        return "".join(bases)

    def _compute_drifts(self, sequence: str) -> list[float]:
        """Compute cosine drifts for *N* random single-nucleotide mutants.

        Parameters
        ----------
        sequence : str
            Reference DNA sequence.

        Returns
        -------
        list[float]
            One drift value per fuzz iteration.
        """
        ref_emb = self.bert.embed(sequence)
        drifts: list[float] = []
        for _ in range(self.cfg.fuzz_iterations):
            mutant = self._mutate(sequence)
            mut_emb = self.bert.embed(mutant)
            drifts.append(float(1.0 - cosine_similarity([ref_emb], [mut_emb])[0][0]))
        return drifts

    def _log_instability(self, gene: str, mean_drift: float, std_drift: float) -> None:
        """Log instability statistics at DEBUG level.

        Parameters
        ----------
        gene : str
            Gene symbol.
        mean_drift : float
            Mean cosine drift across iterations.
        std_drift : float
            Standard deviation of cosine drift.
        """
        logger.debug(
            "%s instability: mean=%.6f  std=%.6f  (n=%d)",
            gene,
            mean_drift,
            std_drift,
            self.cfg.fuzz_iterations,
        )

    def score(self, gene: str, sequence: str) -> float:
        """Compute the instability score for *gene*.

        Parameters
        ----------
        gene : str
            Gene symbol (used only for logging).
        sequence : str
            DNA sequence (A/C/G/T characters).

        Returns
        -------
        float
            Mean cosine drift in [0, 2].  Higher → more unstable.
        """
        if len(sequence) < 10:
            logger.warning("Sequence for %s too short (%d bp)", gene, len(sequence))
            return 0.0
        drifts = self._compute_drifts(sequence)
        mean_drift = float(np.mean(drifts))
        self._log_instability(gene, mean_drift, float(np.std(drifts)))
        return mean_drift

    def _empty_report(self, gene: str) -> dict:
        """Return a zeroed-out instability report for short sequences.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        dict
            Report with all drift values set to zero.
        """
        return {
            "gene": gene,
            "mean_drift": 0,
            "std_drift": 0,
            "max_drift": 0,
            "min_drift": 0,
            "all_drifts": [],
        }

    def _build_report(self, gene: str, drifts: list[float]) -> dict:
        """Assemble an instability report dictionary from raw drifts.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        drifts : list[float]
            Per-mutation cosine drift values.

        Returns
        -------
        dict
            Keys: ``gene``, ``mean_drift``, ``std_drift``, ``max_drift``,
            ``min_drift``, ``all_drifts``.
        """
        return {
            "gene": gene,
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "max_drift": float(np.max(drifts)),
            "min_drift": float(np.min(drifts)),
            "all_drifts": drifts,
        }

    def detailed_report(self, gene: str, sequence: str) -> dict:
        """Return a richer instability breakdown for downstream plots.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        sequence : str
            Reference DNA sequence for the gene.

        Returns
        -------
        dict
            Keys: ``gene``, ``mean_drift``, ``std_drift``, ``max_drift``,
            ``min_drift``, ``all_drifts``.
        """
        if len(sequence) < 10:
            return self._empty_report(gene)
        drifts = self._compute_drifts(sequence)
        return self._build_report(gene, drifts)
