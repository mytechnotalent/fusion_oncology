"""
Tests for the network pharmacology module.

Covers InteractionNetwork graph construction, centrality computation,
polypharmacology scoring, and combination target finding.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fusion_oncology.analysis.network_pharmacology import InteractionNetwork
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path: "Path") -> ProjectConfig:
    """Provide a temporary ProjectConfig for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Configuration with temporary paths.
    """
    return ProjectConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")


@pytest.fixture()
def network(cfg: ProjectConfig) -> InteractionNetwork:
    """Create an InteractionNetwork instance.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    InteractionNetwork
        Network instance.
    """
    return InteractionNetwork(cfg)


class TestInteractionNetwork:
    """Tests for the InteractionNetwork class."""

    def test_network_has_nodes(self, network: InteractionNetwork) -> None:
        """The network should have at least one node.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        assert len(network.nodes) > 0

    def test_network_has_edges(self, network: InteractionNetwork) -> None:
        """The network should have at least one edge.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        assert len(network.edges) > 0

    def test_degree_centrality_returns_dataframe(
        self, network: InteractionNetwork
    ) -> None:
        """Degree centrality should return a DataFrame with expected columns.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        dc = network.degree_centrality()
        assert isinstance(dc, pd.DataFrame)
        assert "Node" in dc.columns
        assert "Centrality" in dc.columns
        assert len(dc) > 0

    def test_centrality_values_bounded(self, network: InteractionNetwork) -> None:
        """Centrality values should be in [0, 1].

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        dc = network.degree_centrality()
        assert dc["Centrality"].min() >= 0
        assert dc["Centrality"].max() <= 1

    def test_betweenness_centrality(self, network: InteractionNetwork) -> None:
        """Betweenness centrality should return a non-empty DataFrame.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        bc = network.betweenness_centrality()
        assert isinstance(bc, pd.DataFrame)
        assert len(bc) > 0

    def test_polypharmacology_known_drug(self, network: InteractionNetwork) -> None:
        """Polypharmacology for a known drug should have targets.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        result = network.polypharmacology_score("Erlotinib")
        assert result["drug"] == "Erlotinib"
        assert result["n_targets"] >= 1

    def test_polypharmacology_unknown_drug(self, network: InteractionNetwork) -> None:
        """Polypharmacology for an unknown drug should return zero targets.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        result = network.polypharmacology_score("FakeDrug123")
        assert result["n_targets"] == 0

    def test_find_combination_targets(self, network: InteractionNetwork) -> None:
        """Combination target search should return a list.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        result = network.find_combination_targets("EGFR")
        assert isinstance(result, list)

    def test_strategic_score_range(self, network: InteractionNetwork) -> None:
        """Strategic score should be between 0 and 1.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        for gene in ["EGFR", "BRAF", "NOTREAL"]:
            score = network.strategic_score(gene)
            assert 0.0 <= score <= 1.0

    def test_strategic_score_known_gene_nonzero(
        self, network: InteractionNetwork
    ) -> None:
        """A well-connected gene should have a nonzero strategic score.

        Parameters
        ----------
        network : InteractionNetwork
            Network instance.
        """
        score = network.strategic_score("EGFR")
        assert score > 0
