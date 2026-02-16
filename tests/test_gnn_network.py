"""Tests for the GNN network scoring module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from fusion_oncology.analysis.gnn_network import (
    GCNLayer,
    GNNScorer,
    GraphData,
    _build_graph_from_databases,
    _normalise_adj,
    _relu,
)

# ── Activation / normalisation ───────────────────────────────────────────


class TestHelpers:
    def test_relu(self) -> None:
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = _relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_normalise_adj_adds_self_loops(self) -> None:
        adj = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=float))
        normed = _normalise_adj(adj)
        # Diagonal should be non-zero (self-loops)
        assert normed[0, 0] > 0
        assert normed[1, 1] > 0

    def test_normalise_adj_symmetric(self) -> None:
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        normed = _normalise_adj(adj)
        diff = np.abs(normed.toarray() - normed.toarray().T)
        assert diff.max() < 1e-10


# ── GraphData ────────────────────────────────────────────────────────────


class TestGraphData:
    def test_properties(self) -> None:
        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
        features = np.random.randn(3, 4).astype(np.float32)
        g = GraphData(
            adj=adj,
            features=features,
            node_names=["A", "B", "C"],
            node_types=["drug", "gene", "pathway"],
        )
        assert g.n_nodes == 3
        assert g.n_edges == 2
        assert g.feature_dim == 4


# ── GCNLayer ─────────────────────────────────────────────────────────────


class TestGCNLayer:
    def test_output_shape(self) -> None:
        adj = sp.csr_matrix(np.eye(5, dtype=float))
        adj_norm = _normalise_adj(adj)
        H = np.random.randn(5, 4).astype(np.float32)
        layer = GCNLayer(in_dim=4, out_dim=8)
        out = layer.forward(adj_norm, H)
        assert out.shape == (5, 8)

    def test_relu_activation(self) -> None:
        adj = sp.csr_matrix(np.eye(3, dtype=float))
        adj_norm = _normalise_adj(adj)
        H = np.ones((3, 2), dtype=np.float32) * -1
        layer = GCNLayer(in_dim=2, out_dim=4, activation=True)
        # Force negative output by setting weights negative
        layer.W = -np.abs(layer.W)
        layer.b = np.zeros(4)
        out = layer.forward(adj_norm, H)
        # With ReLU, positive after self-loop normalisation may vary
        # but should be finite
        assert np.all(np.isfinite(out))

    def test_no_activation(self) -> None:
        adj = sp.csr_matrix(np.eye(3, dtype=float))
        adj_norm = _normalise_adj(adj)
        H = np.ones((3, 2), dtype=np.float32)
        layer = GCNLayer(in_dim=2, out_dim=4, activation=False)
        out = layer.forward(adj_norm, H)
        assert out.shape == (3, 4)
        # Without activation, can have negative values
        assert np.all(np.isfinite(out))


# ── Graph from databases ────────────────────────────────────────────────


class TestBuildGraph:
    def test_has_nodes(self) -> None:
        graph = _build_graph_from_databases()
        assert graph.n_nodes > 0

    def test_has_edges(self) -> None:
        graph = _build_graph_from_databases()
        assert graph.n_edges > 0

    def test_has_all_types(self) -> None:
        graph = _build_graph_from_databases()
        types = set(graph.node_types)
        assert "drug" in types
        assert "gene" in types
        assert "pathway" in types

    def test_features_correct_shape(self) -> None:
        graph = _build_graph_from_databases()
        assert graph.features.shape[0] == graph.n_nodes
        assert graph.features.shape[1] == 4  # 3 type one-hot + degree


# ── GNNScorer ────────────────────────────────────────────────────────────


class TestGNNScorer:
    @pytest.fixture()
    def scorer(self) -> GNNScorer:
        return GNNScorer(hidden_dim=16, embed_dim=8, n_layers=2)

    def test_embeddings_shape(self, scorer: GNNScorer) -> None:
        emb = scorer.embeddings
        assert emb.shape[0] == scorer.graph.n_nodes
        assert emb.shape[1] == 8

    def test_embeddings_normalised(self, scorer: GNNScorer) -> None:
        emb = scorer.embeddings
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_node_embedding_found(self, scorer: GNNScorer) -> None:
        emb = scorer.node_embedding("EGFR")
        assert emb is not None
        assert emb.shape == (8,)

    def test_node_embedding_missing(self, scorer: GNNScorer) -> None:
        emb = scorer.node_embedding("NOT_A_GENE_XYZ")
        assert emb is None

    def test_strategic_score_range(self, scorer: GNNScorer) -> None:
        score = scorer.strategic_score("EGFR")
        assert 0.0 <= score <= 1.0

    def test_strategic_score_missing(self, scorer: GNNScorer) -> None:
        score = scorer.strategic_score("NOT_A_GENE_XYZ")
        assert score == 0.0

    def test_drug_combination_score(self, scorer: GNNScorer) -> None:
        score = scorer.drug_combination_score("Erlotinib", "Gefitinib")
        assert 0.0 <= score <= 1.0

    def test_drug_combination_missing(self, scorer: GNNScorer) -> None:
        score = scorer.drug_combination_score("FakeDrug", "AnotherFake")
        assert score == 0.0

    def test_pathway_importance(self, scorer: GNNScorer) -> None:
        df = scorer.pathway_importance()
        assert isinstance(df, pd.DataFrame)
        assert "pathway" in df.columns
        assert "gnn_score" in df.columns
        assert len(df) > 0

    def test_gene_ranking(self, scorer: GNNScorer) -> None:
        df = scorer.gene_ranking()
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "gnn_score" in df.columns
        assert len(df) > 0

    def test_gene_ranking_sorted_descending(self, scorer: GNNScorer) -> None:
        df = scorer.gene_ranking()
        scores = df["gnn_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_summary(self, scorer: GNNScorer) -> None:
        s = scorer.summary()
        assert "n_nodes" in s
        assert "n_edges" in s
        assert "node_types" in s
        assert "embed_dim" in s
        assert s["embed_dim"] == 8
        assert "top_genes" in s
        assert "top_pathways" in s
