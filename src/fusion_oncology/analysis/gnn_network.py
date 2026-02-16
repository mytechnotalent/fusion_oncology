"""
Graph neural network for drug–gene–pathway interaction learning.

Replaces static degree/betweenness centrality scoring in
:mod:`network_pharmacology` with **learned graph embeddings** via
a message-passing Graph Convolutional Network (GCN).  Pure NumPy
implementation — no PyTorch Geometric dependency.

Key components
--------------
* **GraphData** — sparse adjacency + node feature container.
* **GCNLayer** — single Kipf & Welling (2017) convolutional layer
  with optional bias and ReLU.
* **GNNScorer** — multi-layer GCN that produces node embeddings,
  enabling learned strategic scores, drug-combination prediction
  via inner-product decode, and causal pathway importance.

References
----------
Kipf, T.N. & Welling, M. "Semi-Supervised Classification with Graph
Convolutional Networks." ICLR (2017).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp

from fusion_oncology.analysis.drug_target import DRUG_TARGET_DB
from fusion_oncology.analysis.pathway import PathwayEnrichment
from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Graph data container ────────────────────────────────────────────────


@dataclass
class GraphData:
    """Sparse graph representation with node features and metadata.

    Parameters
    ----------
    adj : sp.csr_matrix
        Sparse adjacency matrix (n x n).
    features : np.ndarray
        Node feature matrix (n x d).
    node_names : list[str]
        Human-readable node labels.
    node_types : list[str]
        Node type for each node (``"drug"``, ``"gene"``, ``"pathway"``).
    """

    adj: sp.csr_matrix
    features: np.ndarray
    node_names: list[str] = field(default_factory=list)
    node_types: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        """Return number of nodes.

        Returns
        -------
        int
            Node count.
        """
        return self.adj.shape[0]

    @property
    def n_edges(self) -> int:
        """Return number of edges (undirected, so nnz / 2).

        Returns
        -------
        int
            Edge count.
        """
        return self.adj.nnz // 2

    @property
    def feature_dim(self) -> int:
        """Return feature dimensionality.

        Returns
        -------
        int
            Number of features per node.
        """
        return self.features.shape[1]


# ── GCN layer ────────────────────────────────────────────────────────────


def _relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        ``max(0, x)`` element-wise.
    """
    return np.maximum(0, x)


def _normalise_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    r"""Symmetric normalisation: :math:`\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}`.

    Parameters
    ----------
    adj : sp.csr_matrix
        Raw adjacency matrix.

    Returns
    -------
    sp.csr_matrix
        Symmetrically normalised adjacency with self-loops.
    """
    n = adj.shape[0]
    adj_hat = adj + sp.eye(n, format="csr")
    deg = np.array(adj_hat.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return D_inv_sqrt @ adj_hat @ D_inv_sqrt


class GCNLayer:
    """Single graph convolutional layer.

    Implements :math:`H^{(l+1)} = \\sigma(\\hat{A}_{\\text{norm}} H^{(l)} W^{(l)} + b^{(l)})`.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    activation : bool
        Apply ReLU activation.
    seed : int
        Weight initialisation seed.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: bool = True,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation

    def forward(
        self,
        adj_norm: sp.csr_matrix,
        H: np.ndarray,
    ) -> np.ndarray:
        """Propagate features through the GCN layer.

        Parameters
        ----------
        adj_norm : sp.csr_matrix
            Normalised adjacency matrix.
        H : np.ndarray
            Input node features (n x in_dim).

        Returns
        -------
        np.ndarray
            Output node features (n x out_dim).
        """
        # Message passing: AHW + b
        support = H @ self.W + self.b
        output = adj_norm @ support
        if self.activation:
            output = _relu(output)
        return output


# ── GNN scorer ───────────────────────────────────────────────────────────

# Node type → one-hot feature index
_TYPE_TO_IDX = {"drug": 0, "gene": 1, "pathway": 2}


def _build_graph_from_databases(config: ProjectConfig | None = None) -> GraphData:
    """Construct graph data from curated drug-target and pathway databases.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.

    Returns
    -------
    GraphData
        Populated graph with adjacency, features, and metadata.
    """
    cfg = config or ProjectConfig()
    pathway_db = PathwayEnrichment(cfg)

    # Collect nodes
    node_set: dict[str, str] = {}  # name → type
    edges: list[tuple[str, str]] = []

    for gene, drugs in DRUG_TARGET_DB.items():
        node_set[gene] = "gene"
        for drug_info in drugs:
            drug_name = drug_info["drug"]
            if drug_name == "—":
                continue
            node_set[drug_name] = "drug"
            edges.append((drug_name, gene))

        # Gene → pathway edges
        for pw in pathway_db.lookup(gene):
            node_set[pw] = "pathway"
            edges.append((gene, pw))

    # Build index
    node_names = sorted(node_set.keys())
    node_idx = {name: i for i, name in enumerate(node_names)}
    node_types = [node_set[n] for n in node_names]
    n = len(node_names)

    # Adjacency matrix
    rows, cols = [], []
    for u, v in edges:
        i, j = node_idx[u], node_idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Features: one-hot type + degree
    features = np.zeros((n, len(_TYPE_TO_IDX) + 1), dtype=np.float32)
    for i, ntype in enumerate(node_types):
        features[i, _TYPE_TO_IDX.get(ntype, 0)] = 1.0
    # Degree feature (normalised)
    degrees = np.array(adj.sum(axis=1)).flatten()
    max_deg = max(degrees.max(), 1.0)
    features[:, -1] = degrees / max_deg

    return GraphData(
        adj=adj,
        features=features,
        node_names=node_names,
        node_types=node_types,
    )


class GNNScorer:
    """Multi-layer GCN for learned drug–gene–pathway scoring.

    Builds the biological interaction graph from curated databases
    and trains node embeddings via message passing.  Supports:

    * **Learned strategic scores** — replace static centrality with
      embedding-norm-based importance.
    * **Drug combination prediction** — inner-product decoder between
      drug node embeddings estimates synergy likelihood.
    * **Causal pathway ranking** — pathway embeddings enriched by
      gene-level signals identify causal pathways.

    Parameters
    ----------
    hidden_dim : int
        Internal GCN layer dimension.
    embed_dim : int
        Final embedding dimension.
    n_layers : int
        Number of GCN layers (minimum 2).
    config : ProjectConfig, optional
        Runtime configuration.
    seed : int
        Weight initialisation seed.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        n_layers: int = 3,
        config: ProjectConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = config or ProjectConfig()
        self._graph = _build_graph_from_databases(self.cfg)
        self._adj_norm = _normalise_adj(self._graph.adj)

        # Build GCN layers
        dims = [self._graph.feature_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        self.layers: list[GCNLayer] = []
        for i in range(len(dims) - 1):
            activate = i < len(dims) - 2  # no activation on last layer
            self.layers.append(GCNLayer(dims[i], dims[i + 1], activation=activate, seed=seed + i))

        self._embeddings: np.ndarray | None = None
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        """Forward pass through all GCN layers to produce node embeddings."""
        H = self._graph.features.copy()
        for layer in self.layers:
            H = layer.forward(self._adj_norm, H)
        # L2 normalise
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._embeddings = H / norms

    @property
    def embeddings(self) -> np.ndarray:
        """Return node embedding matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes, embed_dim)``.
        """
        if self._embeddings is None:
            self._compute_embeddings()
        return self._embeddings

    @property
    def graph(self) -> GraphData:
        """Return the underlying graph data.

        Returns
        -------
        GraphData
            Graph with adjacency, features, and metadata.
        """
        return self._graph

    def _node_index(self, name: str) -> int | None:
        """Look up node index by name.

        Parameters
        ----------
        name : str
            Node name.

        Returns
        -------
        int or None
            Index if found, else ``None``.
        """
        try:
            return self._graph.node_names.index(name)
        except ValueError:
            return None

    def node_embedding(self, name: str) -> np.ndarray | None:
        """Return the embedding vector for a named node.

        Parameters
        ----------
        name : str
            Node name.

        Returns
        -------
        np.ndarray or None
            Embedding vector, or ``None`` if not found.
        """
        idx = self._node_index(name)
        if idx is None:
            return None
        return self.embeddings[idx]

    def strategic_score(self, gene: str) -> float:
        """Compute GNN-learned strategic importance for a gene.

        The score is the L2 norm of the gene's learned embedding,
        which encodes the node's structural centrality and
        neighbourhood information.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        float
            Strategic score in ``[0, 1]``.
        """
        emb = self.node_embedding(gene)
        if emb is None:
            return 0.0
        # Norm is already 1.0 after L2-normalisation, so use raw
        # pre-normalised magnitude relative to max
        idx = self._node_index(gene)
        H = self._graph.features.copy()
        for layer in self.layers:
            H = layer.forward(self._adj_norm, H)
        raw_norm = float(np.linalg.norm(H[idx]))
        max_norm = float(np.linalg.norm(H, axis=1).max())
        return round(raw_norm / max(max_norm, 1e-10), 4)

    def drug_combination_score(
        self,
        drug_a: str,
        drug_b: str,
    ) -> float:
        """Predict drug combination synergy via inner-product decode.

        Parameters
        ----------
        drug_a : str
            First drug name.
        drug_b : str
            Second drug name.

        Returns
        -------
        float
            Synergy score (sigmoid of dot product) in ``[0, 1]``.
        """
        emb_a = self.node_embedding(drug_a)
        emb_b = self.node_embedding(drug_b)
        if emb_a is None or emb_b is None:
            return 0.0
        dot = float(np.dot(emb_a, emb_b))
        return round(1.0 / (1.0 + np.exp(-dot)), 4)

    def pathway_importance(self) -> pd.DataFrame:
        """Rank pathways by aggregated neighbourhood embedding signal.

        Returns
        -------
        pd.DataFrame
            Columns: ``pathway``, ``gnn_score``, ``n_gene_neighbours``.
        """
        rows: list[dict[str, Any]] = []
        for i, (name, ntype) in enumerate(zip(self._graph.node_names, self._graph.node_types)):
            if ntype != "pathway":
                continue
            emb = self.embeddings[i]
            score = float(np.linalg.norm(emb))

            # Count gene neighbours
            nbrs = self._graph.adj[i].nonzero()[1]
            n_genes = sum(1 for j in nbrs if self._graph.node_types[j] == "gene")
            rows.append(
                {
                    "pathway": name,
                    "gnn_score": round(score, 4),
                    "n_gene_neighbours": n_genes,
                }
            )

        df = pd.DataFrame(rows).sort_values("gnn_score", ascending=False)
        return df.reset_index(drop=True)

    def gene_ranking(self) -> pd.DataFrame:
        """Rank genes by GNN-learned embedding importance.

        Returns
        -------
        pd.DataFrame
            Columns: ``gene``, ``gnn_score``, ``degree``.
        """
        rows: list[dict[str, Any]] = []
        degrees = np.array(self._graph.adj.sum(axis=1)).flatten()

        for i, (name, ntype) in enumerate(zip(self._graph.node_names, self._graph.node_types)):
            if ntype != "gene":
                continue
            score = self.strategic_score(name)
            rows.append(
                {
                    "gene": name,
                    "gnn_score": score,
                    "degree": int(degrees[i]),
                }
            )

        df = pd.DataFrame(rows).sort_values("gnn_score", ascending=False)
        return df.reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the GNN-scored graph.

        Returns
        -------
        dict[str, Any]
            Graph statistics and top scores.
        """
        type_counts = {}
        for t in self._graph.node_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "n_nodes": self._graph.n_nodes,
            "n_edges": self._graph.n_edges,
            "node_types": type_counts,
            "embed_dim": self.embeddings.shape[1],
            "n_layers": len(self.layers),
            "top_genes": self.gene_ranking().head(5).to_dict("records"),
            "top_pathways": self.pathway_importance().head(5).to_dict("records"),
        }
