"""
Network pharmacology and drug-target-pathway interaction modelling.

Builds a directed interaction graph connecting drugs → targets →
pathways and computes centrality metrics to identify the most
strategically valuable intervention points in the cancer signalling
network.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.analysis.drug_target import DRUG_TARGET_DB
from fusion_oncology.analysis.pathway import PathwayEnrichment
from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


class InteractionNetwork:
    """Drug–target–pathway interaction graph.

    Builds a tripartite graph with three node types (drugs, genes,
    pathways) and edges representing known interactions.  Supports
    centrality analysis, shortest-path queries, and polypharmacology
    scoring.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def _add_drug_gene_edge(
        self,
        drug_name: str,
        gene: str,
        drug_info: dict[str, str],
    ) -> None:
        """Register a single drug-to-gene edge with metadata.

        Parameters
        ----------
        drug_name : str
            Drug identifier.
        gene : str
            HGNC gene symbol.
        drug_info : dict[str, str]
            Metadata containing ``status`` and ``indication``.
        """
        self._adjacency[drug_name].add(gene)
        self._adjacency[gene].add(drug_name)
        self._edge_metadata[(drug_name, gene)] = {
            "type": "targets",
            "status": drug_info["status"],
            "indication": drug_info["indication"],
        }

    def _add_drug_gene_edges(self) -> None:
        """Create edges from every curated drug to its gene target."""
        for gene, drugs in DRUG_TARGET_DB.items():
            self._node_types[gene] = "gene"
            for drug_info in drugs:
                drug_name = drug_info["drug"]
                if drug_name == "—":
                    continue
                self._node_types[drug_name] = "drug"
                self._add_drug_gene_edge(drug_name, gene, drug_info)

    def _add_pathway_edge(self, gene: str, pw: str) -> None:
        """Register a single gene-to-pathway membership edge.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        pw : str
            Pathway name.
        """
        self._node_types[pw] = "pathway"
        self._adjacency[gene].add(pw)
        self._adjacency[pw].add(gene)
        self._edge_metadata[(gene, pw)] = {"type": "member_of"}

    def _add_gene_pathway_edges(self) -> None:
        """Create edges from every gene node to its annotated pathways."""
        for gene in list(self._node_types.keys()):
            if self._node_types.get(gene) != "gene":
                continue
            for pw in self._pathway_db.lookup(gene):
                self._add_pathway_edge(gene, pw)

    def _count_node_types(self) -> tuple[int, int, int]:
        """Count drug, gene, and pathway nodes.

        Returns
        -------
        tuple[int, int, int]
            ``(n_drugs, n_genes, n_pathways)``.
        """
        vals = list(self._node_types.values())
        return vals.count("drug"), vals.count("gene"), vals.count("pathway")

    def _log_network_stats(self) -> None:
        """Log summary counts of nodes and edges in the network."""
        n_drugs, n_genes, n_pws = self._count_node_types()
        n_edges = sum(len(v) for v in self._adjacency.values()) // 2
        n_total = len(self._node_types)
        msg = "Network: %d nodes (%d drugs, %d genes, %d pw), %d edges"
        logger.info(msg, n_total, n_drugs, n_genes, n_pws, n_edges)

    def _build_network(self) -> None:
        """Populate the graph from curated drug-target and pathway databases.

        Creates edges: drug → gene (targeting), gene → pathway (membership).
        """
        self._add_drug_gene_edges()
        self._add_gene_pathway_edges()
        self._log_network_stats()

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the network and populate from curated databases.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._node_types: dict[str, str] = {}
        self._edge_metadata: dict[tuple[str, str], dict[str, str]] = {}
        self._pathway_db = PathwayEnrichment(self.cfg)
        self._build_network()

    def _collect_unique_edges(self) -> list[tuple[str, str]]:
        """Deduplicate undirected edges from the adjacency map.

        Returns
        -------
        list[tuple[str, str]]
            Deduplicated edge list.
        """
        edges: set[tuple[str, str]] = set()
        for node, neighbours in self._adjacency.items():
            edges.update(tuple(sorted((node, nbr))) for nbr in neighbours)
        return list(edges)

    @property
    def nodes(self) -> list[str]:
        """Return all node names in the network.

        Returns
        -------
        list[str]
            Sorted list of node identifiers.
        """
        return sorted(self._node_types.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Return all undirected edges.

        Returns
        -------
        list[tuple[str, str]]
            Deduplicated edge list.
        """
        return self._collect_unique_edges()

    def _degree_row(
        self,
        node: str,
        ntype: str,
        n: int,
    ) -> dict[str, Any]:
        """Build a single degree-centrality row dict.

        Parameters
        ----------
        node : str
            Node identifier.
        ntype : str
            Node type.
        n : int
            Total number of nodes in the network.

        Returns
        -------
        dict[str, Any]
            Keys ``Node``, ``Type``, ``Degree``, ``Centrality``.
        """
        degree = len(self._adjacency.get(node, set()))
        centrality = degree / (n - 1) if n > 1 else 0
        return {
            "Node": node,
            "Type": ntype,
            "Degree": degree,
            "Centrality": round(centrality, 6),
        }

    def _compute_degree_rows(self) -> list[dict[str, Any]]:
        """Build per-node degree centrality records.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has keys ``Node``, ``Type``, ``Degree``, ``Centrality``.
        """
        n = len(self._node_types)
        return [
            self._degree_row(node, ntype, n) for node, ntype in self._node_types.items()
        ]

    def degree_centrality(self) -> pd.DataFrame:
        """Compute degree centrality for all nodes.

        Degree centrality = (number of edges) / (total nodes − 1).
        Higher values identify hub nodes that connect many entities.

        Returns
        -------
        pd.DataFrame
            Columns: ``Node``, ``Type``, ``Degree``, ``Centrality``.
            Sorted by centrality descending.
        """
        rows = self._compute_degree_rows()
        df = pd.DataFrame(rows).sort_values("Centrality", ascending=False)
        return df.reset_index(drop=True)

    def _bfs_expand(
        self,
        node: str,
        dist: dict[str, int],
        paths: dict[str, int],
        queue: list[str],
    ) -> None:
        """Expand one BFS level, updating distances and path counts.

        Parameters
        ----------
        node : str
            Current node being expanded.
        dist : dict[str, int]
            Shortest-distance map from source.
        paths : dict[str, int]
            Shortest-path count map from source.
        queue : list[str]
            BFS frontier queue.
        """
        for nbr in self._adjacency.get(node, set()):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                paths[nbr] = 0
                queue.append(nbr)
            if dist[nbr] == dist[node] + 1:
                paths[nbr] += paths[node]

    def _bfs_shortest_paths(
        self,
        source: str,
    ) -> tuple[dict[str, int], dict[str, int], list[str]]:
        """Run BFS from *source* to compute shortest-path counts.

        Parameters
        ----------
        source : str
            Starting node.

        Returns
        -------
        tuple[dict[str, int], dict[str, int], list[str]]
            ``(dist, paths, order)`` — distance map, path-count map,
            and BFS traversal order.
        """
        dist, paths = {source: 0}, {source: 1}
        queue, order = [source], []
        while queue:
            node = queue.pop(0)
            order.append(node)
            self._bfs_expand(node, dist, paths, queue)
        return dist, paths, order

    def _accumulate_dependencies(
        self,
        dist: dict[str, int],
        paths: dict[str, int],
        order: list[str],
        all_nodes: list[str],
    ) -> dict[str, float]:
        """Back-propagate dependency fractions for betweenness.

        Parameters
        ----------
        dist : dict[str, int]
            Shortest-distance map from a single BFS source.
        paths : dict[str, int]
            Shortest-path count map.
        order : list[str]
            BFS traversal order.
        all_nodes : list[str]
            All network nodes.

        Returns
        -------
        dict[str, float]
            Per-node dependency delta.
        """
        delta: dict[str, float] = {n: 0.0 for n in all_nodes}
        for node in reversed(order):
            for nbr in self._adjacency.get(node, set()):
                if dist.get(nbr, -1) == dist.get(node, -1) + 1:
                    frac = paths.get(node, 0) / max(paths.get(nbr, 1), 1)
                    delta[node] += frac * (1 + delta[nbr])
        return delta

    def _update_bc(
        self,
        bc: dict[str, float],
        delta: dict[str, float],
        source: str,
        all_nodes: list[str],
    ) -> None:
        """Add dependency deltas into running betweenness totals.

        Parameters
        ----------
        bc : dict[str, float]
            Accumulated betweenness scores (mutated in-place).
        delta : dict[str, float]
            Per-node dependency from one BFS source.
        source : str
            BFS source node (excluded from accumulation).
        all_nodes : list[str]
            All network nodes.
        """
        for node in all_nodes:
            if node != source:
                bc[node] += delta[node]

    def _compute_raw_betweenness(self) -> dict[str, float]:
        """Run BFS from every node and accumulate betweenness scores.

        Returns
        -------
        dict[str, float]
            Un-normalised betweenness centrality per node.
        """
        all_nodes = list(self._node_types.keys())
        bc: dict[str, float] = {n: 0.0 for n in all_nodes}
        for source in all_nodes:
            dist, paths, order = self._bfs_shortest_paths(source)
            delta = self._accumulate_dependencies(dist, paths, order, all_nodes)
            self._update_bc(bc, delta, source, all_nodes)
        return bc

    def _normalize_betweenness(self, bc: dict[str, float]) -> list[dict[str, Any]]:
        """Normalise raw betweenness and build row dicts.

        Parameters
        ----------
        bc : dict[str, float]
            Un-normalised betweenness scores.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has keys ``Node``, ``Type``, ``Betweenness``.
        """
        n = len(bc)
        norm = max(1, (n - 1) * (n - 2))
        return [
            {
                "Node": node,
                "Type": self._node_types[node],
                "Betweenness": round(val / norm, 6),
            }
            for node, val in bc.items()
        ]

    def betweenness_centrality(self) -> pd.DataFrame:
        """Compute approximate betweenness centrality via BFS.

        Betweenness centrality measures how often a node lies on
        shortest paths between other nodes.  Gene targets with high
        betweenness are strategic intervention points.

        Returns
        -------
        pd.DataFrame
            Columns: ``Node``, ``Type``, ``Betweenness``.
            Sorted descending.
        """
        bc = self._compute_raw_betweenness()
        rows = self._normalize_betweenness(bc)
        df = pd.DataFrame(rows).sort_values("Betweenness", ascending=False)
        return df.reset_index(drop=True)

    def _get_neighbours_by_type(self, node: str, node_type: str) -> list[str]:
        """Return neighbours of *node* filtered to a specific type.

        Parameters
        ----------
        node : str
            Node identifier.
        node_type : str
            Desired node type (``"drug"``, ``"gene"``, or ``"pathway"``).

        Returns
        -------
        list[str]
            Matching neighbour identifiers.
        """
        return [
            n
            for n in self._adjacency.get(node, set())
            if self._node_types.get(n) == node_type
        ]

    def _get_target_pathways(self, targets: list[str]) -> set[str]:
        """Collect all pathway neighbours of a list of gene targets.

        Parameters
        ----------
        targets : list[str]
            Gene symbols.

        Returns
        -------
        set[str]
            Union of pathway neighbours.
        """
        pathways: set[str] = set()
        for t in targets:
            pathways.update(self._get_neighbours_by_type(t, "pathway"))
        return pathways

    def polypharmacology_score(self, drug: str) -> dict[str, Any]:
        """Assess how many targets and pathways a drug affects.

        Polypharmacology — hitting multiple targets — can be advantageous
        for overcoming resistance but increases toxicity risk.

        Parameters
        ----------
        drug : str
            Drug name.

        Returns
        -------
        dict[str, Any]
            Keys: ``drug``, ``targets`` (list), ``pathways`` (set),
            ``n_targets``, ``n_pathways``, ``score``.
        """
        targets = self._get_neighbours_by_type(drug, "gene")
        pathways = self._get_target_pathways(targets)
        score = min(1.0, (len(targets) * 0.3 + len(pathways) * 0.1))
        return {
            "drug": drug,
            "targets": targets,
            "pathways": pathways,
            "n_targets": len(targets),
            "n_pathways": len(pathways),
            "score": round(score, 4),
        }

    def _bfs_expand_distance(
        self,
        node: str,
        dist: int,
        visited: dict[str, int],
        queue: list[tuple[str, int]],
    ) -> None:
        """Expand one BFS level for distance-bounded traversal.

        Parameters
        ----------
        node : str
            Current node.
        dist : int
            Distance of *node* from the BFS source.
        visited : dict[str, int]
            Visited map (mutated in-place).
        queue : list[tuple[str, int]]
            BFS frontier (mutated in-place).
        """
        for nbr in self._adjacency.get(node, set()):
            if nbr not in visited:
                visited[nbr] = dist + 1
                queue.append((nbr, dist + 1))

    def _bfs_within_distance(self, start: str, max_distance: int) -> dict[str, int]:
        """Run BFS from *start* up to *max_distance* hops.

        Parameters
        ----------
        start : str
            Starting node.
        max_distance : int
            Maximum number of hops.

        Returns
        -------
        dict[str, int]
            Mapping of reachable node to distance.
        """
        visited: dict[str, int] = {start: 0}
        queue: list[tuple[str, int]] = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if dist < max_distance:
                self._bfs_expand_distance(node, dist, visited, queue)
        return visited

    def _build_candidate(
        self,
        node: str,
        dist: int,
        gene_pathways: set[str],
    ) -> dict[str, Any] | None:
        """Build a combination-target candidate record if druggable.

        Parameters
        ----------
        node : str
            Candidate gene.
        dist : int
            Graph distance from primary target.
        gene_pathways : set[str]
            Pathways of the primary target.

        Returns
        -------
        dict[str, Any] or None
            Candidate dict, or ``None`` if no drugs target *node*.
        """
        node_drugs = self._get_neighbours_by_type(node, "drug")
        if not node_drugs:
            return None
        node_pws = set(self._get_neighbours_by_type(node, "pathway"))
        return {
            "target": node,
            "distance": dist,
            "shared_pathways": list(gene_pathways & node_pws),
            "drugs_available": node_drugs,
        }

    def _is_combo_candidate(
        self,
        node: str,
        gene: str,
        dist: int,
        max_dist: int,
    ) -> bool:
        """Check whether a node qualifies as a combination candidate.

        Parameters
        ----------
        node : str
            Candidate node.
        gene : str
            Primary target gene (excluded).
        dist : int
            Graph distance from *gene*.
        max_dist : int
            Maximum allowed distance.

        Returns
        -------
        bool
            ``True`` if *node* is a different gene within range.
        """
        if node == gene or dist > max_dist:
            return False
        return self._node_types.get(node) == "gene"

    def _filter_druggable_candidates(
        self,
        visited: dict[str, int],
        gene: str,
        max_distance: int,
        gene_pathways: set[str],
    ) -> list[dict[str, Any]]:
        """Filter BFS results to druggable gene targets.

        Parameters
        ----------
        visited : dict[str, int]
            BFS distance map.
        gene : str
            Primary target gene (excluded).
        max_distance : int
            Maximum allowed distance.
        gene_pathways : set[str]
            Pathways of the primary target.

        Returns
        -------
        list[dict[str, Any]]
            Candidate records for nodes with available drugs.
        """
        candidates: list[dict[str, Any]] = []
        for node, dist in visited.items():
            if not self._is_combo_candidate(node, gene, dist, max_distance):
                continue
            candidate = self._build_candidate(node, dist, gene_pathways)
            if candidate:
                candidates.append(candidate)
        return candidates

    def find_combination_targets(
        self,
        gene: str,
        max_distance: int = 2,
    ) -> list[dict[str, Any]]:
        """Find druggable targets within *max_distance* hops of *gene*.

        Useful for identifying rational combination therapy partners
        in adjacent signalling pathway nodes.

        Parameters
        ----------
        gene : str
            HGNC gene symbol of the primary target.
        max_distance : int
            Maximum graph distance to search.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has keys ``target``, ``distance``, ``shared_pathways``,
            ``drugs_available``.
        """
        visited = self._bfs_within_distance(gene, max_distance)
        gene_pathways = set(self._get_neighbours_by_type(gene, "pathway"))
        candidates = self._filter_druggable_candidates(
            visited, gene, max_distance, gene_pathways
        )
        return sorted(candidates, key=lambda x: x["distance"])

    def _compute_strategic_components(self, gene: str) -> tuple[float, float, float]:
        """Compute drug, pathway, and degree sub-scores for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        tuple[float, float, float]
            ``(drug_score, pathway_score, degree_score)`` each in ``[0, 1]``.
        """
        neighbours = self._adjacency.get(gene, set())
        n_drugs = len(self._get_neighbours_by_type(gene, "drug"))
        n_pathways = len(self._get_neighbours_by_type(gene, "pathway"))
        drug_score = min(1.0, n_drugs * 0.2)
        pathway_score = min(1.0, n_pathways * 0.15)
        degree_score = min(1.0, len(neighbours) * 0.1)
        return drug_score, pathway_score, degree_score

    def strategic_score(self, gene: str) -> float:
        """Compute a strategic intervention score for *gene*.

        Combines degree centrality, pathway breadth, and drug
        availability into a single score indicating how strategically
        valuable targeting this gene would be.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        float
            Score in ``[0, 1]`` range.
        """
        if gene not in self._node_types:
            return 0.0
        drug_s, pathway_s, degree_s = self._compute_strategic_components(gene)
        return round(0.4 * drug_s + 0.3 * pathway_s + 0.3 * degree_s, 4)
