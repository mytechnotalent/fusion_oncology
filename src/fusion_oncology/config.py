"""
Central configuration for the Fusion Oncology Suite.

All tuneable parameters live here so that CLI flags, environment
variables, and tests can override them in a single place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectConfig:
    """Runtime configuration for the Fusion Oncology Suite.

    Every field can be overridden at construction or via CLI flags.
    Directories (``cache_dir``, ``output_dir``) are created automatically
    on initialisation.

    Attributes
    ----------
    data_url : str
        URL of the TCGA Pan-Cancer RNA-Seq ZIP archive on the UCI ML Repository.
    cache_dir : Path
        Local directory for caching downloaded artefacts.  Defaults to
        ``~/.cache/fusion_oncology`` or the ``FUSION_CACHE`` env-var.
    model_path : str
        Hugging Face model identifier for the DNABERT-2 transformer.
    dnabert_revision : str
        Pinned Git revision (commit SHA) on Hugging Face Hub.  Prevents
        auto-downloading new (potentially untrusted) remote code files.
    max_seq_len : int
        Maximum token length for DNABERT-2 input sequences.
    entrez_email : str
        Email address supplied to NCBI Entrez for sequence lookups.
    entrez_retmax : int
        Maximum number of Entrez search results to return.
    top_k_genes : int
        Number of top genes to carry forward from XGBoost importance.
    fuzz_iterations : int
        Number of random single-nucleotide mutations per gene for
        instability scoring.
    xgb_n_estimators : int
        Number of boosting rounds for the XGBoost classifier.
    xgb_max_depth : int
        Maximum tree depth for XGBoost.
    xgb_learning_rate : float
        Gradient-descent step size for XGBoost boosting.
    xgb_min_child_weight : int
        Minimum sum of instance weight in a child node.
    xgb_gamma : float
        Minimum loss reduction required to make a further partition.
    xgb_reg_alpha : float
        L1 regularisation term on weights.
    xgb_reg_lambda : float
        L2 regularisation term on weights.
    pathway_db : str
        Pathway database selector (``"kegg"`` or ``"reactome"``).
    enrichment_pval : float
        P-value threshold for pathway enrichment significance.
    survival_time_col : str
        Column name for overall survival time in clinical data.
    survival_event_col : str
        Column name for the event indicator in clinical data.
    drugbank_api : str
        Base URL for the DrugBank search API.
    opentargets_api : str
        Base URL for the OpenTargets GraphQL endpoint.
    civic_api : str
        Base URL for the CIViC GraphQL endpoint.
    clinicaltrials_api : str
        Base URL for the ClinicalTrials.gov REST v2 endpoint.
    mhc_alleles : list[str]
        MHC-I alleles for neoantigen binding prediction.
    neoantigen_binding_threshold : float
        Minimum binding score to call a neoantigen candidate.
    crispr_guides_per_gene : int
        Maximum CRISPR guides to retain per target gene.
    crispr_min_score : float
        Minimum composite score for CRISPR guide inclusion.
    simulation_days : int
        Number of days in digital-twin tumour simulation.
    tumour_growth_rate : float
        Gompertzian growth rate constant (day⁻¹).
    tumour_carrying_capacity : float
        Maximum tumour size for Gompertz saturation.
    output_dir : Path
        Directory for writing results, figures, and reports.
    report_format : str
        Output report format (``"html"``, ``"pdf"``, or ``"csv"``).
    figure_dpi : int
        Resolution (dots per inch) for saved matplotlib figures.
    """

    # ── Data source ──────────────────────────────────────────────────────
    data_url: str = (
        "https://archive.ics.uci.edu/static/public/401/" "gene+expression+cancer+rna+seq.zip"
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FUSION_CACHE", Path.home() / ".cache" / "fusion_oncology")
        )
    )

    # ── DNABERT model ────────────────────────────────────────────────────
    model_path: str = "zhihan1996/DNABERT-2-117M"
    dnabert_revision: str = "7bce263b15377fc15361f52cfab88f8b586abda0"
    max_seq_len: int = 512

    # ── NCBI Entrez ──────────────────────────────────────────────────────
    entrez_email: str = "researcher@example.com"
    entrez_retmax: int = 1

    # ── Analysis hyper-parameters ────────────────────────────────────────
    top_k_genes: int = 5
    fuzz_iterations: int = 20
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_min_child_weight: int = 3
    xgb_gamma: float = 0.1
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.5

    # ── Pathway enrichment ───────────────────────────────────────────────
    pathway_db: str = "kegg"  # "kegg" | "reactome"
    enrichment_pval: float = 0.05

    # ── Survival analysis ────────────────────────────────────────────────
    survival_time_col: str = "OS_MONTHS"
    survival_event_col: str = "OS_STATUS"

    # ── Drug-target mapping ──────────────────────────────────────────────
    drugbank_api: str = "https://go.drugbank.com/unearth/q?searcher=drugs&query="

    # ── Clinical evidence APIs ───────────────────────────────────────────
    opentargets_api: str = "https://api.platform.opentargets.org/api/v4/graphql"
    civic_api: str = "https://civicdb.org/api/graphql"
    clinicaltrials_api: str = "https://clinicaltrials.gov/api/v2/studies"

    # ── Neoantigen prediction ────────────────────────────────────────────
    mhc_alleles: list[str] = field(default_factory=lambda: ["HLA-A*02:01"])
    neoantigen_binding_threshold: float = 0.5

    # ── CRISPR guide design ──────────────────────────────────────────────
    crispr_guides_per_gene: int = 5
    crispr_min_score: float = 0.25

    # ── Digital twin simulation ──────────────────────────────────────────
    simulation_days: int = 365
    tumour_growth_rate: float = 0.02
    tumour_carrying_capacity: float = 1e12

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("results"))
    report_format: str = "html"  # "html" | "pdf" | "csv"
    figure_dpi: int = 300

    def __post_init__(self) -> None:
        """Coerce path fields to ``Path`` objects and create directories.

        Called automatically by the dataclass machinery after
        ``__init__`` completes.  Ensures ``cache_dir`` and ``output_dir``
        exist on disk.
        """
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
