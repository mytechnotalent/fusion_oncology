"""
Proof-of-function runner for Fusion Oncology.

Exercises every major subsystem in the pipeline and prints validated
output for inclusion in ``PROOF.md``.  Each proof function imports the
relevant module, constructs realistic inputs, and reports key metrics
so that the results can be independently verified.

Available proofs (pass as CLI argument):

    xgboost     XGBoost feature engineering, CV, gene ranking
    twin        Digital twin tumour simulation & regimen comparison
    resistance  Resistance mechanism database & risk scoring
    pathway     Cancer pathway enrichment analysis
    drug        Drug-target database lookups
    sl          Synthetic lethality partner detection
    network     Network pharmacology hub & strategic scoring
    cdx         Companion diagnostic end-to-end report
    neoantigen  Neoantigen prediction & MHC-I scoring
    crispr      CRISPR sgRNA guide design
    all         Run every proof sequentially

Usage
-----
.. code-block:: bash

    python proof_runner.py xgboost
    python proof_runner.py all
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── XGBoost ──────────────────────────────────────────────────────────────


def proof_xgboost() -> None:
    """Validate the XGBoost drug-sensitivity engine.

    Demonstrates:
    1. Automated feature engineering (50 → 60 columns via row-level
       statistical moments).
    2. Stratified cross-validation with precision, recall, F1, and
       ROC-AUC metrics.
    3. Gene importance ranking (top-K genes by XGBoost gain).
    4. Rare-class merging to prevent stratification failures.

    All data is synthetic (random normal, four cancer labels) so that
    the proof runs without external datasets.  Metric magnitudes
    reflect the random baseline, not clinical performance.
    """
    from fusion_oncology.config import ProjectConfig
    from fusion_oncology.models.xgboost_engine import XGBoostEngine

    cfg = ProjectConfig(
        xgb_n_estimators=50,
        xgb_max_depth=3,
        top_k_genes=5,
        enable_feature_engineering=False,
    )

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((200, 50)),
        columns=[f"GENE{i}" for i in range(50)],
    )
    y = pd.Series(rng.choice(["BRCA", "LUAD", "COAD", "GBM"], 200))

    # ── feature engineering ──────────────────────────────────────────
    X_eng = XGBoostEngine.engineer_features(X)
    print("=== XGBoost Feature Engineering ===")
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_eng.shape[1]}")
    print(f"Added columns: {[c for c in X_eng.columns if c.startswith('_')]}")

    # ── fit & cross-validate ─────────────────────────────────────────
    engine = XGBoostEngine(cfg)
    engine.fit(X, y)
    cv = engine.cross_validate(X, y)
    print("\n=== XGBoost CV Metrics (5-fold) ===")
    for k, v in cv.items():
        print(f"  {k}: {v:.4f}")

    # ── gene importance ──────────────────────────────────────────────
    top = engine.top_genes()
    print("\n=== Top-5 Genes ===")
    for g, imp in top.items():
        print(f"  {g}: {imp:.6f}")

    # ── rare-class merging ───────────────────────────────────────────
    y2 = pd.Series(["A"] * 100 + ["B"] * 50 + ["C"] * 5 + ["D"] * 3)
    y2m = XGBoostEngine.merge_rare_classes(y2, min_size=10)
    print("\n=== Class Merging ===")
    print(f"Before: {y2.value_counts().to_dict()}")
    print(f"After:  {y2m.value_counts().to_dict()}")


# ── Digital Twin ─────────────────────────────────────────────────────────


def proof_digital_twin() -> None:
    """Validate the ODE-based digital twin tumour simulation.

    Runs a 365-day simulation of Osimertinib monotherapy on a
    Gompertzian tumour with 1 % pre-existing resistance.  Reports:

    * Initial and final tumour burden.
    * Best response day and percentage (RECIST-style).
    * Full trajectory shape and column layout.
    * First / last five rows for manual inspection.
    * Comparison across three regimens (mono, combo, high-dose).
    """
    from fusion_oncology.models.digital_twin import (
        DigitalTwin,
        DrugRegimen,
        SimulationConfig,
    )

    sim_cfg = SimulationConfig(
        initial_tumour_size=1e9,
        simulation_days=365,
        growth_rate=0.02,
    )
    twin = DigitalTwin(sim_config=sim_cfg)
    twin.add_regimen(
        DrugRegimen(
            name="Osimertinib",
            efficacy=0.15,
            resistance_rate=0.001,
            start_day=0,
            duration_days=180,
            cycle_on=21,
            cycle_off=7,
        ),
    )

    df = twin.simulate()

    print("=== Digital Twin Simulation ===")
    print(f"Simulation days: {sim_cfg.simulation_days}")
    print(f"Initial tumour: {sim_cfg.initial_tumour_size:.2e}")
    print(f"Final tumour:   {df['total'].iloc[-1]:.2e}")
    print(f"Best response:  {twin.best_response()}")
    print(f"RECIST:         {twin.recist_response()}")
    print(f"Trajectory shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")

    # ── regimen comparison ───────────────────────────────────────────
    comparison = twin.compare_regimens(
        {
            "Osimertinib_mono": [
                DrugRegimen(name="Osimertinib", efficacy=0.15),
            ],
            "Osimertinib+Capmatinib": [
                DrugRegimen(name="Osimertinib", efficacy=0.15),
                DrugRegimen(name="Capmatinib", efficacy=0.10),
            ],
            "High_dose": [
                DrugRegimen(name="Osimertinib", efficacy=0.25),
            ],
        },
    )
    print(f"\n=== Regimen Comparison ===\n{comparison.to_string()}")


# ── Resistance ───────────────────────────────────────────────────────────


def proof_resistance() -> None:
    """Validate the drug-resistance mechanism database and predictor.

    Prints the total number of genes covered, the total mechanism
    count, then queries four clinically important genes (EGFR, BRAF,
    KRAS, ALK) for their resistance risk scores and individual
    mechanism details — including the triggering drug(s) and the
    recommended clinical strategy.
    """
    from fusion_oncology.analysis.resistance import (
        RESISTANCE_DB,
        ResistancePredictor,
    )

    print("=== Resistance Mechanism Database ===")
    print(f"Genes with resistance data: {len(RESISTANCE_DB)}")
    print(f"Genes: {sorted(RESISTANCE_DB.keys())}")
    total = sum(len(v) for v in RESISTANCE_DB.values())
    print(f"Total mechanisms catalogued: {total}")

    pred = ResistancePredictor()
    for gene in ["EGFR", "BRAF", "KRAS", "ALK"]:
        risk = pred.resistance_risk_score(gene)
        mechs = pred.predict(gene)
        print(f"\n  {gene}: risk={risk:.2f}, mechanisms={len(mechs)}")
        for m in mechs[:2]:
            print(f"    - {m['drug']}: {m['mechanism']}")
            print(f"      Strategy: {m['strategy']}")


# ── Pathway Enrichment ───────────────────────────────────────────────────


def proof_pathway() -> None:
    """Validate cancer pathway enrichment analysis.

    Lists the ten curated cancer signalling pathways with their gene
    counts, then runs enrichment for a test panel of eight driver
    genes (EGFR, BRAF, KRAS, TP53, BRCA1, PTEN, CDK4, VEGFA) and
    reports hits per pathway.
    """
    from fusion_oncology.analysis.pathway import (
        CANCER_PATHWAYS,
        PathwayEnrichment,
    )

    print("=== Pathway Database ===")
    print(f"Pathways: {len(CANCER_PATHWAYS)}")
    for pw, genes in CANCER_PATHWAYS.items():
        print(f"  {pw}: {len(genes)} genes")

    pe = PathwayEnrichment()
    test_genes = [
        "EGFR",
        "BRAF",
        "KRAS",
        "TP53",
        "BRCA1",
        "PTEN",
        "CDK4",
        "VEGFA",
    ]
    summary = pe.enrichment_summary(test_genes)
    print(f"\nEnrichment for {test_genes}:")
    for pw, info in summary.items():
        print(f"  {pw}: {info['count']} hits -> {info['genes']}")


# ── Drug-Target Mapping ──────────────────────────────────────────────────


def proof_drug_target() -> None:
    """Validate the FDA-approved drug-target database.

    Reports the total number of targetable genes, drug-target
    mappings, and FDA-approved entries.  Then performs lookups for
    EGFR, BRAF, ALK, and BRCA1 — printing each matched drug, its
    approval status, and indication(s).
    """
    from fusion_oncology.analysis.drug_target import (
        DRUG_TARGET_DB,
        DrugTargetMapper,
    )

    print("=== Drug-Target Database ===")
    print(f"Targetable genes: {len(DRUG_TARGET_DB)}")
    total_drugs = sum(len(v) for v in DRUG_TARGET_DB.values())
    print(f"Total drug-target mappings: {total_drugs}")
    approved = sum(
        1 for drugs in DRUG_TARGET_DB.values() for d in drugs if d["status"] == "Approved"
    )
    print(f"FDA-Approved entries: {approved}")

    mapper = DrugTargetMapper()
    for gene in ["EGFR", "BRAF", "ALK", "BRCA1"]:
        drugs = mapper.lookup(gene)
        print(f"\n  {gene} ({len(drugs)} drugs):")
        for d in drugs:
            print(f"    {d['drug']} - {d['status']} - {d['indication']}")


# ── Synthetic Lethality ──────────────────────────────────────────────────


def proof_synthetic_lethality() -> None:
    """Validate synthetic lethality partner detection.

    Reports the total number of curated SL pairs from SynLethDB and
    CRISPR screen literature, then queries four commonly lost tumour
    suppressors / oncogenes (BRCA1, TP53, KRAS, PTEN) for their
    known synthetic lethal partners.
    """
    from fusion_oncology.analysis.synthetic_lethality import (
        KNOWN_SL_PAIRS,
        SyntheticLethalityDetector,
    )

    print("=== Synthetic Lethality Database ===")
    print(f"Curated SL pairs: {len(KNOWN_SL_PAIRS)}")

    det = SyntheticLethalityDetector()
    for gene in ["BRCA1", "TP53", "KRAS", "PTEN"]:
        partners = det.known_partners(gene)
        pnames = [p["partner"] for p in partners]
        print(f"  {gene} -> {pnames}")


# ── Network Pharmacology ─────────────────────────────────────────────────


def proof_network() -> None:
    """Validate the network pharmacology tripartite graph.

    Builds the drug → gene → pathway interaction network and reports:

    * Total node and edge counts.
    * Top-10 hub nodes by degree centrality.
    * Strategic intervention scores for EGFR, BRAF, and KRAS,
      combining drug availability, pathway breadth, and degree
      centrality into a single ``[0, 1]`` score.
    """
    from fusion_oncology.analysis.network_pharmacology import InteractionNetwork

    net = InteractionNetwork()

    print("=== Network Pharmacology ===")
    print(f"Total nodes: {len(net.nodes)}")
    print(f"Total edges: {len(net.edges)}")

    dc = net.degree_centrality()
    print("\nTop-10 hub nodes:")
    print(dc.head(10).to_string())

    for gene in ["EGFR", "BRAF", "KRAS"]:
        score = net.strategic_score(gene)
        print(f"\n  Strategic score for {gene}: {score:.4f}")


# ── Companion Diagnostics ────────────────────────────────────────────────


def proof_companion_dx() -> None:
    """Validate the end-to-end companion diagnostic pipeline.

    Constructs a realistic LUAD patient profile with three mutations
    (EGFR L858R, TP53 R273H, KRAS G12C), expression values, and
    copy-number alterations.  Runs the full ``analyse()`` workflow
    and reports:

    * Patient metadata and tumour mutational burden.
    * Number of matched drugs with gene and approval status.
    * Top-5 ranked treatment recommendations with confidence scores.
    """
    from fusion_oncology.models.companion_dx import (
        CompanionDiagnostic,
        PatientProfile,
    )

    patient = PatientProfile(
        patient_id="TEST_001",
        mutations=[
            {"gene": "EGFR", "variant": "L858R", "type": "missense"},
            {"gene": "TP53", "variant": "R273H", "type": "missense"},
            {"gene": "KRAS", "variant": "G12C", "type": "missense"},
        ],
        expression={"EGFR": 8.5, "TP53": 2.1, "KRAS": 6.3, "BRAF": 4.0},
        cna={"EGFR": 1.2, "MYC": 0.8, "PTEN": -1.5},
        cancer_type="LUAD",
    )

    dx = CompanionDiagnostic()
    result = dx.analyse(patient)

    print("=== Companion Diagnostic ===")
    print(f"Patient: {patient.patient_id}")
    print(f"Cancer type: {patient.cancer_type}")
    print(f"TMB: {patient.tumour_mutational_burden}")
    print(f"Mutated genes: {patient.mutated_genes}")
    print(f"Amplified: {patient.amplified_genes()}")
    print(f"Deleted: {patient.deleted_genes()}")

    print(f"\nResult keys: {list(result.keys())}")
    matches = result.get("drug_matches", [])
    print(f"Matched drugs: {len(matches)}")
    for d in matches[:5]:
        print(f"  {d['gene']}: {d['drug']} ({d['status']})")

    plan = result.get("treatment_plan", [])
    print(f"\nTreatment plan ({len(plan)} options):")
    for p in plan[:5]:
        print(
            f"  Rank {p.get('rank', '?')}: {p.get('therapy', '?')} "
            f"-> {p.get('target_gene', '?')} "
            f"({p.get('type', '?')}, confidence={p.get('confidence', '?')})"
        )


# ── Neoantigen Prediction ────────────────────────────────────────────────


def proof_neoantigen() -> None:
    """Validate the neoantigen prediction pipeline.

    Exercises each stage independently:

    1. **DNA → protein translation** using the built-in codon table.
    2. **MHC-I binding scoring** via the HLA-A*02:01 PSSM on three
       test peptides (a strong binder, poly-A control, and a known
       influenza epitope).
    3. **Mutant peptide generation** — creates all overlapping windows
       (8-11 AA) spanning a missense mutation site.
    4. **MAF-based prediction** — processes an EGFR L858R and BRAF
       V600E mutation annotation file and reports candidate peptides
       with MHC scores and priority labels.
    """
    from fusion_oncology.analysis.neoantigen import (
        NeoantigenPredictor,
        generate_mutant_peptides,
        score_mhc_binding,
        translate_sequence,
    )

    # ── DNA → protein ────────────────────────────────────────────────
    prot = translate_sequence("ATGGCGATCAAGCTGACG")
    print("=== Neoantigen Prediction ===")
    print(f"DNA: ATGGCGATCAAGCTGACG -> Protein: {prot}")

    # ── MHC-I binding scoring ────────────────────────────────────────
    test_peptides = ["LMVQIILAV", "AAAAAAAAA", "GILGFVFTL"]
    for pep in test_peptides:
        score = score_mhc_binding(pep)
        print(f"  Peptide {pep}: MHC score={score:.3f}")

    # ── mutant peptide generation ────────────────────────────────────
    wt_seq = "MAIKLTVANG" * 90  # 900-residue synthetic protein
    peptides = generate_mutant_peptides(wt_seq, 100, "R")
    print(f"\n  Mutant peptides generated: {len(peptides)}")
    for p in peptides[:3]:
        print(f"    WT={p['wt_peptide']} MUT={p['mut_peptide']}")

    # ── full MAF-based prediction ────────────────────────────────────
    maf = pd.DataFrame(
        {
            "Hugo_Symbol": ["EGFR", "BRAF"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Missense_Mutation",
            ],
            "HGVSp_Short": ["p.L858R", "p.V600E"],
        },
    )
    pred = NeoantigenPredictor()
    result = pred.predict_from_maf(maf)
    print(f"\n  MAF prediction: {len(result)} candidates")
    if not result.empty:
        print(result.head().to_string())


# ── CRISPR Guide Design ──────────────────────────────────────────────────


def proof_crispr() -> None:
    """Validate CRISPR sgRNA guide design.

    Constructs a synthetic 256-bp sequence containing PAM sites and
    runs the Doench Rule Set 2 scoring pipeline for EGFR.  Reports
    the number of guides found, their sequences, and composite
    on-target scores.
    """
    from fusion_oncology.models.crispr import CRISPRDesigner

    designer = CRISPRDesigner()
    seq = "ATGCGATCGATCGATCGATCGATCGATCGATCNGG" "ACGATCGATCGATCGATCNGG" + "ACGT" * 50

    guides = designer.design_for_gene("EGFR", seq)

    print("=== CRISPR Guide Design ===")
    print(f"Gene: EGFR, Sequence length: {len(seq)}")
    print(f"Guides designed: {len(guides)}")
    for g in guides[:3]:
        print(
            f"  {g.get('guide_id', '')}: "
            f"{g.get('sequence', '')} "
            f"score={g.get('composite_score', 0):.3f}"
        )


# ── SHAP Interpretability ───────────────────────────────────────────────


def proof_interpretability() -> None:
    """Validate the SHAP-based interpretability module.

    Demonstrates:
    1. Construction of a ``ShapExplainer`` from a fitted XGBoost engine.
    2. Gene-level SHAP importance ranking.
    3. Pathway-level SHAP aggregation across cancer signalling pathways.
    4. Single-sample explanation with directional indicators.
    5. Full integrated report generation.

    Uses a small synthetic dataset with 5 genes and 3 cancer types.
    """
    from fusion_oncology.analysis.interpretability import ShapExplainer
    from fusion_oncology.config import ProjectConfig
    from fusion_oncology.models.xgboost_engine import XGBoostEngine

    rng = np.random.default_rng(42)
    genes = ["EGFR", "BRAF", "TP53", "KRAS", "ALK"]
    X = pd.DataFrame(rng.random((60, 5)), columns=genes)
    y = pd.Series(np.repeat(["LUAD", "BRCA", "KIRC"], 20), name="Class")

    cfg = ProjectConfig(
        xgb_n_estimators=10,
        xgb_max_depth=3,
        enable_feature_engineering=False,
    )
    engine = XGBoostEngine(config=cfg)
    engine.fit(X, y)

    explainer = ShapExplainer.from_engine(engine)

    print("=== SHAP Interpretability ===")
    print(f"Features: {len(explainer._feature_names)}")

    # Gene importance
    gi = explainer.gene_importance(X, top_k=5)
    print(f"\nGene importance (top {len(gi)}):")
    for _, row in gi.iterrows():
        print(f"  #{int(row['rank'])} {row['gene']}: mean|SHAP|={row['mean_shap']:.6f}")

    # Pathway importance
    pw = explainer.pathway_importance(X)
    print(f"\nPathway importance ({len(pw)} pathways):")
    for _, row in pw.head(3).iterrows():
        print(f"  {row['pathway']}: SHAP_sum={row['shap_sum']:.6f}")

    # Sample explanation
    sample = explainer.explain_sample(X, sample_idx=0, top_k=3)
    print(f"\nSample 0 explanation:")
    for _, row in sample.iterrows():
        print(f"  {row['gene']}: SHAP={row['shap_value']:.4f} ({row['direction']})")

    # Full report
    report = explainer.full_report(X)
    print(f"\nFull report keys: {sorted(report.keys())}")


# ── PK/PD Pharmacokinetics ──────────────────────────────────────────────


def proof_pkpd() -> None:
    """Validate the two-compartment PK/PD model.

    Demonstrates:
    1. Drug library lookup and parameter display.
    2. 14-day PK simulation with plasma concentration time-course.
    3. Sigmoidal Emax pharmacodynamic effect computation.
    4. Daily kill-rate profile for digital twin coupling.
    5. Steady-state metrics (Cmax, Cmin, AUC₂₄).

    Uses Osimertinib 80 mg QD as the reference drug.
    """
    from fusion_oncology.models.pharmacokinetics import DRUG_PK_LIBRARY, PKPDModel

    drug = DRUG_PK_LIBRARY["Osimertinib"]
    model = PKPDModel(drug)

    print("=== PK/PD Pharmacokinetics ===")
    print(f"Drug: {drug.name}, Dose: {drug.dose_mg} mg Q{drug.dosing_interval_h}H")

    # Simulation
    df = model.simulate(duration_days=7)
    print(f"\n7-day simulation: {len(df)} time points")
    print(f"  Peak plasma: {df['plasma_conc'].max():.2f} ng/mL")
    print(f"  Trough plasma: {df['plasma_conc'].min():.2f} ng/mL")

    # Emax effect
    for conc in [10, 50, 200, 1000]:
        eff = model.emax_effect(conc)
        print(f"  Emax({conc} ng/mL) = {eff:.4f} day⁻¹")

    # Daily kill rates
    daily = model.daily_kill_rate(duration_days=7)
    print(f"\nDaily kill rates: {[f'{r:.4f}' for r in daily[:5]]}")

    # Steady-state
    ss = model.steady_state_metrics(duration_days=14)
    print(f"\nSteady-state metrics:")
    for k, v in ss.items():
        print(f"  {k}: {v}")


# ── Enhanced Immune Model ───────────────────────────────────────────────


def proof_immune() -> None:
    """Validate the structured tumour immune micro-environment model.

    Demonstrates:
    1. Four-population immune simulation (T_eff, Treg, NK, MDSC).
    2. T-cell exhaustion dynamics under chronic antigen stimulation.
    3. Spatial heterogeneity (core vs rim kill rates).
    4. Checkpoint immunotherapy comparison (none, anti-PD-1, combo).
    5. Summary output with final population counts.
    """
    from fusion_oncology.analysis.immune_model import ImmuneConfig, TumourImmuneModel

    tumour = 1e8 * np.exp(0.04 * np.arange(90))
    model = TumourImmuneModel()

    print("=== Enhanced Immune Model ===")

    # Baseline simulation
    df = model.simulate(tumour)
    print(f"90-day simulation: {len(df)} time points")
    print(f"  Final T_eff: {df['t_eff'].iloc[-1]:.0f}")
    print(f"  Final Treg: {df['treg'].iloc[-1]:.0f}")
    print(f"  Final NK: {df['nk'].iloc[-1]:.0f}")
    print(f"  Final MDSC: {df['mdsc'].iloc[-1]:.0f}")
    print(f"  Final exhaustion: {df['exhaustion'].iloc[-1]:.4f}")

    # Spatial
    rates = model.spatial_kill_rates(t_eff=1e6, nk=2e5)
    print(f"\nSpatial kill rates:")
    print(f"  Core: {rates['core_kill_rate']:.6f}")
    print(f"  Rim:  {rates['rim_kill_rate']:.6f}")

    # Checkpoint comparison
    comp = model.compare_immunotherapy(tumour)
    print(f"\nCheckpoint therapy comparison:")
    for _, row in comp.iterrows():
        print(
            f"  {row['regimen']}: T_eff={row['final_t_eff']:.0f}, "
            f"exhaust={row['final_exhaustion']:.3f}, "
            f"kill={row['final_kill_rate']:.6f}"
        )


# ── GDSC Data Loader ────────────────────────────────────────────────────


def proof_gdsc() -> None:
    """Validate the GDSC real data loader.

    Demonstrates:
    1. Offline synthetic data generation matching GDSC schema.
    2. Dose-response data access (IC₅₀, AUC).
    3. Drug sensitivity queries by drug name and tissue.
    4. Resistant/sensitive cell-line identification.
    5. Training matrix construction for XGBoost.
    6. Summary statistics.

    Uses offline mode to avoid network dependency.
    """
    from fusion_oncology.data.gdsc import GDSCLoader

    loader = GDSCLoader(offline=True)

    print("=== GDSC Data Loader ===")
    s = loader.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Drug sensitivity
    first_drug = loader.dose_response["DRUG_NAME"].iloc[0]
    df = loader.drug_sensitivity(first_drug)
    print(f"\nSensitivity for {first_drug}: {len(df)} cell lines")

    # Resistant / sensitive
    resistant = loader.resistant_cell_lines(first_drug)
    sensitive = loader.sensitive_cell_lines(first_drug)
    print(f"  Resistant (LN_IC50 > 2): {len(resistant)}")
    print(f"  Sensitive (LN_IC50 < -1): {len(sensitive)}")

    # Training matrix
    X, y = loader.training_matrix(first_drug)
    print(f"\nTraining matrix: {X.shape[0]} samples x {X.shape[1]} features")
    if len(y) > 0:
        print(f"  Sensitive fraction: {100 * y.mean():.1f}%")


# ── Domain Adaptation ───────────────────────────────────────────────────


def proof_domain_adapt() -> None:
    """Validate the cell-line → patient domain adaptation pipeline.

    Demonstrates:
    1. Quantile normalisation to a reference distribution.
    2. ComBat batch correction reducing domain shift.
    3. Feature alignment filtering genes by cross-domain overlap.
    4. End-to-end adaptation pipeline (align → normalise → correct).
    5. Pre/post batch-effect comparison.

    Uses synthetic cell-line and patient expression matrices.
    """
    from fusion_oncology.data.domain_adaptation import DomainAdapter, combat_correct

    rng = np.random.default_rng(42)
    genes = [f"GENE_{i}" for i in range(100)]

    cl = pd.DataFrame(
        rng.lognormal(3.0, 1.5, (50, 100)),
        index=[f"CL_{i}" for i in range(50)],
        columns=genes,
    )
    pt = pd.DataFrame(
        rng.lognormal(4.0, 1.2, (30, 100)),
        index=[f"PT_{i}" for i in range(30)],
        columns=genes,
    )

    print("=== Domain Adaptation ===")
    pre_diff = abs(cl.mean().mean() - pt.mean().mean())
    print(f"Pre-adaptation mean difference: {pre_diff:.2f}")

    # ComBat alone
    combined = pd.concat([cl, pt])
    batch = np.array(["CL"] * 50 + ["PT"] * 30)
    corrected = combat_correct(combined, batch)
    cl_c = corrected.iloc[:50]
    pt_c = corrected.iloc[50:]
    post_diff = abs(cl_c.mean().mean() - pt_c.mean().mean())
    print(f"Post-ComBat mean difference: {post_diff:.2f}")

    # Full pipeline
    adapter = DomainAdapter()
    cl_out, pt_out = adapter.adapt(cl, pt)
    s = adapter.summary()
    print(f"\nFull pipeline:")
    print(f"  Aligned genes: {s['n_aligned_genes']}")
    print(f"  Output shapes: CL={cl_out.shape}, PT={pt_out.shape}")
    final_diff = abs(cl_out.mean().mean() - pt_out.mean().mean())
    print(f"  Post-adaptation mean difference: {final_diff:.2f}")


# ── RL treatment optimiser ───────────────────────────────────────────────


def proof_rl() -> None:
    """Validate the reinforcement learning treatment optimiser.

    Demonstrates:
    1. TreatmentEnv wrapping the digital twin ODE system.
    2. REINFORCE policy-gradient agent training.
    3. Adaptive therapy (Zhang 2017) baseline comparison.
    4. Maximum tolerated dose (MTD) strategy comparison.
    5. Strategy comparison table.

    Uses small training budget (10 episodes, 10 steps) for fast
    demonstration — not intended for clinical-grade convergence.
    """
    from fusion_oncology.models.digital_twin import SimulationConfig
    from fusion_oncology.models.rl_optimizer import (
        AdaptiveTherapyAgent,
        RLConfig,
        REINFORCEAgent,
        TreatmentEnv,
        compare_strategies,
    )

    print("=== RL Treatment Optimiser ===")

    rl_cfg = RLConfig(n_episodes=10, max_steps=10, decision_interval=7)
    sim_cfg = SimulationConfig(simulation_days=70)

    # 1. Environment
    env = TreatmentEnv(sim_cfg, rl_cfg)
    obs = env.reset(seed=42)
    print(f"  Observation dim: {len(obs)}")
    print(f"  Action space: {env.action_space_n} discrete actions")
    print(f"  Initial obs: {np.round(obs, 3)}")

    # 2. Single step
    obs2, reward, done, info = env.step(2)  # standard dose
    print(f"  After step(2): reward={reward:.3f}, done={done}")

    # 3. RL agent training
    agent = REINFORCEAgent(rl_cfg)
    agent.train(env, verbose=False)
    rl_eval = agent.evaluate(env, n_episodes=3)
    print(f"\n  RL mean reward:        {rl_eval['mean_reward']:.3f}")
    print(f"  RL mean final tumour:  {rl_eval['mean_final_tumour']:.1f}")

    # 4. Adaptive therapy
    adaptive = AdaptiveTherapyAgent()
    adapt_result = adaptive.run(env)
    print(f"  Adaptive total reward: {adapt_result['total_reward']:.3f}")
    print(f"  Adaptive final tumour: {adapt_result['final_tumour']:.1f}")

    # 5. Strategy comparison
    df = compare_strategies(sim_cfg, rl_cfg)
    print("\n  Strategy comparison:")
    for _, row in df.iterrows():
        print(
            f"    {row['strategy']:20s}  reward={row['mean_reward']:+.2f}  "
            f"tumour={row['mean_final_tumour']:.0f}"
        )


# ── GNN network scoring ─────────────────────────────────────────────────


def proof_gnn() -> None:
    """Validate the Graph Neural Network drug-gene-pathway scorer.

    Demonstrates:
    1. Graph construction from curated drug-target and pathway databases.
    2. GCN message-passing to produce learned node embeddings.
    3. Learned strategic gene scoring (replaces static centrality).
    4. Drug combination synergy prediction via inner-product decode.
    5. Pathway importance ranking.
    6. Gene ranking.
    """
    from fusion_oncology.analysis.gnn_network import GNNScorer

    print("=== GNN Drug-Gene-Pathway Network ===")
    scorer = GNNScorer(hidden_dim=32, embed_dim=16, n_layers=3)

    s = scorer.summary()
    print(f"  Nodes: {s['n_nodes']}  Edges: {s['n_edges']}")
    print(f"  Node types: {s['node_types']}")
    print(f"  Embed dim: {s['embed_dim']}  Layers: {s['n_layers']}")

    # Strategic scores
    for gene in ["EGFR", "BRAF", "TP53", "KRAS"]:
        sc = scorer.strategic_score(gene)
        print(f"  Strategic score({gene}): {sc:.4f}")

    # Drug combination
    combo = scorer.drug_combination_score("Erlotinib", "Gefitinib")
    print(f"\n  Drug combo (Erlotinib + Gefitinib): {combo:.4f}")

    # Pathway importance
    pw = scorer.pathway_importance()
    print(f"\n  Top pathways ({len(pw)} total):")
    for _, row in pw.head(5).iterrows():
        print(f"    {row['pathway']:30s} score={row['gnn_score']:.4f}")

    # Gene ranking
    gr = scorer.gene_ranking()
    print(f"\n  Top genes ({len(gr)} total):")
    for _, row in gr.head(5).iterrows():
        print(f"    {row['gene']:15s} score={row['gnn_score']:.4f}  deg={row['degree']}")


# ── Bayesian uncertainty ─────────────────────────────────────────────────


def proof_uncertainty() -> None:
    """Validate Bayesian uncertainty quantification.

    Demonstrates:
    1. Bootstrap ensemble training with 10 XGBoost members.
    2. Posterior probability distributions from ensemble.
    3. Credible intervals with configurable coverage.
    4. Entropy-based confidence scoring.
    5. Decision quality classification (HIGH/MODERATE/LOW/UNCERTAIN).
    6. Calibration curve and Expected Calibration Error (ECE).
    """
    from fusion_oncology.analysis.uncertainty import (
        BayesianPredictor,
        UncertaintyConfig,
        calibration_curve,
        expected_calibration_error,
    )

    print("=== Bayesian Uncertainty Quantification ===")

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((300, 30)),
        columns=[f"GENE{i}" for i in range(30)],
    )
    y = pd.Series(rng.choice(["BRCA", "LUAD", "COAD", "GBM"], 300))

    cfg = UncertaintyConfig(n_bootstrap=10, confidence_level=0.95)
    bp = BayesianPredictor(uq_config=cfg)
    bp.fit(X, y)

    s = bp.summary()
    print(f"  Bootstrap members: {s['n_models_trained']}")
    print(f"  Classes: {s['classes']}")
    print(f"  Confidence level: {s['confidence_level']}")

    # Predictions with uncertainty
    preds = bp.predict(X.iloc[:20])
    print(f"\n  Predictions (first 5):")
    for _, row in preds.head(5).iterrows():
        print(
            f"    {row['prediction']:5s}  conf={row['confidence']:.3f}  "
            f"entropy={row['entropy']:.3f}  quality={row['decision_quality']}"
        )

    # Credible intervals
    ci = bp.credible_intervals(X.iloc[:5])
    print(f"\n  Credible interval columns: {list(ci.columns)[:6]}...")

    # Decision quality distribution
    quality = preds["decision_quality"].value_counts()
    print(f"\n  Decision quality distribution:")
    for q, c in quality.items():
        print(f"    {q}: {c}")

    # Calibration (binary: BRCA vs rest)
    y_bin = (y == "BRCA").astype(int).values
    result = bp.ensemble.predict_with_uncertainty(X)
    brca_idx = list(s["classes"]).index("BRCA") if "BRCA" in s["classes"] else 0
    y_prob = result["mean_proba"][:, brca_idx]
    ece = expected_calibration_error(y_bin, y_prob)
    print(f"\n  ECE (BRCA vs rest): {ece:.4f}")

    cal = calibration_curve(y_bin, y_prob, n_bins=5)
    print(f"  Calibration bins: {len(cal)}")


# ── TCGA patient cohort ─────────────────────────────────────────────────


def proof_tcga() -> None:
    """Validate the TCGA patient cohort loader and validator.

    Demonstrates:
    1. Synthetic TCGA data generation matching real data schema.
    2. Multi-omics data access (expression, mutation, CNA, clinical).
    3. Cancer type distribution and driver gene enrichment.
    4. Retrospective classifier validation on held-out patients.
    5. Mutation enrichment across cancer types.
    6. Survival stratification by mutation status.
    """
    from fusion_oncology.data.tcga import TCGACohortValidator, TCGALoader

    print("=== TCGA Patient Cohort Validation ===")

    loader = TCGALoader(n_patients=200, seed=42)
    s = loader.summary()
    for k, v in s.items():
        if k != "cancer_types":
            print(f"  {k}: {v}")
    print(f"  Cancer type distribution:")
    for ct, n in s["cancer_types"].items():
        print(f"    {ct}: {n}")

    # Training data
    X, y = loader.training_data()
    print(f"\n  Training matrix: {X.shape[0]} patients x {X.shape[1]} genes")

    # Mutation frequency for key drivers
    for gene in ["TP53", "EGFR", "BRAF"]:
        freq = loader.mutation_frequency(gene)
        total = sum(freq.values()) / max(len(freq), 1)
        print(f"  {gene} mean freq: {total:.3f}")

    # Validator
    validator = TCGACohortValidator(loader)

    # Retrospective classifier validation
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    val = validator.validate_classifier(model)
    print(f"\n  Classifier validation:")
    print(f"    Accuracy: {val['accuracy']:.3f}")
    print(f"    Weighted F1: {val['weighted_f1']:.3f}")
    print(f"    Train/Test: {val['n_train']}/{val['n_test']}")

    # Mutation enrichment
    enrich = validator.mutation_enrichment(["TP53", "KRAS", "EGFR"])
    print(f"\n  Mutation enrichment ({len(enrich)} records):")
    for _, row in enrich.head(6).iterrows():
        print(f"    {row['gene']:6s} {row['cancer_type']:5s} freq={row['mutation_freq']:.3f}")

    # Survival stratification
    surv = validator.survival_stratification("TP53")
    print(f"\n  TP53 survival stratification:")
    print(f"    Mutated: {surv['n_mutated']} pts, median={surv['median_survival_mutated']:.1f} mo")
    print(
        f"    Wildtype: {surv['n_wildtype']} pts, median={surv['median_survival_wildtype']:.1f} mo"
    )
    if surv.get("survival_ratio") is not None:
        print(f"    Ratio: {surv['survival_ratio']:.3f}")


# ── Real TCGA data validation ───────────────────────────────────────────


def proof_real_data() -> None:
    """Validate real TCGA data loading and clinical analysis.

    Demonstrates:
    1. TCGA-LUAD data download from cBioPortal (or synthetic fallback).
    2. Gene expression matrix (patients × genes) construction.
    3. Driver gene mutation frequency analysis.
    4. Survival label derivation (median OS split).
    5. Summary statistics matching published cohort parameters.
    """
    from fusion_oncology.validation.real_data import RealDataConfig, RealTCGALoader

    print("=== Real TCGA Data Validation ===")

    cfg = RealDataConfig(cancer_type="LUAD", max_genes=200, seed=42)
    loader = RealTCGALoader(data_config=cfg)

    s = loader.summary()
    print(f"  Data source: {s['data_source']}")
    print(f"  Cancer type: {s['cancer_type']}")
    print(f"  Patients: {s['n_patients']}")
    print(f"  Genes: {s['n_genes']}")
    print(f"  Mutations: {s['n_mutations']}")
    print(f"  Drivers mutated: {s['n_drivers_mutated']}")
    print(f"  Median OS: {s['median_os_months']:.1f} months")

    # Expression matrix
    X, y = loader.expression_matrix()
    print(f"\n  Expression matrix: {X.shape[0]} × {X.shape[1]}")
    print(f"  Label distribution: {y.value_counts().to_dict()}")

    # Driver mutation profile
    profile = loader.driver_mutation_profile()
    print(f"\n  Top driver mutations:")
    for _, row in profile.head(8).iterrows():
        if row["frequency"] > 0:
            print(
                f"    {row['gene']:8s} freq={row['frequency']:.3f} "
                f"({row['n_mutated']}/{row['total_patients']})"
            )

    # Survival data
    surv = loader.survival_data()
    deceased = (surv["os_status"] == "DECEASED").sum()
    print(f"\n  Survival: {deceased}/{len(surv)} deceased")
    print(f"  Median OS: {surv['os_months'].median():.1f} months")


# ── Benchmark suite ──────────────────────────────────────────────────────


def proof_benchmark() -> None:
    """Validate the benchmark framework with full baseline comparison.

    Demonstrates:
    1. Full pipeline vs 4 standard baselines (LR, RF, SVM, vanilla XGB).
    2. Paired Wilcoxon signed-rank tests for statistical significance.
    3. Ablation study (5 variants including component removal).
    4. Cross-dataset stability analysis (5 sub-samples).
    5. Formatted comparison tables with p-values.

    Uses TCGA-LUAD data (or synthetic fallback) for all comparisons.
    """
    from fusion_oncology.validation.benchmark import (
        BenchmarkConfig,
        BenchmarkSuite,
    )
    from fusion_oncology.validation.real_data import RealDataConfig, RealTCGALoader

    print("=== Benchmark Suite ===")

    # Load data
    dcfg = RealDataConfig(cancer_type="LUAD", max_genes=100, seed=42)
    loader = RealTCGALoader(data_config=dcfg)
    X, y = loader.expression_matrix()
    print(f"  Data: {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"  Source: {loader.data_source}")

    bcfg = BenchmarkConfig(n_folds=5, n_repeats=2, seed=42)
    suite = BenchmarkSuite(X, y, config=bcfg)

    # Baseline comparison
    print("\n  Running baseline comparison …")
    comp = suite.baseline_comparison()
    print("\n  Model                    Accuracy    F1        AUC       Δ Acc     p(F1)")
    print("  " + "-" * 80)
    for _, row in comp.iterrows():
        sig = " *" if row.get("significant", False) else "  "
        print(
            f"  {row['model']:25s} {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f}  "
            f"{row['mean_f1']:.4f}±{row['std_f1']:.4f}  "
            f"{row['mean_auc']:.4f}  "
            f"{row['improvement_accuracy']:+.4f}  "
            f"{row['p_value_f1']:.4f}{sig}"
        )

    # Ablation
    print("\n  Running ablation study …")
    abl = suite.ablation_study()
    print("\n  Variant                               F1       Δ F1      p-value")
    print("  " + "-" * 70)
    for _, row in abl.iterrows():
        print(
            f"  {row['variant']:38s} {row['mean_f1']:.4f}  "
            f"{row['delta_f1']:+.4f}  {row['p_value']:.4f}"
        )

    # Stability
    print("\n  Running stability analysis …")
    stab = suite.stability_analysis(n_subsamples=5)
    acc_mean = stab["mean_accuracy"].mean()
    acc_std = stab["mean_accuracy"].std()
    cv_coeff = acc_std / max(acc_mean, 1e-10)
    print(f"  Stability: mean acc={acc_mean:.4f}, std={acc_std:.4f}, CV={cv_coeff:.4f}")


# ── Methodology formalisation ────────────────────────────────────────────


def proof_methodology() -> None:
    """Validate the methodology formalisation and architecture analysis.

    Demonstrates:
    1. Formal 7-layer architecture specification.
    2. Feature engineering contribution (marginal accuracy / F1 gain).
    3. Importance-weighted vs uniform/random embedding fusion comparison.
    4. Hyperparameter sensitivity sweep (depth, estimators, LR).
    5. Component synergy analysis (additive vs super-additive effects).

    Provides empirical evidence for each architectural claim.
    """
    from fusion_oncology.validation.methodology import MethodologyFormaliser
    from fusion_oncology.validation.real_data import RealDataConfig, RealTCGALoader

    print("=== Methodology Formalisation ===")

    # Load data
    dcfg = RealDataConfig(cancer_type="LUAD", max_genes=100, seed=42)
    loader = RealTCGALoader(data_config=dcfg)
    X, y = loader.expression_matrix()

    fm = MethodologyFormaliser(X, y, seed=42)

    # Architecture spec
    spec = fm.architecture_specification()
    print(f"  Architecture: {spec['name']} v{spec['version']}")
    print(f"  Layers defined: 7 (FE → IWC → SE → FFC → FC → DTP → RTO)")

    # Feature engineering contribution
    print("\n  Feature Engineering Contribution:")
    fe = fm.feature_engineering_contribution()
    print(f"    Marginal accuracy: {fe.marginal_accuracy:+.4f}")
    print(f"    Marginal F1:       {fe.marginal_f1:+.4f}")
    print(f"    Information gain:  {fe.information_gain:.4f}")
    print(f"    Utilisation:       {fe.feature_utilisation:.1%}")

    # Importance weighting
    print("\n  Importance-Weighted Fusion Comparison:")
    iw = fm.importance_weighting_analysis()
    print(f"    Importance-weighted accuracy: {iw['weighted_accuracy']:.4f}")
    print(f"    Uniform-weighted accuracy:    {iw['uniform_accuracy']:.4f}")
    print(f"    Random-weighted accuracy:     {iw['random_accuracy']:.4f}")
    print(f"    Improvement over uniform:     {iw['improvement_over_uniform']:+.4f}")
    print(f"    Improvement over random:      {iw['improvement_over_random']:+.4f}")

    # Hyperparameter sensitivity
    print("\n  Hyperparameter Sensitivity:")
    hp = fm.hyperparameter_sensitivity()
    for param in hp["parameter"].unique():
        subset = hp[hp["parameter"] == param]
        best = subset.loc[subset["accuracy"].idxmax()]
        print(
            f"    {param}: best={best['value']} "
            f"(acc={best['accuracy']:.4f}), "
            f"range=[{subset['accuracy'].min():.4f}, {subset['accuracy'].max():.4f}]"
        )

    # Synergy
    print("\n  Component Synergy:")
    syn = fm.synergy_analysis()
    print(f"    Baseline:        {syn['baseline_accuracy']:.4f}")
    print(f"    +FE:             {syn['improvement_A_feature_eng']:+.4f}")
    print(f"    +Deep trees:     {syn['improvement_B_deeper_trees']:+.4f}")
    print(f"    Combined:        {syn['improvement_AB_combined']:+.4f}")
    print(f"    Predicted add.:  {syn['predicted_additive']:+.4f}")
    print(f"    Synergy score:   {syn['synergy_score']:+.4f}")
    print(f"    Synergistic:     {'YES' if syn['is_synergistic'] else 'NO'}")


# ── CLI dispatch ─────────────────────────────────────────────────────────

#: Registry of proof name → callable.
PROOF_REGISTRY: dict[str, Any] = {
    "xgboost": proof_xgboost,
    "twin": proof_digital_twin,
    "resistance": proof_resistance,
    "pathway": proof_pathway,
    "drug": proof_drug_target,
    "sl": proof_synthetic_lethality,
    "network": proof_network,
    "cdx": proof_companion_dx,
    "neoantigen": proof_neoantigen,
    "crispr": proof_crispr,
    "interpretability": proof_interpretability,
    "pkpd": proof_pkpd,
    "immune": proof_immune,
    "gdsc": proof_gdsc,
    "domain_adapt": proof_domain_adapt,
    "rl": proof_rl,
    "gnn": proof_gnn,
    "uncertainty": proof_uncertainty,
    "tcga": proof_tcga,
    "real_data": proof_real_data,
    "benchmark": proof_benchmark,
    "methodology": proof_methodology,
}


def main() -> None:
    """Entry point for the proof runner.

    Reads the first CLI argument to select a proof (or ``"all"`` to
    run every proof sequentially).  Prints a banner between proofs
    when running in batch mode.

    Raises
    ------
    KeyError
        If an unrecognised proof name is supplied.
    """
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "all":
        for name, fn in PROOF_REGISTRY.items():
            print(f"\n{'=' * 60}")
            print(f"  PROOF: {name.upper()}")
            print(f"{'=' * 60}")
            fn()
    elif which in PROOF_REGISTRY:
        PROOF_REGISTRY[which]()
    else:
        available = ", ".join(PROOF_REGISTRY)
        print(f"Unknown proof '{which}'.  Available: {available}")
        sys.exit(1)


if __name__ == "__main__":
    main()
