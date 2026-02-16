![image](https://github.com/mytechnotalent/fusion_oncology/blob/main/fusion_oncology.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Fusion Oncology

Fusion Oncology is an end-to-end precision oncology platform spanning **44 source modules** and **19,000+ lines** of production code, validated by **528 automated tests** (37 test files, 9,400+ test lines).

It builds a **multi-modal fusion model** -- a single XGBoost classifier trained on drug sensitivity features, 10 engineered distributional features, and PCA-compressed sensitivity-weighted DNABERT-2 sequence embeddings -- to prioritise therapeutic targets and generate standardised AMP/ASCO/CAP-tiered companion diagnostic reports. The pipeline includes intelligent class merging, class-balanced sample weights, optional Optuna Bayesian hyperparameter optimisation, and repeated stratified cross-validation for metric stability.

---

## What It Does

| Component            | Method                                  | What It Captures                                |
| -------------------- | --------------------------------------- | ----------------------------------------------- |
| **Drug sensitivity** | XGBoost on GDSC LN_IC50 features        | Which genes best discriminate cancer types      |
| **Engineered**       | 10 row-level distributional features    | Per-sample mean, std, skew, kurtosis, IQR, CV   |
| **Genomic context**  | DNABERT-2 768-dim embeddings, PCA to 50 | Noise-reduced sensitivity-weighted gene context |
| **Fusion model**     | XGBoost on concatenated (N + 10 + 50)   | Jointly learned drug + distribution + sequence  |

The fusion model produces **one unified set of CV metrics** (Accuracy, Precision, Recall, F1, F2, ROC AUC). The **Fusion Index** (importance x instability x 1000) ranks targets that are both biologically important *and* structurally vulnerable.

### Core Analysis

- **Pathway enrichment** -- maps targets to PI3K-Akt, MAPK, p53, Wnt, Notch, cell-cycle, DNA-repair, apoptosis, angiogenesis, and immune-checkpoint pathways
- **Drug-target mapping** -- cross-references 20+ gene targets against approved oncology drugs (EGFR-Osimertinib, BRAF-Vemurafenib, KRAS-Sotorasib, etc.)
- **Survival analysis** -- Kaplan-Meier + log-rank stratification hooks
- **Publication-quality figures** -- bar charts, scatter plots, heatmaps, box plots
- **Self-contained HTML report** -- one-click shareable with collaborators

### Advanced Therapeutic Intelligence

- **Multi-omics integration** -- MAF mutation parsing, copy-number alteration analysis, methylation profiling, and combined feature matrix construction
- **Clinical evidence aggregation** -- real-time queries to OpenTargets, CIViC, and ClinicalTrials.gov APIs with composite evidence scoring
- **Synthetic lethality detection** -- curated database of 24 SL pairs (BRCA1/2-PARP, RB1-Aurora kinase, etc.) plus expression-based anti-correlation screening
- **Neoantigen prediction** -- codon translation, mutant peptide generation, simplified MHC-I binding scoring for immunotherapy candidate ranking
- **Resistance prediction** -- 12-gene resistance mechanism database with risk scoring, drug-specific evasion strategies, and clinical counter-measures
- **Network pharmacology** -- drug-gene-pathway tripartite interaction graph with degree/betweenness centrality, polypharmacology scoring, and combination target identification
- **CRISPR guide design** -- PAM scanning on both strands, Doench-inspired on-target scoring, off-target heuristics, and exportable guide libraries
- **Companion diagnostics** -- patient-level mutation profiling, AMP/ASCO/CAP actionability tiering (Tiers I-IV), drug matching, and ranked treatment plans
- **Digital twin simulation** -- Gompertzian tumour growth ODE model with drug regimens (cycling/scheduling), immune dynamics, RECIST response classification, and regimen comparison

### GNN, RL, PK/PD, and Uncertainty

- **GNN interaction learning** -- graph neural network replacing static centrality with learned drug-gene-pathway embeddings via message passing
- **RL treatment optimiser** -- policy-gradient agent that learns optimal dosing schedules using the digital twin as a gymnasium-style environment
- **Compartmental PK/PD** -- two-compartment pharmacokinetics with Emax pharmacodynamics, replacing a flat efficacy constant in the digital twin
- **Immune micro-environment model** -- structured TIME compartment with CD8+ T cells, Tregs, MDSCs, NK cells, and cytokine-mediated dynamics
- **Bayesian uncertainty quantification** -- posterior credible intervals and calibrated confidence scores for clinical predictions
- **SHAP interpretability** -- gene-level and pathway-level explanations for XGBoost predictions

### Validation and Benchmarks

- **Benchmark framework** -- ablation studies and baseline comparisons (Random Forest, Logistic Regression, SVM, MLP) with paired t-tests
- **Methodology formalisation** -- formal mathematical specification and empirical validation of the architecture
- **Real clinical validation** -- cBioPortal TCGA data ingestion for open-access expression, mutation, and clinical outcome testing
- **Domain adaptation** -- cell-line to patient distribution alignment bridging the in-vitro / in-vivo gap

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/mytechnotalent/fusion_oncology.git
cd fusion_oncology
pip install -e ".[dev]"

# Run the full 7-step pipeline
fusion-oncology run

# Faster smoke test
fusion-oncology run --top-k 3 --fuzz-iterations 5

# Download and cache the data
fusion-oncology ingest

# Query clinical evidence for a gene
fusion-oncology evidence --gene EGFR

# Check resistance mechanisms
fusion-oncology resistance --gene BRAF

# Run a digital twin tumour simulation
fusion-oncology simulate --days 180

# Run companion diagnostics
fusion-oncology companion-dx --mutations "EGFR:T790M,BRAF:V600E" \
    --cancer-type NSCLC

# Regenerate a report from saved results
fusion-oncology report results/fusion_results.csv
```

### Kaggle Integration

Run on **Kaggle** with the **GDSC dataset** (1,000+ cancer cell lines):

```bash
# Upload kaggle_notebook.ipynb to Kaggle
# Add GDSC dataset from:
#   kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc
# Then run the notebook or use CLI:
!pip install git+https://github.com/mytechnotalent/fusion_oncology.git
!fusion-oncology run --top-k 10 --output-dir /kaggle/working/results
```

See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for complete GDSC setup and alternative datasets.

---

## Project Structure

```
fusion_oncology/
 pyproject.toml
 Makefile
 install.sh
 docs/
    architecture.md
 src/fusion_oncology/
    __init__.py
    config.py                    # Central dataclass (29 fields)
    cli.py                       # Click CLI (8 commands)
    data/
       ingestion.py             # TCGA download + ZIP extraction
       preprocessing.py         # Variance filter, log-norm, PCA
       cache.py                 # Filesystem artefact cache
       multi_omics.py           # MAF mutations, CNA, methylation
       gdsc.py                  # GDSC dose-response loader
       tcga.py                  # TCGA patient cohort loader
       domain_adaptation.py     # Cell-line to patient alignment
    models/
       xgboost_engine.py        # XGBoost training + importance
       dnabert_engine.py        # DNABERT-2 sequence embedding
       fusion.py                # Multi-modal fusion pipeline
       crispr.py                # CRISPR guide design + scoring
       companion_dx.py          # AMP/ASCO/CAP companion Dx
       digital_twin.py          # Gompertzian tumour ODE model
       pharmacokinetics.py      # Two-compartment PK/PD model
       rl_optimizer.py          # RL dosing optimiser
    analysis/
       instability.py           # Mutation fuzzing + cosine drift
       pathway.py               # KEGG/Reactome pathway lookup
       drug_target.py           # Drug-target annotation
       survival.py              # Kaplan-Meier survival hooks
       clinical_evidence.py     # OpenTargets + CIViC + CT.gov
       synthetic_lethality.py   # SL pair detection + screening
       neoantigen.py            # Peptide + MHC binding scoring
       resistance.py            # Resistance mechanism prediction
       network_pharmacology.py  # Drug-gene-pathway network
       gnn_network.py           # GNN message-passing embeddings
       immune_model.py          # TIME immune compartment
       interpretability.py      # SHAP model explanations
       uncertainty.py           # Bayesian confidence intervals
    viz/
       plots.py                 # Matplotlib/Seaborn figures
       report.py                # HTML report generator
    validation/
       benchmark.py             # Ablation + baseline comparisons
       methodology.py           # Architecture formalisation
       real_data.py             # cBioPortal TCGA validation
    utils/
       bio.py                   # Entrez fetch, GC, CpG islands
       log.py                   # Logging setup
 tests/
    conftest.py                  # Shared fixtures, OMP guard
    test_benchmark.py
    test_bio.py
    test_cache.py
    test_cli.py
    test_clinical_evidence.py
    test_companion_dx.py
    test_config.py
    test_crispr.py
    test_digital_twin.py
    test_dnabert_engine.py
    test_domain_adaptation.py
    test_drug_target.py
    test_fusion_engine.py
    test_gdsc.py
    test_gnn_network.py
    test_immune_model.py
    test_ingestion.py
    test_instability.py
    test_interpretability.py
    test_log.py
    test_methodology.py
    test_multi_omics.py
    test_neoantigen.py
    test_network_pharmacology.py
    test_pathway.py
    test_pharmacokinetics.py
    test_plots.py
    test_preprocessing.py
    test_real_data.py
    test_report.py
    test_resistance.py
    test_rl_optimizer.py
    test_survival.py
    test_synthetic_lethality.py
    test_tcga.py
    test_uncertainty.py
    test_xgboost_engine.py
```

---

## CLI Reference

### All Commands

```
$ fusion-oncology --help

Usage: fusion-oncology [OPTIONS] COMMAND [ARGS]...

  Fusion Oncology - Precision cancer therapeutics platform

Commands:
  clear-cache   Delete all locally cached artefacts
  companion-dx  Run companion diagnostic analysis on patient mutations
  evidence      Query clinical evidence databases for genes
  ingest        Download and cache the TCGA Pan-Cancer dataset
  report        Regenerate HTML report from saved results
  resistance    Predict resistance mechanisms for genes
  run           Run the full 7-step fusion analysis pipeline
  simulate      Run digital twin tumour growth simulation
```

---

### 1. Evidence Query

Query clinical evidence from OpenTargets, CIViC, and ClinicalTrials.gov.

```bash
$ fusion-oncology evidence EGFR BRAF
```

```
Clinical Evidence for EGFR:
  OpenTargets: 85.0% confidence
  ClinicalTrials.gov: 20 trials
  CIViC: Data unavailable

Clinical Evidence for BRAF:
  OpenTargets: 87.7% confidence
  ClinicalTrials.gov: 15 trials
  CIViC: Data unavailable

Composite Evidence Scores:
  EGFR: 85.0% (high actionability)
  BRAF: 87.7% (high actionability)
```

---

### 2. Resistance Mechanisms

Predict resistance mechanisms and evasion strategies for drug targets.

```bash
$ fusion-oncology resistance EGFR ALK
```

```
Resistance Profile for EGFR:

Mechanisms:
  1. T790M Gatekeeper Mutation
     Risk: HIGH | Clinical: Very common in osimertinib resistance
     Strategy: Switch to 3rd-gen TKI or 4th-gen EGFR-selective

  2. MET Amplification
     Risk: MEDIUM | Clinical: Bypass signaling in 5-20% of cases
     Strategy: Combine EGFR-TKI with MET inhibitor

  3. HER2 Amplification
     Risk: MEDIUM | Clinical: Emerging bypass in 5-12%
     Strategy: Dual EGFR-HER2 blockade

Resistance Profile for ALK:

Mechanisms:
  1. ALK G1202R Gatekeeper Mutation
     Risk: HIGH | Clinical: Most common in lorlatinib resistance
     Strategy: Investigational 4th-gen ALK inhibitor or TPX-0131

  2. ALK L1196M Mutation
     Risk: HIGH | Clinical: Sensitive to lorlatinib
     Strategy: Switch to 3rd-gen lorlatinib
```

---

### 3. Digital Twin Simulation

Simulate tumour growth dynamics with drug regimens using a Gompertzian ODE model.

```bash
$ fusion-oncology simulate --drug Osimertinib --efficacy 0.15 --days 90
```

```
Digital Twin Tumour Simulation

Configuration:
  Initial Volume: 10000.0 mm3
  Simulation Duration: 90 days
  Drug: Osimertinib
  Efficacy: 0.15/day

Final Results (Day 90):
  Tumour Volume: < 0.001 mm3
  Reduction: 100.0%
  RECIST Response: CR (Complete Response)

Trajectory saved to: digital_twin_trajectory.csv
```

Options: `--days`, `--drug`, `--efficacy`, `--days-on`, `--days-off`,
`--initial-volume`, `--growth-rate`, `--carrying-capacity`, `--output`

---

### 4. Companion Diagnostics

Generate treatment recommendations for patient mutation profiles.

```bash
$ cat > patient_mutations.json << 'EOF'
{
  "patient_id": "PT-2024-001",
  "cancer_type": "Non-Small Cell Lung Cancer",
  "mutations": [
    {"gene": "EGFR", "variant": "L858R", "vaf": 0.42},
    {"gene": "TP53", "variant": "R273H", "vaf": 0.38},
    {"gene": "KRAS", "variant": "G12C", "vaf": 0.15}
  ]
}
EOF

$ fusion-oncology companion-dx patient_mutations.json
```

```
=== COMPANION DIAGNOSTIC REPORT ===

Patient: PT-2024-001
Cancer Type: Non-Small Cell Lung Cancer

Detected Mutations:
  EGFR L858R (VAF: 42.0%) - Tier I
  TP53 R273H (VAF: 38.0%) - Tier III
  KRAS G12C (VAF: 15.0%) - Tier I

Treatment Recommendations (Ranked by Confidence):

1. Osimertinib (EGFR L858R) -- 90% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: FDA-approved for EGFR-mutant NSCLC

2. Sotorasib (KRAS G12C) -- 78% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: FDA-approved for KRAS G12C NSCLC

3. Afatinib (EGFR L858R) -- 72% confidence
   AMP/ASCO/CAP Tier: I

Resistance Risk Alerts:
  EGFR T790M Gatekeeper Mutation (HIGH risk)
  MET Amplification (MEDIUM risk)

Synthetic Lethality Opportunities:
  CHK1 inhibition (TP53 R273H synthetic lethal)
  WEE1 inhibition (TP53 R273H synthetic lethal)

Clinical Summary:
  High-confidence actionable targets: 2 (EGFR, KRAS)
  FDA-approved matched therapies: 6
  Primary recommendation: Osimertinib monotherapy
```

---

### 5. Full Pipeline

```bash
# Full analysis
$ fusion-oncology run

# Fast smoke test
$ fusion-oncology run --top-k 3 --fuzz-iterations 5

# High-resolution analysis
$ fusion-oncology run --top-k 20 --fuzz-iterations 50 --xgb-trees 200
```

Options: `--top-k`, `--fuzz-iterations`, `--xgb-trees`, `--xgb-depth`,
`--output-dir`, `--log-level`, `--skip-plots`, `--skip-report`

---

### 6. Data Ingestion

```bash
$ fusion-oncology ingest
```

Downloads the TCGA Pan-Cancer dataset (801 samples x 20,531 genes) from
the UCI Machine Learning Repository and caches it locally.

---

### 7. Report Generation

```bash
$ fusion-oncology report results/fusion_results.csv
```

Regenerates the HTML report from previously saved results.

---

### 8. Cache Management

```bash
$ fusion-oncology clear-cache
```

Clears all locally cached data and model artefacts.

---

## How the Fusion Pipeline Works

```
               GDSC Drug Sensitivity Data
                         |
         +---------------+---------------+
         |                               |
  Step 1: XGBoost Baseline       Step 2: DNABERT-2
  (multi-class on LN_IC50)       (768-dim gene embeddings)
         |                               |
    Top-K gene importances        Embedding matrix (K x 768)
         |                               |
         +--------> Step 3: Fusion <-----+
                    Weight embeddings by
                    drug sensitivity per
                    cell line -> 768-dim
                    context vector
                         |
              Concatenate: original
              features + 10 engineered
              + 50 PCA DNABERT-2 dims
                         |
              Step 4: Fusion XGBoost
              5-fold x 3-repeat CV
              -> unified metrics
                         |
         +-------+-------+-------+
         |       |       |       |
   Step 5:  Step 6:  Step 7:  Advanced:
   Pathway  Drug     Resist.  GNN, PK/PD,
   Enrich.  Annot.           RL, Immune,
                             Twin, CDx
```

---

## Development

```bash
# Install dev dependencies
make dev

# Run tests (528 tests, ~45 s)
make test

# Lint
make lint

# Type-check
make typecheck

# Coverage report
make test-cov
```

---

## Dependencies

| Package                  | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| `xgboost`                | Gradient-boosted gene importance         |
| `transformers` + `torch` | DNABERT-2 sequence embeddings            |
| `biopython`              | NCBI Entrez sequence retrieval           |
| `scikit-learn`           | Preprocessing, CV, cosine similarity     |
| `pandas` + `numpy`       | Data manipulation                        |
| `scipy`                  | Statistical tests (Spearman, log-rank)   |
| `matplotlib` + `seaborn` | Visualisation                            |
| `lifelines`              | Kaplan-Meier survival analysis           |
| `requests`               | API queries (OpenTargets, CIViC, CT.gov) |
| `pyarrow`                | Parquet file I/O for caching             |
| `click` + `rich`         | CLI interface                            |

---

## Data Source

**TCGA Pan-Cancer (PANCAN) HiSeq RNA-Seq** from the UCI Machine
Learning Repository:
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

- 801 samples x 20,531 genes
- 5 cancer types: BRCA, KIRC, COAD, LUAD, PRAD

---

## License

MIT [LICENSE](LICENSE)
