![image](https://github.com/mytechnotalent/fusion_oncology/blob/main/fusion_oncology.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Fusion Oncology

Fusion Oncology is an end-to-end research pipeline that transforms drug sensitivity data into standardized AMP/ASCO/CAP-tiered reports. It builds a **production-grade multi-modal fusion model** — a single XGBoost classifier trained on drug sensitivity features, 10 engineered distributional features, and sensitivity-weighted DNABERT-2 sequence embeddings — to prioritise therapeutic targets. The pipeline includes intelligent class merging, optional Optuna Bayesian hyperparameter optimization, and repeated stratified cross-validation for maximum metric stability.

---

## What It Does

Fusion Oncology builds a **production-grade multi-modal fusion model** that combines three signal sources into a single learned classifier:

| Component        | Method                                          | What it captures                                           |
| ---------------- | ----------------------------------------------- | ---------------------------------------------------------- |
| **Drug sens.**   | XGBoost on GDSC LN_IC50 features                | Which genes best discriminate cancer types                 |
| **Engineered**   | 10 row-level distributional features per sample | Per-sample mean, std, skew, kurtosis, IQR, CV              |
| **Genomic ctx.** | DNABERT-2 768-dim embeddings × drug sensitivity | Sensitivity-weighted structural gene context               |
| **Fusion model** | XGBoost on concatenated (N + 10 + 768) features | Jointly learned drug sensitivity + distribution + sequence |

The pipeline includes:
- **Intelligent class merging** — rare cancer types (< 40 samples) → "OTHER"
- **Repeated stratified CV** — 5-fold × 3 repeats for stable metric estimates
- **Optional Optuna HPO** — 50-trial Bayesian hyperparameter search

The fusion model produces **one unified set of CV metrics** (Accuracy, Precision, Recall, F1, F2, ROC AUC). The **Fusion Index** (importance × instability × 1000) ranks targets that are both biologically important *and* structurally vulnerable.

### Core Analysis Layers

- **Pathway enrichment** -- maps targets to PI3K-Akt, MAPK, p53, Wnt, Notch, cell-cycle, DNA-repair, apoptosis, angiogenesis, and immune-checkpoint pathways
- **Drug-target mapping** -- cross-references 20+ gene targets against approved oncology drugs (EGFROsimertinib, BRAFVemurafenib, KRASSotorasib, etc.)
- **Survival analysis** -- Kaplan-Meier + log-rank stratification hooks (requires clinical columns)
- **Publication-quality figures** -- bar charts, scatter plots, heatmaps, box plots
- **Self-contained HTML report** -- one-click shareable with collaborators

### Advanced Therapeutic Intelligence

- **Multi-omics integration** -- MAF mutation parsing, copy-number alteration analysis, methylation profiling, and combined feature matrix construction
- **Clinical evidence aggregation** -- real-time queries to OpenTargets, CIViC, and ClinicalTrials.gov APIs with composite evidence scoring
- **Synthetic lethality detection** -- curated database of 24 SL pairs (BRCA1/2-PARP, RB1-Aurora kinase, etc.) plus expression-based anti-correlation screening
- **Neoantigen prediction** -- codon translation, mutant peptide generation, simplified MHC-I binding scoring for immunotherapy candidate ranking
- **Resistance prediction** -- 12-gene resistance mechanism database with risk scoring, drug-specific evasion strategies, and clinical counter-measures
- **Network pharmacology** -- druggenepathway tripartite interaction graph with degree/betweenness centrality, polypharmacology scoring, and combination target identification
- **CRISPR guide design** -- PAM scanning on both strands, Doench-inspired on-target scoring, off-target heuristics, and exportable guide libraries
- **Companion diagnostics** -- patient-level mutation profiling, AMP/ASCO/CAP actionability tiering (Tiers I-IV), drug matching, and ranked treatment plans
- **Digital twin simulation** -- Gompertzian tumour growth ODE model with drug regimens (cycling/scheduling), immune dynamics, RECIST response classification, and regimen comparison

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

# Just download & cache the data
fusion-oncology ingest

# Query clinical evidence for a gene
fusion-oncology evidence --gene EGFR

# Check resistance mechanisms
fusion-oncology resistance --gene BRAF

# Run a digital twin tumour simulation
fusion-oncology simulate --days 180

# Run companion diagnostics
fusion-oncology companion-dx --mutations "EGFR:T790M,BRAF:V600E" --cancer-type NSCLC

# Regenerate a report from saved results
fusion-oncology report results/fusion_results.csv
```

###  Kaggle Integration

Run on **Kaggle** with the **GDSC dataset** (1,000+ cancer cell lines):

```bash
# Upload kaggle_notebook.ipynb to Kaggle
# Add GDSC dataset from: kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc
# Then run the notebook or use CLI:
!pip install git+https://github.com/mytechnotalent/fusion_oncology.git
!fusion-oncology run --top-k 10 --output-dir /kaggle/working/results
```

**See**: [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for complete GDSC setup and alternative datasets

---

## Project Structure

```
fusion_oncology/
 pyproject.toml                  # Package metadata & dependencies
 Makefile                        # Dev shortcuts (make test, make lint, ...)
 install.sh                      # One-step venv + editable install
 .github/workflows/ci.yml        # GitHub Actions CI
 docs/
    architecture.md             # System design & data-flow diagrams
 src/fusion_oncology/
    __init__.py
    config.py                   # Central dataclass configuration (29 fields)
    cli.py                      # Click CLI (run, ingest, report, evidence, resistance, simulate, companion-dx, clear-cache)
    data/
       ingestion.py            # TCGA download + ZIP extraction + caching
       preprocessing.py        # Variance filter, log-norm, PCA
       cache.py                # File-system artefact cache
       multi_omics.py          # MAF mutations, CNA, methylation, feature integration
    models/
       xgboost_engine.py       # XGBoost training & feature importance
       dnabert_engine.py       # DNABERT-2 sequence embedding
       fusion.py               # True multi-modal fusion (XGBoost + DNABERT-2)
       crispr.py               # CRISPR guide design, on/off-target scoring
       companion_dx.py         # Companion diagnostics & treatment planning
       digital_twin.py         # Gompertzian tumour growth simulation
    analysis/
       instability.py          # Mutation fuzzing + cosine drift
       pathway.py              # KEGG/Reactome pathway lookup
       drug_target.py          # Drug-target annotation
       survival.py             # Kaplan-Meier survival hooks
       clinical_evidence.py    # OpenTargets + CIViC + ClinicalTrials.gov
       synthetic_lethality.py  # SL pair detection + expression screening
       neoantigen.py           # Peptide generation + MHC binding scoring
       resistance.py           # Resistance mechanism prediction
       network_pharmacology.py # Drug-gene-pathway interaction network
    viz/
       plots.py                # Matplotlib/Seaborn figure functions
       report.py               # HTML report generator
    utils/
        bio.py                  # Entrez fetch, GC content, CpG islands
        log.py                  # Logging setup
 tests/
     conftest.py                 # Shared fixtures
     test_config.py
     test_preprocessing.py
     test_xgboost_engine.py
     test_cache.py
     test_pathway.py
     test_drug_target.py
     test_bio.py
     test_plots.py
     test_report.py
     test_multi_omics.py
     test_clinical_evidence.py
     test_synthetic_lethality.py
     test_neoantigen.py
     test_resistance.py
     test_network_pharmacology.py
     test_crispr.py
     test_companion_dx.py
     test_digital_twin.py
```

---

## CLI Reference & Examples

### All Available Commands

```bash
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

Query clinical evidence from OpenTargets, CIViC, and ClinicalTrials.gov for target genes.

```bash
$ fusion-oncology evidence EGFR BRAF
```

**Output:**
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

**Options:**
- `GENES...` -- One or more gene symbols (space-separated)
- `--log-level` -- Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

---

### 2. Resistance Mechanisms

Predict resistance mechanisms and evasion strategies for drug targets.

```bash
$ fusion-oncology resistance EGFR ALK
```

**Output:**
```
Resistance Profile for EGFR:

Mechanisms:
  1. T790M Gatekeeper Mutation
     Risk: HIGH | Clinical: Very common in osimertinib resistance
     Strategy: Switch to 3rd-gen TKI (osimertinib) or 4th-gen EGFR-selective

  2. MET Amplification
     Risk: MEDIUM | Clinical: Bypass signaling in 5-20% of cases
     Strategy: Combine EGFR-TKI with MET inhibitor (capmatinib, tepotinib)

  3. HER2 Amplification
     Risk: MEDIUM | Clinical: Emerging bypass in 5-12% of EGFR-TKI treated
     Strategy: Dual EGFR-HER2 blockade or switch to HER2-directed therapy

  4. BRAF V600E Mutation
     Risk: LOW | Clinical: Rare transformation escape mechanism
     Strategy: Add BRAF inhibitor to EGFR-TKI regimen

Resistance Profile for ALK:

Mechanisms:
  1. ALK G1202R Gatekeeper Mutation
     Risk: HIGH | Clinical: Most common in lorlatinib resistance
     Strategy: Investigational 4th-gen ALK inhibitor or TPX-0131

  2. ALK L1196M Mutation
     Risk: HIGH | Clinical: Resistant to 1st/2nd-gen but sensitive to lorlatinib
     Strategy: Switch to 3rd-gen lorlatinib if not already on it

  3. EGFR Amplification
     Risk: MEDIUM | Clinical: Bypass pathway activation in 5-15%
     Strategy: Dual ALK-EGFR inhibition combination therapy
```

**Options:**
- `GENES...` -- One or more gene symbols (space-separated)

---

### 3. Digital Twin Simulation

Simulate tumour growth dynamics with drug regimens using a Gompertzian ODE model.

```bash
$ fusion-oncology simulate --drug Osimertinib --efficacy 0.15 --days 90
```

**Output:**
```
Digital Twin Tumour Simulation

Configuration:
  Initial Volume: 10000.0 mm3
  Initial CTCs: 100 cells/mL
  Simulation Duration: 90 days
  Drug: Osimertinib
  Efficacy: 0.15 day
  Days On/Off: 0/0 (continuous)

Running ODE integration...

Final Results (Day 90):
  Tumour Volume: < 0.001 mm3
  Reduction: 100.0%
  CTCs: < 0.001 cells/mL
  RECIST Response: CR (Complete Response)

Trajectory saved to: digital_twin_trajectory.csv
```

**Options:**
- `--days INT` -- Simulation duration (default: 365)
- `--drug TEXT` -- Drug name (default: Generic)
- `--efficacy FLOAT` -- Drug efficacy rate (default: 0.1)
- `--days-on INT` -- Days drug is active in cycle (default: 0, continuous)
- `--days-off INT` -- Days drug is off in cycle (default: 0)
- `--initial-volume FLOAT` -- Initial tumour volume in mm3 (default: 10000.0)
- `--growth-rate FLOAT` -- Gompertzian growth rate (default: 0.01)
- `--carrying-capacity FLOAT` -- Maximum tumour volume (default: 100000.0)
- `--output PATH` -- Output CSV file path

---

### 4. Companion Diagnostics

Generate comprehensive treatment recommendations for patient mutation profiles.

```bash
# Create patient mutation file
$ cat > patient_mutations.json << EOF
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

**Output:**
```
=== COMPANION DIAGNOSTIC REPORT ===

Patient: PT-2024-001
Cancer Type: Non-Small Cell Lung Cancer
Analysis Date: 2026-02-14

Detected Mutations:
   EGFR L858R (VAF: 42.0%) - Tier I: Strong clinical significance
   TP53 R273H (VAF: 38.0%) - Tier III: Potential clinical significance
   KRAS G12C (VAF: 15.0%) - Tier I: Strong clinical significance

Treatment Recommendations (Ranked by Confidence):

1. Osimertinib (EGFR L858R) -- 90% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: FDA-approved for EGFR-mutant NSCLC, first-line standard
   Rationale: Strong clinical benefit in L858R-positive patients

2. Sotorasib (KRAS G12C) -- 78% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: FDA-approved for KRAS G12C NSCLC
   Rationale: Selective KRAS G12C inhibitor, response rate ~37%

3. Afatinib (EGFR L858R) -- 72% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: 2nd-gen EGFR TKI, proven efficacy in L858R
   Rationale: Alternative to osimertinib, pan-HER inhibitor

4. Erlotinib (EGFR L858R) -- 65% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: 1st-gen EGFR TKI, established NSCLC therapy
   Rationale: Historical standard, now 2nd/3rd line

5. Adagrasib (KRAS G12C) -- 62% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: FDA-approved KRAS G12C inhibitor
   Rationale: Alternative to sotorasib, CNS penetration

6. Gefitinib (EGFR L858R) -- 58% confidence
   AMP/ASCO/CAP Tier: I
   Evidence: 1st-gen EGFR TKI
   Rationale: Similar to erlotinib, regional variation

7. Carboplatin/Pemetrexed (TP53 R273H) -- 35% confidence
   AMP/ASCO/CAP Tier: III
   Evidence: Chemotherapy backbone for NSCLC
   Rationale: Standard option for targetable cases

8. Pembrolizumab (TP53 R273H) -- 27% confidence
   AMP/ASCO/CAP Tier: III
   Evidence: Immune checkpoint inhibitor
   Rationale: Consider if PD-L1 high, TP53 may predict response

Resistance Risk Alerts:

    EGFR T790M Gatekeeper Mutation (HIGH risk)
      Monitor for acquired resistance during osimertinib therapy
      Strategy: Re-biopsy at progression, consider 4th-gen EGFR TKI

    MET Amplification (MEDIUM risk)
      Bypass pathway activation in EGFR-mutant cases
      Strategy: Combination EGFR-MET inhibition if MET amplified

    HER2 Amplification (MEDIUM risk)
      Emerging resistance mechanism in 5-12% of EGFR TKI-treated
      Strategy: HER2-directed therapy or dual blockade

    BRAF V600E Mutation (LOW risk)
      Rare transformation mechanism
      Strategy: BRAF inhibitor addition if detected

    STK11 Loss (MEDIUM risk)
      May confer immune checkpoint inhibitor resistance
      Strategy: Avoid ICI monotherapy, prefer targeted agents

    KEAP1 Mutation (MEDIUM risk)
      Associated with poor ICI response
      Strategy: Targeted therapy preferred over immunotherapy

    PTEN Loss (LOW risk)
      May reduce TKI sensitivity
      Strategy: Monitor closely, consider PI3K pathway inhibition

    PIK3CA Activation (LOW risk)
      Bypass signaling mechanism
      Strategy: Dual EGFR-PI3K inhibition investigational

Synthetic Lethality Opportunities:

    PARP1 inhibition (if DDR pathway defects emerge)
      Associated genes: BRCA1, BRCA2
      Drugs: Olaparib, rucaparib, niraparib

    CHK1 inhibition (TP53 R273H synthetic lethal)
      Associated genes: TP53
      Drugs: Prexasertib, rabusertib (investigational)

    WEE1 inhibition (TP53 R273H synthetic lethal)
      Associated genes: TP53
      Drugs: Adavosertib (investigational)

    ATM inhibition (if ATR pathway intact)
      Associated genes: ATM
      Drugs: AZD0156, AZD1390 (investigational)

Clinical Summary:
  High-confidence actionable targets: 2 (EGFR, KRAS)
  FDA-approved matched therapies: 6
  Primary recommendation: Osimertinib monotherapy
  Secondary option: Sotorasib (if KRAS-driven)
  Resistance monitoring: Essential for EGFR pathway
```

**Options:**
- `MUTATIONS_FILE` -- JSON file with patient mutations (required)
- `--log-level` -- Set logging verbosity

---

### 5. Full Pipeline

Run the complete fusion analysis pipeline on the TCGA Pan-Cancer dataset.

```bash
# Full analysis (default: top 5 genes, 20 mutation iterations)
$ fusion-oncology run

# Fast smoke test
$ fusion-oncology run --top-k 3 --fuzz-iterations 5

# High-resolution analysis
$ fusion-oncology run --top-k 20 --fuzz-iterations 50 --xgb-trees 200
```

**Output:**
```
Step 1: Training XGBoost baseline on drug sensitivity features...
  Top 5 genes: TP53, BRCA1, KRAS, EGFR, PTEN

Step 2: Computing DNABERT-2 embeddings (768-dim) for top genes...
  TP53: embedded | instability = 0.0823
  BRCA1: embedded | instability = 0.1156
  KRAS: embedded | instability = 0.0647

Step 3: Building fusion features (sensitivity-weighted embeddings)...
  Concatenated 768 DNABERT-2 dims with original features

Step 4: Training fusion XGBoost on combined feature space...
  Fusion Model CV Metrics (5-fold stratified):
    Accuracy:  0.9850 +/- 0.0045
    Precision: 0.9812 +/- 0.0063
    Recall:    0.9793 +/- 0.0051
    F1-Score:  0.9801 +/- 0.0056
    ROC AUC:   0.9987 +/- 0.0008

Step 5: Pathway enrichment + Drug annotation + Resistance prediction...

Final Fusion Index:
  1. TP53: 2,847.5
  2. BRCA1: 2,456.2
  3. KRAS: 1,823.6
  4. EGFR: 2,891.3
  5. PTEN: 1,654.8

Results saved to: results/fusion_results.csv
Report generated: results/fusion_report.html
Figures saved to: results/figures/
```

**Options:**
- `--top-k INT` -- Number of top genes to analyze (default: 5)
- `--fuzz-iterations INT` -- Mutation iterations per gene (default: 20)
- `--xgb-trees INT` -- XGBoost estimators (default: 50)
- `--xgb-depth INT` -- XGBoost max depth (default: 4)
- `--output-dir PATH` -- Output directory (default: results/)
- `--log-level TEXT` -- Logging level (DEBUG, INFO, WARNING, ERROR)
- `--skip-plots` -- Skip figure generation
- `--skip-report` -- Skip HTML report generation

---

### 6. Data Ingestion

Download and cache the TCGA Pan-Cancer dataset.

```bash
$ fusion-oncology ingest
```

**Output:**
```
Downloading TCGA Pan-Cancer dataset...
Source: UCI ML Repository
Size: ~69 MB

Download complete: ~/.cache/fusion_oncology/
Extracted: 801 samples x 20,531 genes
Cache ready for analysis
```

---

### 7. Report Generation

Regenerate HTML report from previously saved results.

```bash
$ fusion-oncology report results/fusion_results.csv
```

**Output:**
```
Loading results from: results/fusion_results.csv
Generating HTML report...
Report saved to: results/fusion_report.html
```

---

### 8. Cache Management

Clear all locally cached data and model artefacts.

```bash
$ fusion-oncology clear-cache
```

**Output:**
```
Clearing cache directory: ~/.cache/fusion_oncology/
Removed: 245 MB
Cache cleared successfully
```

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
                    For each cell line:
                    weight embeddings by
                    drug sensitivity values
                    -> 768-dim context vector
                         |
                  Concatenate: original
                  features + 768 DNABERT-2
                         |
              Step 4: Fusion XGBoost
              Train on (N + 768) features
              5-fold stratified CV
              -> ONE set of metrics
                         |
         +---------------+---------------+
         |               |               |
   Step 5: Pathway  Step 6: Drug   Step 7:
   Enrichment       Annotation     Resistance
                         |
         +---------------+---------------+
         |               |               |
    CRISPR Guide   Companion Dx    Digital Twin
    Design         (AMP/ASCO/CAP)  (Gompertz ODE)
```

---

## Development

```bash
# Install dev dependencies
make dev

# Run tests
make test

# Run fast tests only (no network / model loading)
make test-fast

# Lint
make lint

# Type-check
make typecheck

# Coverage report
make test-cov
```

---

## Dependencies

| Package                  | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| `xgboost`                | Gradient-boosted gene importance                   |
| `transformers` + `torch` | DNABERT-2 sequence embeddings                      |
| `biopython`              | NCBI Entrez sequence retrieval                     |
| `scikit-learn`           | Preprocessing, cross-validation, cosine similarity |
| `pandas` + `numpy`       | Data manipulation                                  |
| `scipy`                  | Statistical tests (Spearman, log-rank)             |
| `matplotlib` + `seaborn` | Visualization                                      |
| `lifelines`              | Kaplan-Meier survival analysis                     |
| `requests`               | API queries (OpenTargets, CIViC, ClinicalTrials)   |
| `pyarrow`                | Parquet file I/O for artefact caching              |
| `click` + `rich`         | CLI interface                                      |

---

## Data Source

**TCGA Pan-Cancer (PANCAN) HiSeq RNA-Seq** from the UCI Machine Learning Repository:  
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

- 801 samples x 20,531 genes
- 5 cancer types: BRCA, KIRC, COAD, LUAD, PRAD

---

## License

MIT [LICENSE](LICENSE)
