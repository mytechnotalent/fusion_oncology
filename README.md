![image](https://github.com/mytechnotalent/fusion_oncology/blob/main/fusion_oncology.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Fusion Oncology

Comprehensive precision oncology platform fusing XGBoost feature importance, DNABERT-2 sequence embeddings, multi-omics integration, clinical evidence aggregation, and computational pharmacology to identify, validate, and prioritise high-impact therapeutic strategies for cancer.

---

## What It Does

Fusion Oncology combines two complementary signals to rank cancer gene targets:

| Signal      | Method                                 | What it captures                              |
| ----------- | -------------------------------------- | --------------------------------------------- |
| **Dynamic** | XGBoost on TCGA Pan-Cancer RNA-Seq     | Which genes best discriminate cancer types    |
| **Static**  | DNABERT-2 embedding + mutation fuzzing | How structurally fragile a gene's sequence is |

The **Fusion Index** (importance × instability × 1000) highlights genes that are both biologically important *and* structurally vulnerable — promising candidates for therapeutic intervention.

### Core Analysis Layers

- **Pathway enrichment** — maps targets to PI3K-Akt, MAPK, p53, Wnt, Notch, cell-cycle, DNA-repair, apoptosis, angiogenesis, and immune-checkpoint pathways
- **Drug-target mapping** — cross-references 20+ gene targets against approved oncology drugs (EGFR→Osimertinib, BRAF→Vemurafenib, KRAS→Sotorasib, etc.)
- **Survival analysis** — Kaplan–Meier + log-rank stratification hooks (requires clinical columns)
- **Publication-quality figures** — bar charts, scatter plots, heatmaps, box plots
- **Self-contained HTML report** — one-click shareable with collaborators

### Advanced Therapeutic Intelligence

- **Multi-omics integration** — MAF mutation parsing, copy-number alteration analysis, methylation profiling, and combined feature matrix construction
- **Clinical evidence aggregation** — real-time queries to OpenTargets, CIViC, and ClinicalTrials.gov APIs with composite evidence scoring
- **Synthetic lethality detection** — curated database of 24 SL pairs (BRCA1/2–PARP, RB1–Aurora kinase, etc.) plus expression-based anti-correlation screening
- **Neoantigen prediction** — codon translation, mutant peptide generation, simplified MHC-I binding scoring for immunotherapy candidate ranking
- **Resistance prediction** — 12-gene resistance mechanism database with risk scoring, drug-specific evasion strategies, and clinical counter-measures
- **Network pharmacology** — drug→gene→pathway tripartite interaction graph with degree/betweenness centrality, polypharmacology scoring, and combination target identification
- **CRISPR guide design** — PAM scanning on both strands, Doench-inspired on-target scoring, off-target heuristics, and exportable guide libraries
- **Companion diagnostics** — patient-level mutation profiling, AMP/ASCO/CAP actionability tiering (Tiers I–IV), drug matching, and ranked treatment plans
- **Digital twin simulation** — Gompertzian tumour growth ODE model with drug regimens (cycling/scheduling), immune dynamics, RECIST response classification, and regimen comparison

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

---

## Project Structure

```
fusion_oncology/
├── pyproject.toml                  # Package metadata & dependencies
├── Makefile                        # Dev shortcuts (make test, make lint, …)
├── install.sh                      # One-step venv + editable install
├── .github/workflows/ci.yml        # GitHub Actions CI
├── docs/
│   └── architecture.md             # System design & data-flow diagrams
├── src/fusion_oncology/
│   ├── __init__.py
│   ├── config.py                   # Central dataclass configuration (29 fields)
│   ├── cli.py                      # Click CLI (run, ingest, report, evidence, resistance, simulate, companion-dx, clear-cache)
│   ├── data/
│   │   ├── ingestion.py            # TCGA download + ZIP extraction + caching
│   │   ├── preprocessing.py        # Variance filter, log-norm, PCA
│   │   ├── cache.py                # File-system artefact cache
│   │   └── multi_omics.py          # MAF mutations, CNA, methylation, feature integration
│   ├── models/
│   │   ├── xgboost_engine.py       # XGBoost training & feature importance
│   │   ├── dnabert_engine.py       # DNABERT-2 sequence embedding
│   │   ├── fusion.py               # 7-step orchestrator combining all models
│   │   ├── crispr.py               # CRISPR guide design, on/off-target scoring
│   │   ├── companion_dx.py         # Companion diagnostics & treatment planning
│   │   └── digital_twin.py         # Gompertzian tumour growth simulation
│   ├── analysis/
│   │   ├── instability.py          # Mutation fuzzing + cosine drift
│   │   ├── pathway.py              # KEGG/Reactome pathway lookup
│   │   ├── drug_target.py          # Drug–target annotation
│   │   ├── survival.py             # Kaplan–Meier survival hooks
│   │   ├── clinical_evidence.py    # OpenTargets + CIViC + ClinicalTrials.gov
│   │   ├── synthetic_lethality.py  # SL pair detection + expression screening
│   │   ├── neoantigen.py           # Peptide generation + MHC binding scoring
│   │   ├── resistance.py           # Resistance mechanism prediction
│   │   └── network_pharmacology.py # Drug-gene-pathway interaction network
│   ├── viz/
│   │   ├── plots.py                # Matplotlib/Seaborn figure functions
│   │   └── report.py               # HTML report generator
│   └── utils/
│       ├── bio.py                  # Entrez fetch, GC content, CpG islands
│       └── log.py                  # Logging setup
└── tests/
    ├── conftest.py                 # Shared fixtures
    ├── test_config.py
    ├── test_preprocessing.py
    ├── test_xgboost_engine.py
    ├── test_cache.py
    ├── test_pathway.py
    ├── test_drug_target.py
    ├── test_bio.py
    ├── test_plots.py
    ├── test_report.py
    ├── test_multi_omics.py
    ├── test_clinical_evidence.py
    ├── test_synthetic_lethality.py
    ├── test_neoantigen.py
    ├── test_resistance.py
    ├── test_network_pharmacology.py
    ├── test_crispr.py
    ├── test_companion_dx.py
    └── test_digital_twin.py
```

---

## CLI Reference

```
Usage: fusion-oncology [OPTIONS] COMMAND [ARGS]...

Commands:
  run            Run the full 7-step fusion analysis pipeline
  ingest         Download and cache the TCGA dataset
  report         Regenerate HTML report from saved CSV
  evidence       Query clinical evidence databases for a gene
  resistance     Predict resistance mechanisms for a gene
  simulate       Run digital twin tumour growth simulation
  companion-dx   Run companion diagnostic analysis on a patient profile
  clear-cache    Delete all locally cached artefacts

Options for `run`:
  --top-k INT              Number of top genes to analyse (default: 5)
  --fuzz-iterations INT    Mutation iterations per gene (default: 20)
  --xgb-trees INT          XGBoost estimators (default: 50)
  --xgb-depth INT          XGBoost max depth (default: 4)
  --output-dir PATH        Output directory (default: results/)
  --log-level LEVEL        DEBUG | INFO | WARNING | ERROR
  --skip-plots             Skip figure generation
  --skip-report            Skip HTML report generation

Options for `evidence`:
  --gene TEXT              Gene symbol to query (required)

Options for `resistance`:
  --gene TEXT              Gene symbol to query (required)

Options for `simulate`:
  --days INT               Simulation duration in days (default: 365)

Options for `companion-dx`:
  --mutations TEXT          Comma-separated gene:variant pairs (required)
  --cancer-type TEXT        Cancer type label (required)
```

---

## How the 7-Step Pipeline Works

```
                  TCGA Pan-Cancer RNA-Seq
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
  Step 1: XGBoost Classifier    Step 2: NCBI Entrez
        (multi-class cancer)    (RefSeq DNA fetch)
              │                       │
              ▼                       ▼
        Feature Importance      DNABERT-2 Embedding
        (gain score)            (768-dim vector)
              │                       │
              │              Step 3: N random SNP mutations
              │                  re-embed each mutant
              │                  cosine distance to ref
              │                       │
              ▼                       ▼
         importance              instability
              │                       │
              └───────── × ───────────┘
                         │
                         ▼
                   Fusion Index
                  (× 1000 scaling)
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
     Step 4: Pathway  Step 5:    Step 6:
     Enrichment       Drug-Target Resistance
                      Annotation  Prediction
                         │
                         ▼
              Step 7: Network Pharmacology
              + Synthetic Lethality Screening
                         │
                         ▼
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         CRISPR       Companion  Digital Twin
         Guide        Diagnostics Simulation
         Design       (AMP/ASCO/  (Gompertz ODE)
                       CAP Tiers)
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
| `lifelines`              | Kaplan–Meier survival analysis                     |
| `requests`               | API queries (OpenTargets, CIViC, ClinicalTrials)   |
| `pyarrow`                | Parquet file I/O for artefact caching              |
| `click` + `rich`         | CLI interface                                      |

---

## Data Source

**TCGA Pan-Cancer (PANCAN) HiSeq RNA-Seq** from the UCI Machine Learning Repository:  
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

- 801 samples × 20,531 genes
- 5 cancer types: BRCA, KIRC, COAD, LUAD, PRAD

---

## License

MIT [LICENSE](LICENSE)
