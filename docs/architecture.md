# Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              CLI (click)                                 │
│  fusion-oncology run | ingest | report | evidence | resistance |         │
│                  simulate | companion-dx | clear-cache                   │
└───────────┬──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FusionEngine (7-step orchestrator)                  │
│                                                                          │
│  1. DataIngestion    ──►  TCGA Pan-Cancer expression matrix              │
│  2. XGBoostEngine    ──►  Top-K gene importance                          │
│  3. DNABERTEngine    ──►  Sequence embeddings + instability scoring      │
│  4. PathwayEnrich    ──►  KEGG/Reactome pathway membership               │
│  5. DrugTargetMapper ──►  Druggability + approved drug annotation        │
│  6. ResistancePredict──►  Resistance mechanisms + risk scoring           │
│  7. NetworkPharmacol ──►  Interaction graph + synthetic lethality        │
└───────────┬──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Downstream Analysis Modules                          │
│                                                                          │
│  ClinicalEvidence    ──►  OpenTargets + CIViC + ClinicalTrials.gov       │
│  NeoantigenPredictor ──►  Mutant peptides + MHC-I binding scoring        │
│  CRISPRDesigner      ──►  Guide library with on/off-target scores        │
│  CompanionDiagnostic ──►  Patient profiling + AMP/ASCO/CAP tiering       │
│  DigitalTwin         ──►  Gompertzian ODE tumour simulation + RECIST     │
│  MultiOmicsIntegrator──►  MAF + CNA + methylation feature fusion         │
│  Visualization       ──►  Figures + HTML report                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Module Map

| Module                             | Responsibility                                                |
| ---------------------------------- | ------------------------------------------------------------- |
| `config.py`                        | Dataclass holding all tuneable parameters (29 fields)         |
| `data/ingestion.py`                | Downloads TCGA ZIP, extracts CSVs, caches as Parquet          |
| `data/preprocessing.py`            | Variance filtering, log-normalisation, PCA                    |
| `data/cache.py`                    | File-system artefact cache (DataFrames, arrays, bytes, JSON)  |
| `data/multi_omics.py`              | MAF mutation, CNA, methylation loading + feature integration  |
| `models/xgboost_engine.py`         | XGBoost training, cross-validation, feature importance        |
| `models/dnabert_engine.py`         | DNABERT-2 loading, single & batch embedding                   |
| `models/fusion.py`                 | 7-step pipeline orchestrating all analysis layers             |
| `models/crispr.py`                 | PAM scanning, on/off-target scoring, guide library export     |
| `models/companion_dx.py`           | Patient profiling, drug matching, AMP/ASCO/CAP tiering        |
| `models/digital_twin.py`           | Gompertzian ODE tumour simulation, RECIST, regimen comparison |
| `analysis/instability.py`          | Random point-mutation + cosine-drift scoring                  |
| `analysis/pathway.py`              | Curated pathway lookup + enrichment summary                   |
| `analysis/drug_target.py`          | Curated drug-target database + annotation                     |
| `analysis/survival.py`             | Kaplan-Meier + log-rank (requires clinical data)              |
| `analysis/clinical_evidence.py`    | OpenTargets, CIViC, ClinicalTrials.gov API queries            |
| `analysis/synthetic_lethality.py`  | 24 curated SL pairs + expression anti-correlation screening   |
| `analysis/neoantigen.py`           | Codon translation, peptide generation, MHC-I binding          |
| `analysis/resistance.py`           | 12-gene resistance mechanism database + risk scoring          |
| `analysis/network_pharmacology.py` | Drug→gene→pathway graph, centrality, polypharmacology         |
| `viz/plots.py`                     | Matplotlib/Seaborn plot functions                             |
| `viz/report.py`                    | Self-contained HTML report generator                          |
| `utils/bio.py`                     | NCBI Entrez fetch, GC content, CpG island detection           |
| `utils/log.py`                     | Logging configuration                                         |
| `cli.py`                           | Click CLI entry-point (8 commands)                            |

## Data Flow

```
UCI Archive (ZIP)
       │
       ▼
  DataIngestion                ArtifactCache
  ├─ download ──────────────► .cache/fusion_oncology/
  ├─ parse CSV                  (parquet, npy, bin)
  └─ clean
       │
       ├──────► X (expression matrix)
       └──────► y (labels)
                │
                ▼
          XGBoostEngine
          ├─ fit(X, y)
          └─ top_genes() ──► {gene: importance}
                                    │
                              ┌─────┘
                              ▼
                    NCBI Entrez (fetch_gene_sequence)
                              │
                              ▼
                    DNABERTEngine.embed(seq) ──► ref embedding
                              │
                              ▼
                    InstabilityAnalyzer.score()
                    (N random mutations → cosine drift)
                              │
                              ▼
                    Fusion Index = importance x instability x 1000
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    PathwayEnrich      DrugTargetMapper    ResistancePredictor
    (annotate)         (annotate)          (annotate)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                    InteractionNetwork + SyntheticLethalityDetector
                    (graph analysis + SL partner screening)
                              │
                    ┌─────────┼──────────────┐
                    ▼         ▼              ▼
              CRISPRDesigner CompanionDx   DigitalTwin
              (guide libs)   (treatment)   (simulation)
                              │
                    ┌─────────┼──────────────┐
                    ▼         ▼              ▼
             ClinicalEvidence  Neoantigen  MultiOmics
             (API queries)     (peptides)  (feature fusion)
                              │
                              ▼
                        Visualization
                        (plots + HTML report)
```

## External API Integration

| API                | Endpoint                                      | Purpose                             |
| ------------------ | --------------------------------------------- | ----------------------------------- |
| OpenTargets        | `api.platform.opentargets.org/api/v4/graphql` | Gene-disease associations, scores   |
| CIViC              | `civicdb.org/api/graphql`                     | Clinical interpretation of variants |
| ClinicalTrials.gov | `clinicaltrials.gov/api/v2/studies`           | Active clinical trial matching      |
| NCBI Entrez        | `eutils.ncbi.nlm.nih.gov/entrez/eutils/`      | RefSeq gene sequence retrieval      |

## Key Design Decisions

1. **Lazy model loading** - DNABERT-2 is only loaded when the first
   embedding is requested, keeping CLI startup fast.

2. **Disk caching** - The ~80 MB TCGA download is cached as raw bytes;
   cleaned DataFrames are cached as Parquet.  Cache can be cleared via
   `fusion-oncology clear-cache`.

3. **Offline pathway / drug / resistance databases** - Curated dictionaries
   are embedded in the source so the tool works without network access
   beyond the initial data download and Entrez lookups.  Clinical evidence
   and trial queries are optional enhancements.

4. **Graceful degradation** - Missing packages (`lifelines`) or failed
   network calls produce warnings rather than hard crashes.

5. **Composable architecture** - Each analysis module exposes both a
   standalone API and an `annotate(df)` method for pipeline integration,
   letting modules be used independently or orchestrated through FusionEngine.

6. **Patient-centric design** - CompanionDiagnostic + DigitalTwin enable
   translating population-level genomic findings into individualised
   treatment strategies with AMP/ASCO/CAP-tiered actionability.
