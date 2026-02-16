# Fusion Oncology on Kaggle

Run Fusion Oncology's **44-module precision oncology platform** on Kaggle.
The pipeline trains a multi-modal XGBoost fusion model on drug sensitivity
features, 10 engineered distributional features, and PCA-compressed
DNABERT-2 sequence embeddings (768 to 50 dims), with intelligent class
merging, class-balanced sample weights, optional Optuna HPO, and repeated
stratified CV -- backed by **528 automated tests** across 37 test files.

---

## Quick Start (5 minutes)

### Option 1: Upload the Notebook

1. Upload `kaggle_notebook.ipynb` to Kaggle
2. Add the **GDSC dataset** (link below)
3. Enable GPU (optional -- speeds up DNABERT-2)
4. **Run All Cells**

The notebook is 62 cells (47 code, 15 markdown) covering 12 sections:
Setup, Configuration, GDSC Loading, EDA, Fusion Analysis, Target
Visualisation, Clinical Evidence, Resistance, Pathway Enrichment,
Digital Twin, Companion Diagnostics, and End-to-End Summary.

### Option 2: CLI in a Kaggle Code Cell

```python
# Install (with numpy compatibility fix)
!pip install -q "numpy<2.2" --force-reinstall
!pip install -q git+https://github.com/mytechnotalent/fusion_oncology.git

# IMPORTANT: Restart kernel after installation, then run:
!fusion-oncology run --top-k 5 --fuzz-iterations 10 \
    --output-dir /kaggle/working/results
```

**Note:** If you see numpy/scipy errors, restart the kernel after install.

---

## Recommended Dataset: GDSC

**Genomics of Drug Sensitivity in Cancer**

[kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc)

Why it works:

- 1,000+ cancer cell lines across multiple tissue types
- Gene expression, mutations, and copy-number variations
- Drug sensitivity data (IC50 values) for validation
- 4 clean files, 10.0/10 usability, 4,600+ downloads
- Multi-omics -- exercises every Fusion Oncology module

```python
import pandas as pd
import numpy as np

base = "/kaggle/input/genomics-of-drug-sensitivity-in-cancer-gdsc"
df = pd.read_csv(f"{base}/GDSC_DATASET.csv")

y = df["TCGA_DESC"].dropna()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df.loc[y.index, numeric_cols]

print(f"Loaded {len(X)} samples x {X.shape[1]} features")
print(f"Cancer types: {y.nunique()}")
```

---

## Other Dataset Options

### Built-in UCI Dataset (No Upload Needed)

801 samples, 5 cancer types (BRCA, KIRC, COAD, LUAD, PRAD).
Works directly with the built-in ingestion module.

```python
from fusion_oncology.data.ingestion import DataIngestion
from fusion_oncology.config import ProjectConfig

cfg = ProjectConfig()
ingestor = DataIngestion(cfg)
X, y = ingestor.get_patient_data()
```

### BRCA Multi-Omics (TCGA)

[kaggle.com/datasets/samdemharter/brca-multiomics-tcga](https://www.kaggle.com/datasets/samdemharter/brca-multiomics-tcga)

705 breast cancer samples with mutations, copy numbers, gene expression,
and protein levels. Single cancer type only.

### Search Kaggle

Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and search
for `TCGA gene expression`, `TCGA RNA-seq`, or `pan-cancer`. Look for
datasets with 500+ samples, gene expression matrices, and cancer type
labels in CSV/TSV format.

---

## What the Notebook Covers

The 62-cell notebook walks through the full platform:

| Section                 | Cells | What It Does                      |
| ----------------------- | ----- | --------------------------------- |
| 1 -- Setup              | 3-7   | Install, env vars, imports        |
| 2 -- Configuration      | 9-10  | Hyperparameters, print settings   |
| 3 -- Load GDSC          | 12-18 | Read files, pivot, build labels   |
| 4 -- EDA                | 20-23 | Bar, histogram, heatmap, pie      |
| 5 -- Fusion Analysis    | 25-29 | XGBoost + DNABERT-2 pipeline      |
| 6 -- Target Viz         | 31-32 | Fusion Index chart, scatter       |
| 7 -- Clinical Evidence  | 34-36 | Evidence scores, drug mapping     |
| 8 -- Resistance         | 38    | Resistance mechanisms             |
| 9 -- Pathway Enrichment | 40-41 | Pathway analysis + chart          |
| 10 -- Digital Twin      | 43-46 | Tumour simulation, plots          |
| 11 -- Companion Dx      | 48-54 | Patient profiling, treatment plan |
| 12 -- Summary           | 56-60 | End-to-end recap                  |

Every code cell is 25 lines or fewer. All lines are under 100
characters. No cross-cell variable leaks. Each plot creates its own
`fig, ax` pair.

---

## CLI Commands for Kaggle

### Full Pipeline

```python
!fusion-oncology run --top-k 5 --fuzz-iterations 10 \
    --output-dir /kaggle/working/results
```

### Evidence Query

```python
!fusion-oncology evidence EGFR BRAF KRAS --log-level WARNING
```

### Resistance Mechanisms

```python
!fusion-oncology resistance EGFR ALK
```

### Digital Twin Simulation

```python
!fusion-oncology simulate --drug Osimertinib --efficacy 0.15 --days 180
```

### Companion Diagnostics

```python
import json

patient = {
    "patient_id": "KGL-001",
    "cancer_type": "NSCLC",
    "mutations": [
        {"gene": "EGFR", "variant": "L858R", "vaf": 0.42},
        {"gene": "TP53", "variant": "R273H", "vaf": 0.38},
    ],
}
with open("patient.json", "w") as f:
    json.dump(patient, f)

!fusion-oncology companion-dx patient.json
```

---

## Custom Dataset Integration

```python
from pathlib import Path
import pandas as pd
from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.fusion import FusionEngine

X = pd.read_csv("/kaggle/input/your-dataset/expression.csv", index_col=0)
y = pd.read_csv("/kaggle/input/your-dataset/labels.csv").squeeze()

cfg = ProjectConfig(
    top_k=10,
    fuzz_iterations=20,
    output_dir=Path("/kaggle/working/results"),
)

engine = FusionEngine(cfg)
results = engine.fit(X, y).results
print(results[["Gene", "Fusion_Index", "Pathways", "Drugs"]])
```

---

## Expected Runtime

| Component                 | Default     | Fast Mode   |
| ------------------------- | ----------- | ----------- |
| Data loading              | 10 s        | 10 s        |
| XGBoost baseline          | 30 s        | 15 s        |
| DNABERT-2 embedding       | 2-5 min     | 1 min       |
| Fusion model training     | 30 s        | 15 s        |
| Pathway + drug annotation | 5 s         | 5 s         |
| Clinical evidence         | 10-30 s     | 10 s        |
| **Total**                 | **5-8 min** | **2-3 min** |

---

## Output Files

Results saved to `/kaggle/working/results/`:

- `fusion_results.csv` -- main target ranking
- `fusion_report.html` -- interactive HTML report
- `figures/` -- publication-quality plots
- `simulation_trajectory.csv` -- digital twin outputs
- `companion_dx_*.txt` -- companion diagnostic reports

---

## Tips

1. Enable **GPU** for faster DNABERT-2 inference
2. Reduce `--fuzz-iterations` to 10-20 for demos
3. Use `--log-level WARNING` to reduce output verbosity
4. Save outputs before session expires (download `/kaggle/working/`)
5. Restart the kernel after `pip install` to avoid numpy conflicts

---

## Troubleshooting

### Out of Memory

```python
cfg = ProjectConfig(
    top_k=3,
    fuzz_iterations=10,
    xgb_n_estimators=30,
)
```

### Slow DNABERT-2

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### API Rate Limits

```python
!fusion-oncology run --skip-evidence
```

---

## Citation

```bibtex
@software{fusion_oncology,
  title={Fusion Oncology: Multi-Modal AI for Cancer Target Discovery},
  author={Thomas, Kevin},
  year={2026},
  url={https://github.com/mytechnotalent/fusion_oncology}
}
```

---

**Need help?** Open an issue on [GitHub](https://github.com/mytechnotalent/fusion_oncology).

**License:** MIT
