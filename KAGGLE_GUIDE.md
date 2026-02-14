#  Fusion Oncology on Kaggle

Quick guide to running Fusion Oncology on Kaggle datasets.

## Quick Start (5 minutes)

### Option 1: Use the Notebook Template

1. **Upload** `kaggle_notebook.ipynb` to Kaggle
2. **Enable GPU** (optional, speeds up DNABERT-2)
3. **Add Dataset**: TCGA Pan-Cancer Atlas
4. **Run All Cells**

### Option 2: Command Line in Kaggle

```python
# Install (with numpy compatibility fix)
!pip install -q "numpy<2.2" --force-reinstall
!pip install -q git+https://github.com/mytechnotalent/fusion_oncology.git

# IMPORTANT: Restart kernel after installation, then run:
!fusion-oncology run --top-k 5 --fuzz-iterations 10 --output-dir /kaggle/working/results
```

**Note:** If you see numpy/scipy errors, restart the kernel after installation.

## Compatible Kaggle Datasets

###  RECOMMENDED: Genomics of Drug Sensitivity in Cancer (GDSC)

**The BEST dataset for Fusion Oncology** - verified and tested!

**Kaggle Link:** [kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc](https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc)

**Why it's perfect:**
-  **1000+ cancer cell lines** across multiple cancer types
-  **Gene expression + mutations + copy number variations**
-  **Drug sensitivity data (IC50 values)** - validate your predictions!
-  **Perfect usability: 10.0/10** (4,617 downloads, 120 upvotes)
-  **4 clean files** ready for analysis
-  **Multi-omics** - showcases all Fusion Oncology features

**Quick start with GDSC:**
```python
# After adding GDSC dataset to your Kaggle notebook
import pandas as pd
import numpy as np

# Load all GDSC files (exact paths):
gdsc_main = "/kaggle/input/genomics-of-drug-sensitivity-in-cancer-gdsc/GDSC_DATASET.csv"
cell_lines = "/kaggle/input/genomics-of-drug-sensitivity-in-cancer-gdsc/Cell_Lines_Details.xlsx"
gdsc2 = "/kaggle/input/genomics-of-drug-sensitivity-in-cancer-gdsc/GDSC2-dataset.csv"
compounds = "/kaggle/input/genomics-of-drug-sensitivity-in-cancer-gdsc/Compounds-annotation.csv"

# Load main dataset
df = pd.read_csv(gdsc_main)

# Extract tissue types and features
y = df['TCGA_DESC'].dropna()  # Cancer tissue classification
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df.loc[y.index, numeric_cols]  # Feature matrix

print(f" Loaded {len(X)} samples x {X.shape[1]} features")
print(f" Cancer types: {y.nunique()}")
```

**What makes GDSC special:**
- Compare your drug recommendations against **actual drug sensitivity** (IC50 values)
- Identify genomic biomarkers that correlate with drug response
- Multi-cancer analysis showcases pan-cancer classification
- Pathway-level drug targeting information included

---

###  Other Dataset Options

#### 1. **Built-in UCI Dataset** (Default - No Upload Needed)

#### 1. **Built-in UCI Dataset** (Default - No Upload Needed)
- 801 samples, 5 cancer types (BRCA, KIRC, COAD, LUAD, PRAD)
- Works directly with built-in ingestion - **no Kaggle upload needed**
- From UCI ML Repository: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
```python
from fusion_oncology.data.ingestion import DataIngestion
from fusion_oncology.config import ProjectConfig

cfg = ProjectConfig()
ingestor = DataIngestion(cfg)
X, y = ingestor.get_patient_data()
```

#### 2. **BRCA Multi-Omics (TCGA)**
**Link:** [kaggle.com/datasets/samdemharter/brca-multiomics-tcga](https://www.kaggle.com/datasets/samdemharter/brca-multiomics-tcga)
- 705 breast cancer samples
- Mutations, copy numbers, gene expression, protein levels
- 2,894 downloads, 31 upvotes
-  Only breast cancer (single type)

#### 3. **Search Kaggle for More**
Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and search:
- `"TCGA gene expression"`
- `"TCGA RNA-seq"`
- `"cancer genome atlas"`
- `"pan-cancer"`

Look for datasets with:
-  Gene expression matrices (samples x genes)
-  Cancer type labels
-  500+ samples for robust analysis
-  CSV/TSV format

###  Alternative: Use Official TCGA Data Sources

If you can't find suitable datasets on Kaggle, download from these verified sources and upload to Kaggle:

1. **UCSC Xena Browser** - https://xenabrowser.net/datapages/
   - TCGA Pan-Cancer (PANCAN) cohort
   - Pre-normalized gene expression matrices
   - Download  Upload to Kaggle Datasets

2. **GDC Data Portal** - https://portal.gdc.cancer.gov/
   - Official NCI Genomic Data Commons
   - RNA-Seq expression data
   - Requires some preprocessing

3. **UCI ML Repository** (Already Integrated)
   - https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq
   - 801 samples, works out-of-the-box with Fusion Oncology
   - No upload needed - use built-in ingestion

###  Data Format Requirements

Your data should have:
- **Expression matrix** (samples x genes): CSV/TSV with genes as columns
- **Labels** (cancer types): Single column with cancer type per sample

Example:
```
# Expression data (X)
        Gene1    Gene2    Gene3    ...
Sample1  2.45     1.23     0.87    ...
Sample2  3.12     2.01     1.45    ...

# Labels (y)
Sample1    BRCA
Sample2    LUAD
```

## Full Pipeline Commands

### 1. Evidence Query
```python
!fusion-oncology evidence EGFR BRAF KRAS --log-level WARNING
```

### 2. Resistance Mechanisms
```python
!fusion-oncology resistance EGFR ALK
```

### 3. Digital Twin Simulation
```python
!fusion-oncology simulate --drug Osimertinib --efficacy 0.15 --days 180
```

### 4. Companion Diagnostics
```python
# Create patient mutation file
import json
with open('patient.json', 'w') as f:
    json.dump({
        "patient_id": "KGL-001",
        "cancer_type": "NSCLC",
        "mutations": [
            {"gene": "EGFR", "variant": "L858R", "vaf": 0.42},
            {"gene": "TP53", "variant": "R273H", "vaf": 0.38}
        ]
    }, f)

!fusion-oncology companion-dx patient.json
```

## Custom Dataset Integration

If using a non-standard Kaggle dataset:

```python
from pathlib import Path
import pandas as pd
from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.fusion import FusionEngine

# Load your data
X = pd.read_csv('/kaggle/input/your-dataset/expression.csv', index_col=0)
y = pd.read_csv('/kaggle/input/your-dataset/labels.csv').squeeze()

# Configure
cfg = ProjectConfig(
    top_k=10,
    fuzz_iterations=20,
    output_dir=Path('/kaggle/working/results')
)

# Run fusion analysis
engine = FusionEngine(cfg)
results = engine.fit(X, y).results

# View results
print(results[['Gene', 'Fusion_Index', 'Pathways', 'Drugs']])
```

## Expected Runtime

| Component               | Time (default) | Time (fast)             |
| ----------------------- | -------------- | ----------------------- |
| Data loading            | 10s            | 10s                     |
| XGBoost training        | 30s            | 15s (fewer trees)       |
| DNABERT-2 embedding     | 2-5min         | 1min (fewer iterations) |
| Pathway/drug annotation | 5s             | 5s                      |
| Clinical evidence       | 10-30s         | 10s (cached)            |
| Total                   | **~5-8 min**   | **~2-3 min**            |

## Output Files

Results saved to `/kaggle/working/results/`:
- `fusion_results.csv` - Main target ranking
- `fusion_report.html` - Interactive HTML report
- `figures/` - All publication-quality plots
- `simulation_trajectory.csv` - Digital twin outputs (if run)

## Examples & Kernels

Check out these public Kaggle kernels:
- **[Coming Soon]** Fusion Oncology: TCGA Pan-Cancer Analysis
- **[Coming Soon]** Drug Target Discovery with AI
- **[Coming Soon]** Digital Twin Tumor Simulations

## Tips for Kaggle

1. **Enable GPU** for faster DNABERT-2 inference
2. **Reduce `--fuzz-iterations`** (10-20 instead of 50) for demos
3. **Cache results** between runs to save API calls
4. **Use `--log-level WARNING`** to reduce output verbosity
5. **Save outputs** before session expires (download results/)

## Troubleshooting

### Out of Memory
```python
cfg = ProjectConfig(
    top_k=3,              # Reduce from 5
    fuzz_iterations=10,   # Reduce from 20
    xgb_n_estimators=30   # Reduce from 50
)
```

### Slow DNABERT-2
```python
# Use CPU-only mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### API Rate Limits
```python
# Skip clinical evidence for demo
!fusion-oncology run --skip-evidence
```

## Publication & Sharing

When you publish a Kaggle kernel using Fusion Oncology:

1.  **Star the repo**: [github.com/mytechnotalent/fusion_oncology](https://github.com/mytechnotalent/fusion_oncology)
2.  **Cite the tool** in your kernel
3.  **Tag** @mytechnotalent
4.  **Share your results** - we'd love to see them!

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

**Need help?** Open an issue on GitHub or ping me on Kaggle!

**License**: MIT
