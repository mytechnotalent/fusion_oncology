# How the Fusion Oncology Model Works

---

## The Problem

Each GDSC (Genomics of Drug Sensitivity in Cancer) record has two kinds of information:

**Structured data:** Drug sensitivity numbers (LN_IC50 values) across hundreds of gene targets for 1,002 cancer cell lines. Each cell line has a cancer-type label (e.g., `BRCA`, `LUAD`, `COREAD`).

**DNA sequence data:** The actual nucleotide sequences (A/C/G/T) of genes, fetched from NCBI RefSeq.

We need to answer: **Which genes are the best therapeutic targets for cancer drugs?**

No single model handles both data types well. XGBoost is great with tabular drug-sensitivity numbers but can't read DNA. DNABERT-2 is great with DNA sequences but can't process tabular features. So we use both.

---

## The Models We Train (In Order)

We train two XGBoost classifiers, one after the other. DNABERT-2 is NOT trained — it's a pre-trained transformer we use as a frozen feature extractor.

```
MODEL 1:  XGBoost (Baseline)     trained FIRST, on drug-sensitivity features only
          DNABERT-2              NOT trained — used as a frozen feature extractor between models
MODEL 2:  XGBoost (Fusion)       trained SECOND, on drug-sensitivity + DNABERT-2 features combined
```

That's it. Two XGBoost classifiers. The first one goes first because Model 2 needs its output (the top gene list) to know which genes to embed with DNABERT-2.

---

## What the Data Looks Like Before Anything Happens

The raw GDSC dataset has rows like:

```
COSMIC_ID  |  TCGA_DESC  |  TARGET         |   LN_IC50
906826     |  BRCA       |  EGFR           |   2.14
906826     |  BRCA       |  BRAF, MAP2K1   |  -0.53
687983     |  LUAD       |  ALK            |   1.87
```

Each row = one cell line tested against one drug. `TARGET` is the gene(s) that drug hits. `LN_IC50` is how sensitive the cell line was (lower = more sensitive to the drug).

**Problem:** Multi-target entries like `"BRAF, MAP2K1"` exist. One row can have two genes.

---

## Data Preparation (Before Any Model Trains)

Here is exactly what happens, step by step.

### A. Explode multi-target entries

```
BEFORE:  COSMIC_ID=906826, TARGET="BRAF, MAP2K1", LN_IC50=-0.53

AFTER:   COSMIC_ID=906826, TARGET="BRAF",   LN_IC50=-0.53
         COSMIC_ID=906826, TARGET="MAP2K1", LN_IC50=-0.53
```

Every comma-separated target becomes its own row. The LN_IC50 value is duplicated.

### B. Pivot into a matrix

Rows become cell lines. Columns become genes. Values become mean LN_IC50.

```
              EGFR    BRAF    ALK    MAP2K1   ...  (hundreds of gene columns)
COSMIC 906826  2.14   -0.53   0.00   -0.53
COSMIC 687983  0.00    0.00   1.87    0.00
    ...
```

Missing values (cell line never tested against that gene's drug) are filled with `0.0`.

Shape: **~968 cell lines × ~200 gene columns** (after filtering).

### C. Build cancer-type labels

For each cell line, take the most frequent `TCGA_DESC` value as its label.

```
COSMIC 906826 → "BRCA"
COSMIC 687983 → "LUAD"
```

### D. Filter gene columns

Only keep columns whose names match a gene-symbol regex: `^[A-Z][A-Z0-9/-]*$`

This removes non-gene columns like `"Retinoic acid"` or `"others"` that XGBoost could train on but DNABERT-2 can't embed.

### E. Filter by coverage

Drop any gene column where fewer than 10 cell lines have a non-zero value. This removes genes with too little data.

### F. Merge rare cancer types

Any cancer type with fewer than 40 cell lines gets renamed to `"OTHER"`. If `"OTHER"` itself still has fewer than 40, drop those rows entirely.

**After this step** we have:

```
X = DataFrame of shape (968 cell lines, ~200 gene columns)
    Values are mean LN_IC50 drug-sensitivity scores.
    
y = Series of shape (968,)
    Values are cancer-type labels like "BRCA", "LUAD", "COREAD", etc.
```

These are frozen. Both models train on this same X and y.

### G. Feature engineering (10 extra columns)

For each cell line (row), compute 10 row-level statistics across all its gene columns:

```
Cell line 906826 has values: [2.14, -0.53, 0.0, -0.53, ...]

_row_mean   = mean of all values           →  0.27
_row_std    = standard deviation           →  0.94
_row_skew   = skewness (asymmetry)         →  1.12
_row_kurt   = kurtosis (tail heaviness)    →  0.45
_row_max    = maximum value                →  2.14
_row_min    = minimum value                → -0.53
_row_range  = max - min                    →  2.67
_row_median = median value                 →  0.00
_row_iqr    = 75th percentile - 25th       →  0.53
_row_cv     = std / |mean|                 →  3.48
```

These are appended as 10 extra columns. Now X has ~210 columns instead of ~200.

**Why?** Single-gene columns tell XGBoost "this cell line is sensitive to EGFR drugs." The row statistics tell it "this cell line is broadly sensitive across many targets" — distributional context that individual columns can't provide.

---

## MODEL 1: Baseline XGBoost

### What It Sees

The augmented feature matrix: ~968 rows × ~210 columns (gene sensitivity values + 10 statistical features). It never sees any DNA sequences.

The labels: cancer type strings encoded as integers (e.g., BRCA=0, LUAD=1, COREAD=2, ...).

### What It Does

**Step 1a: Optuna Hyperparameter Optimisation (HPO)**

Before training, Optuna runs 50 trials to find the best XGBoost hyperparameters.

```
For each of 50 trials:
    Optuna picks random hyperparameters:
        n_estimators:      random int in [300, 1500]
        max_depth:         random int in [4, 8]
        learning_rate:     random float in [0.01, 0.1] (log scale)
        min_child_weight:  random int in [1, 10]
        gamma:             random float in [0.0, 0.5]
        subsample:         random float in [0.6, 1.0]
        colsample_bytree:  random float in [0.5, 1.0]
        reg_alpha:         random float in [0.0001, 1.0] (log scale)
        reg_lambda:        random float in [0.5, 5.0]

    Train XGBoost with these params using 5-fold stratified CV
    Compute mean weighted F1 score across the 5 folds
    Optuna remembers this score

After 50 trials:
    Take the trial with the best F1 score
    Overwrite the config with those hyperparameters
```

This is Bayesian optimisation — Optuna doesn't pick purely randomly. After the first few trials, it uses past results to guess which parameter ranges are promising and focuses exploration there.

**Step 1b: Train the final baseline XGBoost**

```
Input:  X_augmented (968 × ~210), y_encoded (968 integers)

Class balancing:
    Compute sample weights inversely proportional to class frequency.
    If "BRCA" has 100 samples and "SCLC" has 20 samples out of 968 total:
        weight_BRCA = 968 / (n_classes × 100)
        weight_SCLC = 968 / (n_classes × 20) ← higher weight
    This prevents XGBoost from ignoring rare cancer types.

Train:
    xgb.XGBClassifier(
        n_estimators=<best from Optuna>,    # number of boosting rounds (trees)
        max_depth=<best from Optuna>,       # max tree depth
        learning_rate=<best from Optuna>,   # step size per tree
        ...all 9 params from Optuna...
        objective="multi:softprob",         # multi-class classification
        eval_metric="mlogloss",             # log-loss for multi-class
    ).fit(X_augmented, y_encoded, sample_weight=weights)

Each boosting round:
    1. Look at current prediction errors
    2. Build a new decision tree that corrects those errors
    3. Add that tree's predictions (scaled by learning_rate) to the ensemble
    4. Repeat for n_estimators rounds
```

**Step 1c: Cross-validate the baseline**

```
If dataset ≥ 200 samples:
    Use RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    = 15 total train/test splits
    
Else:
    Use StratifiedKFold(n_splits=5)
    = 5 total train/test splits

"Stratified" means each fold has approximately the same cancer-type 
distribution as the full dataset.

Metrics computed per fold:
    1. Accuracy         (% correct overall)
    2. Precision        (weighted average across classes)
    3. Recall           (weighted average across classes)
    4. F1 Score         (weighted harmonic mean of precision & recall)
    5. F2 Score         (weighted, emphasises recall more than F1)
    6. ROC AUC          (one-vs-rest, weighted)

Final result: mean ± std for each metric across all 15 folds.
```

**Step 1d: Extract top-K gene importances**

```
After training, XGBoost assigns an importance score to every feature.
Importance = total gain (reduction in loss) from all splits using that feature.

Sort all ~210 features by importance, take the top 20 (cfg.top_k_genes=20).

Example result:
    {"EGFR": 0.0823, "BRAF": 0.0641, "ALK": 0.0512, "PIK3CA": 0.0489, ...}
    
These are the top 20 gene targets. Only real gene columns appear here 
(the _row_mean etc. columns are filtered out by name).
```

**After This Step**

We have:
- `top_genes`: an ordered dict of 20 gene names → importance scores
- `cv_metrics`: baseline cross-validation metrics (6 metrics × mean/std)
- A trained XGBoost model (used only for importance extraction — it is NOT used again for prediction)

---

## DNABERT-2: The Frozen Feature Extractor

DNABERT-2 sits BETWEEN Model 1 and Model 2. It is never trained by us. We only use it to convert DNA sequences into numbers.

### Where Does DNABERT-2 Come From?

We download it. We don't create it. Here's what happened before our project:

```
Pre-Training (done by others — we never run this):

Step 1 (Zhihan Zhou et al., 2023):
    Trained DNABERT-2 on multi-species genomes using:
    
    BPE Tokenization: Unlike DNABERT-1 which used fixed k-mers (e.g., 6-mers),
    DNABERT-2 uses Byte Pair Encoding — a data-driven tokenization that learns
    the most common DNA subword patterns. This handles variable-length motifs.
    
    Masked Language Modeling (MLM): Randomly mask 15% of tokens, predict them
    using context from both directions (bidirectional).
    
    "ACGT[MASK]GCTA" → model predicts the masked token using left AND right context
    
    This is the SAME pre-training task as BERT for English text, but applied to
    DNA characters instead of words.

Result: zhihan1996/DNABERT-2-117M — 117 million parameters that understand
        DNA sequence patterns, motifs, and structural features.
```

### What We Download vs. What We Use

```
DOWNLOADED (pre-trained):           WHAT WE DO WITH IT:
├── 12 transformer layers           ├── Feed it DNA sequences
├── 768 hidden dimensions           ├── Extract the embeddings
├── ~117M parameters                └── Use embeddings as features for XGBoost
└── Understands DNA patterns            (that's it — no fine-tuning, no training)
    (via MLM pre-training)
```

We NEVER run MLM. We NEVER update DNABERT-2's weights. We use it in `eval()` mode as a frozen feature extractor.

### What Actually Happens (Step by Step)

**Step 2a: Fetch DNA sequences from NCBI**

For each of the 20 top genes from Model 1:

```
For gene "EGFR":
    1. Query NCBI Entrez: "EGFR[Gene Name] AND Homo sapiens[Organism] AND REFSEQ"
    2. Get back an NCBI ID (e.g., "NM_005228.5")
    3. Fetch the FASTA sequence for that ID
    4. Parse out the nucleotide string: "ATGCGACCCTCCGGGACGGCCG..."
    5. Strip any non-ACGT characters
    
    If lookup fails (network error, gene not found):
        Generate a deterministic synthetic sequence:
        Use the gene name as a seed → numpy RNG → 1000 random A/C/G/T characters
        Different genes get different synthetic sequences (seeded by gene name hash)
    
    If sequence is too short (< 20 bp):
        Pad it with synthetic bases to reach ≥ 200 bp
```

After this step we have:

```
sequences = {
    "EGFR":   "ATGCGACCCTCCGGGACGGCCG...",    # ~5,000+ bp from NCBI
    "BRAF":   "ATGCGGCGCTGAGCGGTGG...",       # ~2,800+ bp from NCBI
    "ALK":    "ATGGAAGTTAAGGATGAAATG...",     # ~6,000+ bp from NCBI
    ...20 genes total...
}
```

**Step 2b: Compute DNABERT-2 embeddings**

For each gene, feed its DNA sequence through DNABERT-2:

```
For gene "EGFR" with sequence "ATGCGACCCTCCGGG...":

    1. TRUNCATE to 512 tokens (max_seq_len)
       The raw sequence is often thousands of base pairs, but DNABERT-2
       can only process 512 tokens at a time. Excess is dropped.
    
    2. TOKENIZE using BPE (Byte Pair Encoding)
       "ATGCGACCCTCCGGG..." → [101, 4523, 8712, 3341, ..., 102]
       
       101 = [CLS] token (start marker)
       102 = [SEP] token (end marker)
       Rest = BPE subword tokens representing DNA subsequences
       
       Pad with 0s to exactly 512 positions.
       Create an attention mask: 1 for real tokens, 0 for padding.
       
    3. MOVE tensors to device (CUDA → MPS → CPU, whichever is available)
    
    4. FORWARD PASS through 12 transformer layers (no gradients computed)
       
       512 token IDs → DNABERT-2 (12 layers of self-attention) → 512 vectors of size 768
       
       Each of the 512 token positions now has a 768-dimensional vector
       representing that position in context of all other positions.
    
    5. MEAN POOL (not [CLS] — this is different from typical BERT usage!)
       
       Take all 512 position vectors (512 × 768 matrix)
       Multiply by the attention mask (zero out padding positions)
       Sum along the sequence dimension → 1 × 768 vector
       Divide by the count of non-padding tokens
       
       Result: one 768-dimensional vector summarizing the entire EGFR gene
       
       Example: [0.0234, -0.1456, 0.0891, ..., 0.0412]   (768 numbers)
```

### 768 Is NOT the Sequence Length

A common confusion:

```
512 = sequence length — how many tokens each DNA sequence is padded/truncated to
768 = hidden dimension — the size of DNABERT-2's internal vector for EACH token
```

Each token becomes a 768-dim vector. We average ALL token vectors (not just [CLS]) to get the gene embedding.

After this step we have:

```
gene_embeddings = {
    "EGFR":   np.array of shape (768,)    # e.g. [0.0234, -0.1456, ...]
    "BRAF":   np.array of shape (768,)    # e.g. [0.0189, -0.0923, ...]
    "ALK":    np.array of shape (768,)    # e.g. [0.0312, -0.1167, ...]
    ...20 genes total...
}
```

---

## Building the Fusion Feature Matrix

This is the most important step. This is where the two modalities meet.

### The Intuition

We want to create a per-cell-line "genomic context" vector. Cell lines that are sensitive to similar genes should have similar context vectors. We do this by weighting each gene's DNABERT-2 embedding by how sensitive that cell line is to that gene's drug.

### Step by Step

**Step 3a: Build the embedding matrix**

Stack the 20 gene embeddings into a matrix:

```
emb_matrix = shape (20, 768)

         dim0    dim1    dim2   ...   dim767
EGFR   [ 0.023, -0.145,  0.089, ...,  0.041]
BRAF   [ 0.019, -0.092,  0.067, ...,  0.033]
ALK    [ 0.031, -0.117,  0.103, ...,  0.058]
 ...
(20 rows × 768 columns)
```

**Step 3b: Extract sensitivity values**

For each cell line, pull its LN_IC50 values for just these 20 genes:

```
sensitivity = shape (968, 20)

              EGFR    BRAF    ALK    ...   (20 genes)
Cell 906826 [ 2.14,  -0.53,   0.00, ...]
Cell 687983 [ 0.00,   0.00,   1.87, ...]
 ...
(968 rows × 20 columns)
```

If a cell line was never tested against a gene's drug, the value is 0.0.

**Step 3c: Matrix multiplication — the fusion point**

```
weighted_embeddings = sensitivity @ emb_matrix

    (968 × 20) @ (20 × 768) = (968 × 768)
```

For a single cell line, this computes:

```
Cell 906826's context vector =
    2.14 × EGFR_embedding     (+)     ← heavily weighted (very sensitive to EGFR drugs)
   -0.53 × BRAF_embedding     (+)     ← negative weight (resistant to BRAF drugs)
    0.00 × ALK_embedding      (+)     ← zero weight (no data for ALK drugs)
    ...
    = one 768-dim vector that summarizes this cell line's drug-sensitivity profile
      through the lens of DNA sequence knowledge
```

**This is the fusion.** Each cell line's drug-sensitivity numbers weight the DNA-level representations. A cell line that is highly sensitive to EGFR drugs gets a context vector dominated by EGFR's DNA embedding. A cell line resistant to BRAF drugs gets a context vector that subtracts BRAF's DNA embedding.

**Step 3d: L2-normalize each row**

```
For each cell line's 768-dim vector:
    norm = sqrt(sum of squares of all 768 values)
    divide every value by norm
    
Result: every row is a unit vector (length = 1.0)

This prevents cell lines with large absolute sensitivity values from 
dominating. All cell lines now live on the same scale.
```

**Step 3e: PCA compression (768 → 64 dimensions)**

```
768 dimensions is too many for XGBoost to handle efficiently.
Many of those dimensions are noise.

PCA (Principal Component Analysis):
    1. Center the data (subtract column means)
    2. Compute the covariance matrix (768 × 768)
    3. Find the top 64 eigenvectors (directions of maximum variance)
    4. Project each 768-dim row onto these 64 directions
    
Result: (968 × 64) matrix

The cfg.bert_pca_dims = 64 controls this. We keep 64 principal components.
Typically retains 80-95% of the total variance.
```

New columns are named `BERT_pca_0`, `BERT_pca_1`, ..., `BERT_pca_63`.

**Step 3f: Concatenate with original features**

```
X_fused = [original drug-sensitivity features | PCA-reduced DNABERT-2 embeddings]

         EGFR   BRAF   ALK  ...  _row_mean  _row_std  ...  BERT_pca_0  BERT_pca_1  ...  BERT_pca_63
Cell 1 [ 2.14, -0.53, 0.00, ...,     0.27,     0.94, ...,     0.312,    -0.089,  ...,     0.045 ]
Cell 2 [ 0.00,  0.00, 1.87, ...,     0.31,     0.82, ...,     0.198,     0.023,  ...,    -0.112 ]
...

Shape: (968, ~210 + 64) = (968, ~274 columns)
```

Now each cell line's feature vector contains:
1. **~200 gene columns:** direct drug-sensitivity values (how sensitive to each gene's drug)
2. **10 statistical columns:** row-level distribution summary
3. **64 BERT columns:** DNA-sequence-informed genomic context (sensitivity-weighted embeddings)

---

## MODEL 2: Fusion XGBoost

### What It Sees

The fused feature matrix: ~968 rows × ~274 columns. This includes everything from Model 1 PLUS the 64 DNABERT-2 embedding columns.

Same labels as Model 1: cancer type integers.

### What It Does

Exactly the same training procedure as Model 1:

```
Step 4a: Optuna HPO (50 trials)
    Same search space as Model 1
    Same 5-fold stratified CV inner loop
    Same F1 maximisation objective
    But now operating on ~274 features instead of ~210
    Finds DIFFERENT optimal hyperparameters (the extra BERT features change the landscape)

Step 4b: Train final fusion XGBoost
    Same class-balanced sample weights
    Same multi:softprob objective
    But trained on X_fused (~274 columns) instead of X_augmented (~210 columns)
    
Step 4c: Cross-validate the fusion model
    Same RepeatedStratifiedKFold (5 × 3 = 15 folds)
    Same 6 metrics (accuracy, precision, recall, F1, F2, ROC AUC)
    Results stored separately as fusion_cv_metrics
```

The key comparison is:

```
Baseline CV metrics:  XGBoost trained on  ~210 features (sensitivity + stats only)
Fusion CV metrics:    XGBoost trained on  ~274 features (sensitivity + stats + DNABERT-2)
Fusion lift:          fusion_F1 - baseline_F1  (positive = DNABERT-2 helped)
```

If fusion F1 > baseline F1, the DNABERT-2 embeddings contain information about cancer-type classification that the raw sensitivity numbers alone don't capture.

### What Gets Updated When We Train Model 2

```
Component         Trained?    Parameters
Fusion XGBoost    YES         ~1000 trees × internal nodes
DNABERT-2         NO          117M params — completely frozen, never touched
PCA projection    NO          Fitted once on weighted embeddings, then frozen
Baseline XGBoost  NO          Already done, not used again
```

---

## Instability Scoring (Embedding Drift)

After Model 2 trains, we score each top gene's "mutational instability." This measures how fragile a gene's DNA structure is — how much its embedding changes when we introduce small mutations.

```
For each of the 20 top genes:

    1. Get the reference embedding:
       ref_emb = DNABERT-2.embed(original_sequence)          # 768-dim vector
    
    2. Repeat 30 times (cfg.fuzz_iterations=30):
       a. Introduce random point mutations:
          n_mutations = max(3, len(sequence) // 100)         # ~1% mutation rate
          
          For n_mutations random positions:
              Replace the base with a DIFFERENT random base
              e.g., position 847: A → C
              e.g., position 2103: G → T
              
       b. Embed the mutant:
          mut_emb = DNABERT-2.embed(mutated_sequence)        # 768-dim vector
          
       c. Compute cosine distance:
          similarity = cosine_similarity(ref_emb, mut_emb)   # value in [-1, 1]
          drift = 1.0 - similarity                           # value in [0, 2]
          
          drift = 0.0  means the embedding didn't change at all
          drift = 1.0  means the vectors are orthogonal (very different)
          drift = 2.0  means the vectors point in opposite directions
    
    3. instability_score = mean of all 30 drift values

Example:
    EGFR:  30 drifts = [0.0023, 0.0031, 0.0019, ...]  → mean = 0.0025 (stable gene)
    TP53:  30 drifts = [0.0089, 0.0102, 0.0076, ...]  → mean = 0.0091 (less stable)
```

**Why this matters:** A gene whose embedding shifts a lot from small mutations is structurally "fragile" — its DNA landscape is sensitive to perturbation. These genes may be better drug targets because small molecular disruptions (what a drug does) could have outsized effects.

---

## Fusion Index: The Final Ranking Score

```
For each gene:
    Fusion_Index = XGB_Importance × Instability × 1000

Example:
    EGFR:    0.0823 × 0.0025 × 1000 = 0.2058
    BRAF:    0.0641 × 0.0091 × 1000 = 0.5833    ← higher despite lower importance
    PIK3CA:  0.0489 × 0.0067 × 1000 = 0.3276
```

The `× 1000` is just a scaling factor so the numbers aren't tiny decimals.

**This is NOT a learned score.** It's a hand-designed heuristic that multiplies two independently computed quantities:
- **XGB_Importance:** How much the gene matters for classifying cancer types by drug sensitivity
- **Instability:** How structurally fragile the gene's DNA is to point mutations

Genes rank high when they are BOTH important for drug sensitivity AND structurally fragile.

---

## Downstream Annotations (After Scoring)

After the Fusion Index ranks all 20 genes, six more annotation steps run. None of these train any model — they just look up data.

### Step 5: Pathway Enrichment

```
For each gene:
    Look up which KEGG/Reactome pathways it belongs to.
    Uses a hardcoded pathway database.
    
Result: a "Pathways" column like "PI3K-Akt signaling; MAPK signaling"
```

### Step 6: Drug-Target Mapping

```
For each gene:
    Look up in a curated database of ~55 gene targets × ~130 drugs.
    Each entry has: drug name, status (Approved/Phase I-III), indication.
    
Example:
    EGFR → ["Erlotinib (Approved, NSCLC)", "Osimertinib (Approved, NSCLC)"]
    BRAF → ["Dabrafenib (Approved, Melanoma)", "Vemurafenib (Approved, Melanoma)"]
    
Result: "Approved_Drugs" and "Druggable" columns
```

This database is a **static Python dictionary** — clinically accurate but not updated from live APIs.

### Step 7: Resistance Risk Scoring

```
For each gene:
    Look up known resistance mechanisms in a curated database of ~28 genes × ~55 mechanisms.
    
Example:
    EGFR → "T790M gatekeeper mutation (~60%), MET amplification (~10-25%)"
    BRAF → "MAPK pathway reactivation (~50%)"
    
Risk score = min(1.0, number_of_mechanisms × 0.2)
    1  mechanism  → 0.2
    3  mechanisms → 0.6
    5+ mechanisms → 1.0
    
Result: "Resistance_Risk" column
```

### Step 8: Synthetic Lethality Screening

```
For each gene:
    Look up known synthetic lethality partners.
    e.g., BRCA1 ↔ PARP (if BRCA1 is mutated, PARP inhibitors are lethal)
    
Result: "SL_Partners" column
```

### Step 9: Network Pharmacology Score

```
For each gene:
    Compute a "strategic score" based on the gene's position in 
    protein-protein interaction networks.
    Genes that are hubs (many connections) score higher.
    
Result: "Strategic_Score" column
```

---

## Clinical Evidence (Separate from the Pipeline)

After the pipeline runs, the notebook queries three live APIs for the top 5 genes:

```
For each of the top 5 genes:

    1. OpenTargets (GraphQL API):
       Query: gene → all disease associations
       Returns: overall association score (0.0 to 1.0)
       e.g., EGFR → 0.89 (strong associations with many cancers)
    
    2. CIViC (GraphQL API):
       Query: gene → all variant-level evidence items
       Returns: number of evidence items with type/level/direction/significance
       e.g., EGFR → 47 evidence items
    
    3. ClinicalTrials.gov (REST API):
       Query: gene name as search term
       Returns: number of matching clinical trials
       e.g., EGFR → 823 trials

    Composite score = 0.4 × OpenTargets + 0.3 × CIViC_normalised + 0.3 × Trials_normalised
    
    CIViC normalised    = min(1.0, n_evidence_items / 10)
    Trials normalised   = min(1.0, n_trials / 10)
```

These are **real HTTP requests** to public APIs, not cached or mocked. Each has a 15-second timeout with graceful fallback to 0 on failure.

---

## Digital Twin Tumour Simulation

A separate analysis that simulates how a tumour would respond to treatment over 180 days.

### The ODE System

Three cell populations tracked simultaneously:

```
dS/dt = r·S·ln(K/S)  −  drug_kill·S  −  immune_kill·S·I  −  conversion·S
        ↑ Gompertz      ↑ drugs kill    ↑ immune cells     ↑ some sensitive
          growth          sensitive       kill sensitive     cells become
                          cells           cells              resistant

dR/dt = r·R·ln(K/R)  −  0.3·immune_kill·R·I  +  conversion·S
         ↑ Gompertz      ↑ immune is less         ↑ gain from
           growth          effective on             sensitive
                           resistant cells          conversion

dI/dt = recruitment·(S+R)  −  exhaustion·I
         ↑ immune cells       ↑ immune cells
           recruited by         die over time
           tumour presence

Where:
    S = sensitive cell count
    R = resistant cell count
    I = immune cell count
    r = tumour growth rate (0.02 per day)
    K = carrying capacity (10^12 cells)
    drug_kill = sum of all active drug efficacies at that time point
    immune_kill = 10^-9 (very small — immune effect is modest)
    conversion = resistant_fraction × drug_effect (drugs select for resistance)
    recruitment = 10^-7 × total tumour size
    exhaustion = 0.03 per day
```

### Integration Method

```
Forward Euler with dt = 0.5 days:
    For each half-day step:
        Compute dS/dt, dR/dt, dI/dt from current state
        S_new = S + dS/dt × 0.5
        R_new = R + dR/dt × 0.5
        I_new = I + dI/dt × 0.5
        Clamp all values to ≥ 0 (can't have negative cells)
```

**Not RK4.** Euler is simpler but can accumulate numerical error. Adequate for qualitative screening but not for precision clinical predictions.

### RECIST Classification

```
At each time point, compute:
    response = (current_total - initial_total) / initial_total × 100

Final classification based on best response:
    CR (Complete Response):   best ≤ -100%   (tumour gone)
    PR (Partial Response):    best ≤ -30%    (significant shrinkage)  
    SD (Stable Disease):      best > -30% AND best < +20%
    PD (Progressive Disease): best ≥ +20%    (tumour growing)
```

---

## Companion Diagnostics

A rule-based system that takes a patient mutation profile and generates treatment recommendations.

### Input: Patient Profile

```
PatientProfile(
    patient_id = "GDSC-DEMO-001",
    cancer_type = "BRCA",
    mutations = [
        {"gene": "EGFR", "variant": "L858R",  "vaf": 0.42},
        {"gene": "BRAF", "variant": "V600E",  "vaf": 0.35},
        ...12 mutations total...
    ],
    cna = {"MYC": 2.1, "CDKN2A": -1.5, "ERBB2": 1.8}
)

TMB (Tumour Mutational Burden) = number of mutations = 12
```

**This patient is synthetic** — manually constructed for demonstration, not from real clinical data.

### Tiering Logic (AMP/ASCO/CAP-inspired)

```
For each mutation:
    Look up that gene in the drug-target database
    
    Tier I:    Gene has an FDA-approved drug           (e.g., EGFR → Erlotinib)
    Tier II:   Gene has a clinical-trial drug          (e.g., Phase II/III)
    Tier III:  Gene has any drug entry (preclinical)
    Tier IV:   No drug matches (variant of unknown significance, VUS)
```

This is driven by the drug-target DB — NOT hardcoded per-gene. If you add a new gene to the drug DB with status "Approved," it automatically becomes Tier I.

### Treatment Recommendations

```
For each actionable mutation:
    If drug is approved:           confidence = 0.90
    If synthetic lethality match:  confidence = 0.65
    If drug is investigational:    confidence = 0.50
    If drug has known resistance:  confidence = 0.27 (de-prioritised)

If TMB ≥ 10:
    Recommend pembrolizumab (anti-PD-1 immunotherapy)
    This matches the FDA tissue-agnostic TMB-high approval.

Sort all recommendations by confidence score descending.
```

---

## End-to-End Summary

```
WHAT OTHERS DID (before our project):
──────────────────────────────────────
  Zhihan Zhou et al. (2023): Trained DNABERT-2 on multi-species genomes (MLM)
  Result: zhihan1996/DNABERT-2-117M (we download this, never retrain it)


WHAT WE DO (in this notebook):
──────────────────────────────────────────

   1. LOAD DATA: Read GDSC CSVs, explode multi-target entries, pivot to matrix
      Result: X (968 × ~200 genes), y (968 cancer-type labels)

   2. FEATURE ENGINEERING: Add 10 row-level statistics to X
      Result: X_augmented (968 × ~210)

   3. TRAIN MODEL 1: Baseline XGBoost
      Input:   X_augmented (~210 features)
      Steps:   Optuna HPO (50 trials) → train → 15-fold CV → extract top 20 genes
      Output:  top_genes dict, baseline CV metrics
      Status:  DONE. Model used only for gene ranking, then FROZEN.

   4. DNABERT-2 FEATURE EXTRACTION (no training)
      Input:   20 gene names
      Steps:   Fetch DNA from NCBI → tokenize → forward pass → mean pool
      Output:  20 embeddings of shape (768,)
      Status:  DNABERT-2 weights NEVER updated. Frozen feature extractor.

   5. BUILD FUSION FEATURES
      Steps:
        a. Stack embeddings into (20 × 768) matrix
        b. Extract sensitivity values for 20 genes → (968 × 20)
        c. Matrix multiply: (968 × 20) @ (20 × 768) = (968 × 768) weighted embeddings
        d. L2-normalise each row
        e. PCA compress: (968 × 768) → (968 × 64)
        f. Concatenate with original features: (968 × ~210) + (968 × 64) = (968 × ~274)
      Output:  X_fused (968 × ~274)

   6. TRAIN MODEL 2: Fusion XGBoost
      Input:   X_fused (~274 features)
      Steps:   Optuna HPO (50 trials) → train → 15-fold CV
      Output:  fusion CV metrics
      Compare: fusion_F1 vs baseline_F1 = the "fusion lift"

   7. SCORE INSTABILITY: For each top gene, mutate DNA 30 times, measure embedding drift
      Output:  instability scores (one per gene)

   8. COMPUTE FUSION INDEX: importance × instability × 1000
      Output:  ranked list of 20 genes with Fusion Index scores

   9. ANNOTATE: Pathways, drugs, resistance, synthetic lethality, network pharmacology
      Output:  enriched results DataFrame

  10. CLINICAL EVIDENCE: Query OpenTargets, CIViC, ClinicalTrials.gov APIs
      Output:  evidence scores for top 5 genes

  11. DIGITAL TWIN: Simulate tumour response with Gompertzian ODE (180 days)
      Output:  trajectory, RECIST classification, best response %

  12. COMPANION DX: Tier mutations, recommend treatments based on drug DB
      Output:  tiered mutations, ranked treatment plan, resistance alerts
```

---

## Key Differences from a Late Fusion Neural Network

```
                           NEISS Late Fusion           Fusion Oncology
                           ──────────────────          ─────────────────
Text/Sequence model        ClinicalBERT (fine-tuned)   DNABERT-2 (FROZEN)
Tabular model              XGBoost                     XGBoost (×2)
Fusion method              Concatenate at the end      Matrix multiply + PCA + concatenate
What gets trained          BERT + Linear layer         XGBoost only (×2 separate trainings)
How BERT is used           Fine-tuned end-to-end       Frozen feature extractor
Fusion point               CLS vector + XGB score      Sensitivity-weighted embeddings
Neural network involved?   YES (PyTorch)               NO (all sklearn/xgboost/numpy)
Backpropagation through    BERT + Linear               N/A — no neural network training
BERT?
Prediction task            Binary (Severe/Not)         Multi-class (cancer type)
Final score                Sigmoid probability         Fusion Index (heuristic product)
```

The biggest difference: **DNABERT-2 is never fine-tuned.** In the NEISS model, ClinicalBERT's 110M parameters are all updated via backpropagation. Here, DNABERT-2's 117M parameters are completely frozen — we only extract embeddings from it. The actual "learning" happens entirely inside the two XGBoost classifiers.

---

## Disclaimer

This is a research tool for hypothesis generation. It is not a clinical diagnostic device. Not FDA approved. The Fusion Index is a ranking heuristic, not a validated biomarker. The digital twin uses simplified ODE dynamics. The companion diagnostic uses rules-based tiering, not full AMP/ASCO/CAP variant-level interpretation.
