# ML-Based Intrusion Detection for IEC 104 SCADA Networks

**Course:** ECE1513 — Introduction to Machine Learning, University of Toronto  
**Authors:** Hengyu Liu, Jake Shi, Jiakai Tang

---

## Project Overview

This project develops and evaluates machine learning classifiers for detecting network intrusions in IEC 104 SCADA power grid traffic. We train and compare two models — a Support Vector Machine (SVM) and a Feedforward Neural Network (FNN) — on a labeled dataset of 319,971 network flows captured from a substation testbed.

The dataset contains 8 classes: 7 attack types (DoS, flood, fuzzy, IEC 104 starvation, MITM, NTP DDoS, port scan) and one attack-free (normal) class. A key challenge is severe class imbalance — the DoS class alone accounts for 95.2% of samples. We address this with a per-class resampling strategy combining SMOTE, ADASYN, random over/undersampling, and ClusterCentroids.

**Best results:**
- SVM (Phase 3, optimized): 100% accuracy, 93.3% macro-F1
- FNN: 99.6% validation accuracy

See [`Project report/report.pdf`](Project%20report/report.pdf) for the full write-up.

---

## Repository Structure

```
ECE1513/
├── Data Treatment/
│   └── data-treatment.ipynb      # Data loading, cleaning, encoding, resampling
├── FNN/
│   ├── Fnn.ipynb                 # FNN model training (Jupyter notebook)
│   └── fnn.py                    # FNN model training (Python script)
├── SVM/
│   ├── svm_ids.py                # SVM 3-phase pipeline (baseline → improved → optimized)
│   ├── requirements.txt          # SVM-specific pip requirements
│   ├── README.md                 # SVM module documentation
│   └── output/                   # Generated figures (10 PNG, 300 dpi)
├── iec104_pre-processed/         # Raw dataset (download separately — see below)
│   ├── attack-data/              # 7 CSV files, one per attack type
│   └── attack-free-data/         # 1 CSV file (normal traffic)
├── post-treatment/
│   └── iec104_treated_balanced.csv  # Balanced dataset produced by data-treatment.ipynb
├── Project report/
│   ├── report.pdf                # Full project report
│   └── report.tex                # LaTeX source
├── environment.yaml              # Conda environment (all dependencies)
└── README.md                     # This file
```

---

## Dataset

**Source:** [IEC 104 SCADA Network Intrusion Detection Dataset](https://zenodo.org/records/15487636) (Zenodo)

| Property | Value |
|----------|-------|
| Total samples | 319,971 network flows |
| Features | 84 columns (80 numeric + 4 metadata) |
| Classes | 8 (7 attack types + 1 normal) |
| Feature extractor | CICFlowMeter |

**Class distribution (raw):**

| Class | Samples | Share |
|-------|---------|-------|
| dosattack | 304,627 | 95.2% |
| portscanattack | 9,710 | 3.0% |
| ntpddosattack | 2,278 | 0.7% |
| iec104starvationattack | 2,028 | 0.6% |
| fuzzyattack | 939 | 0.3% |
| attackfree | 255 | 0.1% |
| floodattack | 108 | 0.0% |
| mitmattack | 26 | 0.0% |

**Setup:** Download and extract the dataset from Zenodo. Place the contents so that the directory structure matches:

```
ECE1513/
└── iec104_pre-processed/
    ├── attack-data/
    │   ├── capture104-dosattack.csv
    │   ├── capture104-floodattack.csv
    │   ├── capture104-fuzzyattack.csv
    │   ├── capture104-iec104starvationattack.csv
    │   ├── capture104-mitmattack.csv
    │   ├── capture104-ntpddosattack.csv
    │   └── capture104-portscanattack.csv
    └── attack-free-data/
        └── capture104-attackfree.csv
```

---

## Environment Setup

Requires [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yaml
conda activate ece1513-ids
```

To verify the environment:

```bash
python -c "import pandas, numpy, sklearn, imblearn, torch, matplotlib, seaborn; print('OK')"
```

---

## Workflow

Run the three steps in order. Step 1 produces the balanced dataset that Steps 2 and 3 consume.

### Step 1 — Data Preprocessing

**Notebook:** [`Data Treatment/data-treatment.ipynb`](Data%20Treatment/data-treatment.ipynb)

```bash
jupyter notebook "Data Treatment/data-treatment.ipynb"
```

This notebook:
1. Loads all 8 CSV files from `iec104_pre-processed/`
2. Drops NaN rows and encodes categorical features (e.g., IP addresses, Protocol)
3. Applies a per-class resampling strategy to address severe imbalance:
   - DoS: Random undersampling (304,627 → 10,000)
   - Port scan: ClusterCentroids (9,710 → 5,000)
   - NTP DDoS, IEC 104 starvation, attack-free: SMOTE (→ 3,000 each)
   - Fuzzy: ADASYN (939 → 2,000)
   - Flood: SMOTE (108 → 1,000)
   - MITM: Random oversampling (26 → 300)
4. Saves the balanced dataset to `post-treatment/iec104_treated_balanced.csv`

> **Note:** The notebook was originally written for Google Colab. If running locally, update the file paths at the top of the first cell (remove the `drive.mount` call and set `attack_dir`, `attack_free_dir`, and `dest_folder` to local paths).

### Step 2 — SVM Training

```bash
python SVM/svm_ids.py
```

This script runs a 3-phase improvement pipeline evaluated across 5 random seeds:

| Phase | Description | Macro-F1 |
|-------|-------------|----------|
| Phase 1 — Baseline | Linear SVM, raw features, no scaling | 44.9% |
| Phase 2 — Improved | RBF kernel, StandardScaler, ANOVA feature selection (top 30), balanced class weights | 89.4% |
| Phase 3 — Optimized | GridSearchCV over C, gamma, kernel (poly wins: C=100, gamma=0.1) | 93.3% |

Outputs 10 publication-quality figures to `SVM/output/`. See [`SVM/README.md`](SVM/README.md) for details.

### Step 3 — FNN Training

**Notebook (recommended):**
```bash
jupyter notebook FNN/Fnn.ipynb
```

**Or as a script:**
```bash
python FNN/fnn.py
```

The FNN architecture:
```
Input (80 features)
→ Dense(256) + ReLU + BatchNorm + Dropout(0.3)
→ Dense(128) + ReLU + BatchNorm + Dropout(0.3)
→ Dense(64)  + ReLU
→ Dense(8)   + Softmax
```

Training uses Adam (lr=1e-3), cross-entropy loss with balanced class weights, batch size 128, 20 epochs. The script loads `post-treatment/iec104_treated_balanced.csv` produced in Step 1.

---

## Results Summary

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| SVM Phase 1 (baseline) | 96.4% | 44.9% | 96.4% |
| SVM Phase 2 (improved) | 99.9% | 89.4% | 99.9% |
| SVM Phase 3 (optimized) | 100.0% | 93.3% | 100.0% |
| FNN (balanced data) | 99.6% | — | — |

SVM metrics are mean ± std over 5 random seeds. The weakest per-class result is MITM (F1 ≈ 0.57–0.67) due to only 26 original training samples.

---

## Project Report

The full report is at [`Project report/report.pdf`](Project%20report/report.pdf). It covers dataset analysis, preprocessing design, SVM and FNN methodology, numerical experiments, and discussion of results.

---

## AI Usage Disclosure

AI tools (Gemini) were used as a code explainer for researched code and to help structure and refine written portions of the report. All technical decisions — including model architecture, hyperparameter choices, resampling strategies, and experimental design — were made independently by the group members. AI served as a productivity tool rather than a decision-maker, and all outputs were critically reviewed and validated by the team.
