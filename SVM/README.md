# IEC 104 SCADA Intrusion Detection System — SVM

Detects 6 types of cyber attacks (+ normal traffic) on IEC 104 SCADA networks using Support Vector Machines. The model goes through a **3-phase improvement process** — baseline → improved → optimised — to demonstrate how preprocessing and tuning boost performance.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the 8 CSV data files in a folder called data/
mkdir data
cp /path/to/capture104*.csv data/

# 3. Run the full pipeline (produces 10 figures + console metrics)
python svm_ids.py

---

## Files

| File | Purpose |
|------|---------|
| `svm_ids.py` | Main script — trains 3 SVM phases, prints metrics, saves 10 figures to `output/` |
| `requirements.txt` | Python packages needed |
| `data/` | Put the 7 CSV files here |
| `output/` | Figures are saved here automatically |

---

## Dataset

8 CSV files from CICFlowMeter (319,971 total flows, 84 columns each):

| Class | Samples |
|-------|--------:|
| dosattack | 304,627 |
| portscanattack | 9,710 |
| ntpddosattack | 2,278 |
| iec104starvationattack | 2,028 |
| fuzzyattack | 939 |
| attackfree (normal) | 255 |
| floodattack | 108 |
| mitmattack | 26 |

---

## How the 3 Phases Work

**Phase 1 — Baseline:** Linear SVM on raw features. No scaling, no tuning. Performs poorly (~30% accuracy) but sets the floor.

**Phase 2 — Improved:** Adds StandardScaler, ANOVA feature selection (top 30), RBF kernel, and class balancing. Jumps to ~99% accuracy.

**Phase 3 — Optimised:** GridSearchCV tunes C, gamma, and kernel with 5-fold CV. Reaches ~99.5% accuracy.

---

## Results

| Metric | Baseline | Improved | Optimised |
|--------|----------|----------|-----------|
| Accuracy | 29.5% | 99.1% | **99.5%** |
| F1 (macro) | 40.3% | 90.6% | **91.8%** |
| F1 (weighted) | 20.5% | 99.2% | **99.5%** |

Best model: `poly` kernel, `C=100`, `gamma=0.1`, `class_weight=balanced`, 30 ANOVA-selected features.

---

## Output Figures

The script generates 10 figures at 300 dpi in `output/`. Each has a `REPORT USAGE` comment in the code explaining where to use it:

1. Class distribution bar chart
2. Feature correlation heatmap
3. Feature boxplots by attack class
4. ANOVA feature importance ranking
5. Performance comparison (bars + improvement trajectory)
6. Confusion matrices (all 3 phases)
7. Per-class F1 scores
8. PCA scatter plots
9. GridSearchCV heatmap
10. Learning curve

---

## Requirements

- Python 3.9+
- scikit-learn, pandas, numpy, matplotlib, seaborn

Install everything with: `pip install -r requirements.txt`