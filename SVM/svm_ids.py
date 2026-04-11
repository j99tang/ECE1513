"""
================================================================================
IEC 104 SCADA Network Intrusion Detection System — SVM Implementation
================================================================================

PURPOSE:
    Classify network traffic flows captured from an IEC 104 SCADA environment
    into 7 categories:
        - Normal traffic  (attack-free)
        - 6 attack types  (flood, fuzzy, MITM, IEC-104 starvation,
                           NTP DDoS, port scan)

METHODOLOGY — Three-Phase Improvement Pipeline:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Phase 1 ▸ Baseline SVM                                            │
    │    Linear kernel, raw features, default C=1.0, no scaling.         │
    │    Establishes a performance floor to beat.                        │
    │                                                                    │
    │  Phase 2 ▸ Improved SVM                                            │
    │    + StandardScaler feature normalization                          │
    │    + ANOVA F-test feature selection (top-30 features)              │
    │    + RBF kernel (captures non-linear decision boundaries)          │
    │    + class_weight='balanced' (handles class imbalance)             │
    │                                                                    │
    │  Phase 3 ▸ Optimized SVM                                           │
    │    + Exhaustive GridSearchCV over {C, gamma, kernel}               │
    │    + 5-fold stratified cross-validation                            │
    │    + Best estimator selected by weighted-F1                        │
    └──────────────────────────────────────────────────────────────────────┘

DATASET:
    7 CSV files produced by CICFlowMeter from IEC 104 packet captures.
    84 columns per file: 80 numeric flow-level features + 4 metadata cols.
    Total: 15,344 flow records across all classes.

OUTPUTS:
    • Full console log with metrics and classification reports per phase
    • 10 publication-quality PNG figures (300 dpi) for direct report use
    • Each figure is annotated in code with a "REPORT USAGE" comment
      explaining what it shows and how to discuss it in a write-up.

Author : Jake Shi
Date   : 2026
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, os

from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Multiple seeds for robust evaluation ─────────────────────────────────
# Instead of a single random_state, we run the full pipeline with multiple
# seeds and report the AVERAGE metrics across all runs.  This reduces the
# effect of a single lucky/unlucky train/test split and gives more reliable
# performance estimates.
SEEDS = [42, 123, 256, 789, 1024]
N_SEEDS = len(SEEDS)

np.random.seed(42)  # for reproducible EDA / figure generation

# ── Global plot style — clean, academic-friendly ─────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "sans-serif",
})

OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


###############################################################################
#                                                                             #
#   SECTION 1 — DATA LOADING                                                 #
#                                                                             #
#   Each CSV file represents one traffic class captured from the IEC 104      #
#   SCADA testbed.  We load them individually, print row counts, then         #
#   concatenate into a single DataFrame for unified processing.               #
#                                                                             #
###############################################################################

print("=" * 72)
print("  SECTION 1 — DATA LOADING")
print("=" * 72)

DATA_DIR = "./data/"

# Map short names to filenames for clarity and logging
file_map = {
    "attackfree":    "capture104-attackfree.csv",
    "flood":         "capture104-floodattack.csv",
    "fuzzy":         "capture104-fuzzyattack.csv",
    "starvation":    "capture104-iec104starvationattack.csv",
    "mitm":          "capture104-mitmattack.csv",
    "ntpddos":       "capture104-ntpddosattack.csv",
    "portscan":      "capture104-portscanattack.csv",
    "dos":           "capture104-dosattack.csv",
}

frames = []
for tag, fname in file_map.items():
    df = pd.read_csv(DATA_DIR + fname)
    frames.append(df)
    print(f"  Loaded {fname:<45s}  {len(df):>6,} rows   label='{df['Label'].iloc[0]}'")

# Concatenate all files into one dataset
data = pd.concat(frames, ignore_index=True)
print(f"\n  Combined dataset : {data.shape[0]:,} rows  x  {data.shape[1]} columns")


###############################################################################
#                                                                             #
#   SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)                              #
#                                                                             #
#   Before modelling we examine class distribution, feature correlations,     #
#   and the statistical spread of key features.  These EDA visualizations     #
#   are essential for the report's "Dataset" section.                         #
#                                                                             #
###############################################################################

print("\n" + "=" * 72)
print("  SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 72)

# ── 2a. Class distribution ───────────────────────────────────────────────
class_counts = data["Label"].value_counts()
print("\n  Class distribution:")
for cls, cnt in class_counts.items():
    pct = cnt / len(data) * 100
    print(f"    {cls:<26s} {cnt:>6,}  ({pct:5.1f}%)")

# REPORT USAGE — Figure 1: Class Distribution
# Include in the "Dataset" section of your report.  This chart highlights
# the severe class imbalance: portscan dominates at 63.3% while MITM has
# only 26 samples (0.2%).  This motivates two key design decisions:
#   1) Using class_weight='balanced' in the SVM to prevent the model from
#      simply predicting the majority class.
#   2) Reporting macro-F1 alongside accuracy, since accuracy alone would
#      be misleadingly high if the model ignored minority classes.
fig, ax = plt.subplots(figsize=(10, 5.5))
palette = sns.color_palette("Set2", n_colors=len(class_counts))
bars = ax.bar(class_counts.index, class_counts.values, color=palette,
              edgecolor="black", linewidth=0.6)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}", ha="center", fontweight="bold", fontsize=10)
ax.set_title("Figure 1 — Class Distribution in the IEC 104 Dataset")
ax.set_ylabel("Number of Flows")
ax.set_xlabel("Traffic Class")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig01_class_distribution.png", bbox_inches="tight")
plt.close()
print("\n  Saved fig01_class_distribution.png")

# ── 2b. Feature correlation heatmap (sampled for readability) ────────────
# Drop non-numeric columns for correlation analysis
drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]
numeric_df = data.drop(columns=drop_cols, errors="ignore")
numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

# REPORT USAGE — Figure 2: Feature Correlation Heatmap
# Include in the "Feature Analysis" subsection.  This shows how strongly
# correlated many CICFlowMeter features are (e.g., Fwd Packet Length Mean
# and Fwd Segment Size Avg are perfectly correlated).  Highly correlated
# features carry redundant information — this justifies feature selection
# in Phase 2 to remove redundancy and reduce dimensionality.
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, ax=ax,
            square=True, linewidths=0.1,
            cbar_kws={"shrink": 0.7, "label": "Pearson Correlation"},
            xticklabels=False, yticklabels=False)
ax.set_title("Figure 2 — Feature Correlation Heatmap (79 Features)\n"
             "Red = positive correlation, Blue = negative correlation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  Saved fig02_correlation_heatmap.png")

# ── 2c. Boxplots of key features by attack class ────────────────────────
# REPORT USAGE — Figure 3: Feature Distributions by Attack Class
# Include in the "Feature Analysis" subsection.  These boxplots show how
# specific features vary across attack types — for example, Flow Duration
# tends to be much shorter for port scan attacks (quick SYN probes) and
# longer for starvation attacks (persistent connections).  Protocol and
# RST Flag Count clearly separate certain classes, confirming they are
# strong discriminators (matching the ANOVA results in Phase 2).
key_features = ["Flow Duration", "Total Fwd Packet", "Protocol",
                "Flow Bytes/s", "Fwd Packet Length Mean", "RST Flag Count"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feat in zip(axes.ravel(), key_features):
    plot_data = data[[feat, "Label"]].copy()
    plot_data[feat] = pd.to_numeric(plot_data[feat], errors="coerce")
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna()
    sns.boxplot(data=plot_data, x="Label", y=feat, ax=ax,
                palette="Set2", fliersize=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("")

plt.suptitle("Figure 3 — Distribution of Key Features by Attack Class",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig03_feature_boxplots.png", bbox_inches="tight")
plt.close()
print("  Saved fig03_feature_boxplots.png")


###############################################################################
#                                                                             #
#   SECTION 3 — DATA PREPROCESSING                                           #
#                                                                             #
#   Steps: drop metadata → encode labels → handle inf/NaN → split data       #
#                                                                             #
###############################################################################

print("\n" + "=" * 72)
print("  SECTION 3 — DATA PREPROCESSING")
print("=" * 72)

# Step 3a: Drop non-numeric identifier columns.
# These columns identify specific hosts and sessions rather than traffic
# patterns.  Including them would cause the model to memorize IP addresses
# rather than learning generalizable attack signatures.
drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
data_clean = data.drop(columns=drop_cols, errors="ignore")
print(f"  Dropped metadata columns: {drop_cols}")

# Step 3b: Encode the target label as integers.
# scikit-learn requires numeric targets for classification.
le = LabelEncoder()
data_clean["Label_enc"] = le.fit_transform(data_clean["Label"])
print(f"  Label encoding:")
for i, cls in enumerate(le.classes_):
    print(f"    {i} -> {cls}")

# Step 3c: Separate features (X) and target (y).
X = data_clean.drop(columns=["Label", "Label_enc"])
y = data_clean["Label_enc"]

# Step 3d: Handle infinities and missing values.
# CICFlowMeter sometimes produces Inf for ratio features when the
# denominator is zero (e.g., Flow Bytes/s when flow duration = 0).
# Strategy: replace Inf → NaN → column median.  Median is more robust
# to outliers than mean, which is important since network traffic
# features often have heavy-tailed distributions.
X = X.replace([np.inf, -np.inf], np.nan)
nan_count = X.isna().sum().sum()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
print(f"  Replaced {nan_count:,} inf/NaN cells with column medians")
print(f"  Final feature matrix : {X.shape[0]:,} samples x {X.shape[1]} features")

# Step 3e: Stratified train/test split and training are performed inside
# the multi-seed loop below.  We run the full 3-phase pipeline with
# multiple different random seeds and average the results for robustness.


###############################################################################
#                                                                             #
#   HELPER FUNCTION — Evaluate and Return Metrics                            #
#                                                                             #
###############################################################################

def evaluate(model, Xtr, Xte, ytr, yte, name, seed_num=None):
    """
    Train a model, predict on the held-out test set, print all evaluation
    metrics and a full classification report, then return the metrics dict.

    Parameters
    ----------
    model : sklearn estimator  – the SVM model to train
    Xtr, Xte  : array-like    – training / test feature matrices
    ytr, yte   : array-like    – training / test targets
    name       : str           – display name for this phase
    seed_num   : int or None   – current seed index for display

    Returns
    -------
    metrics : dict  – accuracy, f1_weighted, f1_macro, precision, recall, time
    y_pred  : np.ndarray – predictions on the test set
    """
    t0 = time.time()
    model.fit(Xtr, ytr)
    elapsed = time.time() - t0

    y_pred = model.predict(Xte)

    # Compute aggregate metrics
    acc  = accuracy_score(yte, y_pred)
    f1w  = f1_score(yte, y_pred, average="weighted", zero_division=0)
    f1m  = f1_score(yte, y_pred, average="macro",    zero_division=0)
    prec = precision_score(yte, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(yte, y_pred, average="weighted", zero_division=0)

    # Pretty-print summary
    seed_label = f" [seed {seed_num}]" if seed_num is not None else ""
    print(f"\n  {'─' * 50}")
    print(f"  {name}{seed_label}")
    print(f"  {'─' * 50}")
    print(f"    Accuracy              {acc:.4f}")
    print(f"    F1 Score (weighted)   {f1w:.4f}")
    print(f"    F1 Score (macro)      {f1m:.4f}")
    print(f"    Precision (weighted)  {prec:.4f}")
    print(f"    Recall (weighted)     {rec:.4f}")
    print(f"    Training time         {elapsed:.2f}s")
    print(f"  {'─' * 50}")

    # Per-class classification report
    print("\n  Per-class Classification Report:")
    print(classification_report(
        yte, y_pred, target_names=le.classes_, zero_division=0
    ))

    metrics = dict(
        accuracy=acc, f1_weighted=f1w, f1_macro=f1m,
        precision=prec, recall=rec, time=elapsed,
    )
    return metrics, y_pred


###############################################################################
#                                                                             #
#   SECTIONS 4–6 — MULTI-SEED TRAINING LOOP                                  #
#                                                                             #
#   We run the full 3-phase pipeline (Baseline → Improved → Optimized)       #
#   with each seed in SEEDS.  Each seed produces a different stratified       #
#   train/test split, so the final reported metrics are the AVERAGE across   #
#   all runs — giving a more robust and honest performance estimate than     #
#   a single lucky split.                                                    #
#                                                                             #
#   The predictions and objects from the LAST seed are kept for generating   #
#   the report figures (confusion matrices, PCA scatter, etc.).              #
#                                                                             #
###############################################################################

# Collect per-seed metrics for each phase
all_run_metrics = {
    "Phase 1: Baseline":  [],
    "Phase 2: Improved":  [],
    "Phase 3: Optimized": [],
}

# These will hold the last run's objects for figure generation
pred_p1 = pred_p2 = pred_p3 = None
y_test = None
X_test_sel = None
best_model = None
grid = None
selected_names = None
selected_scores = None
selected_mask = None

for run_idx, seed in enumerate(SEEDS):

    print("\n" + "#" * 72)
    print(f"  RUN {run_idx + 1}/{N_SEEDS}  —  seed = {seed}")
    print("#" * 72)

    # ── Train/test split with this seed ──────────────────────────────────
    X_train, X_test_run, y_train, y_test_run = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    print(f"  Training set : {X_train.shape[0]:,} samples  (75%)")
    print(f"  Test set     : {X_test_run.shape[0]:,} samples  (25%)")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — BASELINE SVM
    #   Linear kernel, C=1.0 (default), raw features, no scaling,
    #   no class balancing.  Deliberately suboptimal to set the floor.
    #
    #   EXPECTED WEAKNESSES:
    #     1. No feature scaling — high-range features dominate
    #     2. Linear kernel — cannot capture non-linear boundaries
    #     3. No class balancing — minority classes ignored
    # ════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 1 — BASELINE SVM (seed={seed})")

    svm_baseline = SVC(kernel="linear", C=1.0, random_state=seed, max_iter=10000)
    m1, pred1 = evaluate(svm_baseline, X_train, X_test_run, y_train, y_test_run,
                         "Phase 1: Baseline", seed)
    all_run_metrics["Phase 1: Baseline"].append(m1)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — IMPROVED SVM
    #   + StandardScaler normalisation (zero-mean, unit-variance)
    #   + ANOVA F-test feature selection (top 30 features)
    #   + RBF kernel (non-linear decision boundaries)
    #   + class_weight='balanced' (up-weight minority classes)
    #
    #   Each improvement addresses a specific Phase-1 weakness:
    #     StandardScaler  → feature magnitude imbalance
    #     ANOVA selection → high dimensionality / redundancy
    #     RBF kernel      → non-linear class boundaries
    #     balanced weight → class imbalance
    # ════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 2 — IMPROVED SVM (seed={seed})")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test_run)

    # Feature Selection
    selector = SelectKBest(f_classif, k=30)
    X_train_sel = selector.fit_transform(X_train_sc, y_train)
    X_test_sel_run = selector.transform(X_test_sc)

    # Record selected features (from last run, for figures)
    selected_mask_run   = selector.get_support()
    selected_names_run  = X.columns[selected_mask_run].tolist()
    selected_scores_run = selector.scores_[selected_mask_run]

    print(f"  Selected top 30 features by ANOVA F-score:")
    rank = np.argsort(-selected_scores_run)
    for i, idx in enumerate(rank):
        print(f"      {i+1:2d}. {selected_names_run[idx]:<30s}  F = {selected_scores_run[idx]:>12,.1f}")

    svm_improved = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", random_state=seed, max_iter=15000,
    )
    m2, pred2 = evaluate(svm_improved, X_train_sel, X_test_sel_run,
                         y_train, y_test_run, "Phase 2: Improved", seed)
    all_run_metrics["Phase 2: Improved"].append(m2)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3 — OPTIMIZED SVM (GridSearchCV)
    #   Exhaustive search over C × gamma × kernel with 5-fold
    #   stratified CV scored by weighted-F1.
    #
    #   Search space:
    #     C      : {0.1, 1, 10, 100}
    #     gamma  : {scale, auto, 0.01, 0.1}
    #     kernel : {rbf, poly}
    #   Total fits: 4 × 4 × 2 × 5 folds = 160 per seed
    # ════════════════════════════════════════════════════════════════════
    print(f"\n  PHASE 3 — OPTIMIZED SVM (seed={seed})")

    param_grid = {
        "C":      [0.1, 1, 10, 100],
        "gamma":  ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "poly"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    grid_run = GridSearchCV(
        estimator=SVC(class_weight="balanced", random_state=seed, max_iter=20000),
        param_grid=param_grid, cv=cv,
        scoring="f1_weighted", n_jobs=-1, verbose=1, refit=True,
    )

    print(f"  Running grid search (32 combos x 5 folds = 160 fits)...")
    t0 = time.time()
    grid_run.fit(X_train_sel, y_train)
    grid_time = time.time() - t0

    print(f"  Grid search completed in {grid_time:.1f}s")
    print(f"  Best parameters : {grid_run.best_params_}")
    print(f"  Best CV F1 (wt) : {grid_run.best_score_:.4f}")

    best_model_run = grid_run.best_estimator_
    pred3 = best_model_run.predict(X_test_sel_run)

    acc3  = accuracy_score(y_test_run, pred3)
    f1w3  = f1_score(y_test_run, pred3, average="weighted", zero_division=0)
    f1m3  = f1_score(y_test_run, pred3, average="macro",    zero_division=0)
    prec3 = precision_score(y_test_run, pred3, average="weighted", zero_division=0)
    rec3  = recall_score(y_test_run, pred3, average="weighted", zero_division=0)

    print(f"\n  {'─' * 50}")
    print(f"  Phase 3: Optimized [seed {seed}]")
    print(f"  {'─' * 50}")
    print(f"    Accuracy              {acc3:.4f}")
    print(f"    F1 Score (weighted)   {f1w3:.4f}")
    print(f"    F1 Score (macro)      {f1m3:.4f}")
    print(f"    Precision (weighted)  {prec3:.4f}")
    print(f"    Recall (weighted)     {rec3:.4f}")
    print(f"    Grid search time      {grid_time:.1f}s")
    print(f"  {'─' * 50}")

    print("\n  Per-class Classification Report:")
    print(classification_report(
        y_test_run, pred3, target_names=le.classes_, zero_division=0
    ))

    m3 = dict(accuracy=acc3, f1_weighted=f1w3, f1_macro=f1m3,
              precision=prec3, recall=rec3, time=grid_time)
    all_run_metrics["Phase 3: Optimized"].append(m3)

    # Keep last run's objects for figure generation
    pred_p1 = pred1
    pred_p2 = pred2
    pred_p3 = pred3
    y_test = y_test_run
    X_test_sel = X_test_sel_run
    best_model = best_model_run
    grid = grid_run
    selected_names = selected_names_run
    selected_scores = selected_scores_run
    selected_mask = selected_mask_run


###############################################################################
#                                                                             #
#   MULTI-SEED AVERAGED RESULTS                                              #
#                                                                             #
#   Compute the mean and standard deviation of each metric across all seeds. #
#   These averaged results are the final reported numbers.                   #
#                                                                             #
###############################################################################

print("\n" + "=" * 72)
print(f"  AVERAGED RESULTS ACROSS {N_SEEDS} SEEDS: {SEEDS}")
print("=" * 72)

metric_names = ["accuracy", "f1_weighted", "f1_macro", "precision", "recall"]

# Build the averaged results dict (used by figures and summary table)
results = {}
for phase_name in all_run_metrics:
    runs = all_run_metrics[phase_name]
    avg = {}
    for m in metric_names:
        vals = [r[m] for r in runs]
        avg[m] = np.mean(vals)
        avg[f"{m}_std"] = np.std(vals)
    avg["time"] = np.mean([r["time"] for r in runs])

    print(f"\n  {'─' * 60}")
    print(f"  {phase_name}  (mean ± std over {N_SEEDS} runs)")
    print(f"  {'─' * 60}")
    for m in metric_names:
        print(f"    {m:<22s}  {avg[m]:.4f} ± {avg[f'{m}_std']:.4f}")
    print(f"    {'time':<22s}  {avg['time']:.2f}s")
    print(f"  {'─' * 60}")

    results[phase_name] = avg

# Attach last run's predictions to results for figure generation
results["Phase 1: Baseline"]["y_pred"]  = pred_p1
results["Phase 2: Improved"]["y_pred"]  = pred_p2
results["Phase 3: Optimized"]["y_pred"] = pred_p3

# ── Per-seed summary table ───────────────────────────────────────────────
print(f"\n  Per-seed breakdown:")
for phase_name in all_run_metrics:
    print(f"\n  {phase_name}:")
    print(f"    {'Seed':<8s}  {'Accuracy':>10s}  {'F1(wt)':>10s}  {'F1(macro)':>10s}  {'Precision':>10s}  {'Recall':>10s}")
    print(f"    {'─' * 62}")
    for seed, run in zip(SEEDS, all_run_metrics[phase_name]):
        print(f"    {seed:<8d}  {run['accuracy']:>10.4f}  {run['f1_weighted']:>10.4f}  "
              f"{run['f1_macro']:>10.4f}  {run['precision']:>10.4f}  {run['recall']:>10.4f}")
    avgs = [np.mean([r[m] for r in all_run_metrics[phase_name]]) for m in metric_names]
    print(f"    {'AVERAGE':<8s}  {avgs[0]:>10.4f}  {avgs[1]:>10.4f}  {avgs[2]:>10.4f}  {avgs[3]:>10.4f}  {avgs[4]:>10.4f}")


# REPORT USAGE — Figure 4: Feature Importance (ANOVA F-Scores)
# Include in the "Feature Selection" subsection.  This horizontal bar chart
# ranks the 30 selected features by their ANOVA F-score.  Key observations
# to discuss: Protocol is by far the most discriminative feature (F ≈ 156K),
# followed by RST Flag Count and FIN Flag Count.  This makes intuitive
# sense — different attack types use distinct protocols and produce
# characteristic TCP flag patterns (e.g., port scans trigger RST flags,
# starvation attacks maintain open connections with ACK/PSH).
fig, ax = plt.subplots(figsize=(10, 8))
order = np.argsort(selected_scores)
bars = ax.barh(
    np.array(selected_names)[order],
    selected_scores[order],
    color="#3498db", edgecolor="white", linewidth=0.5,
)
# Add value labels at the end of each bar
for bar, val in zip(bars, selected_scores[order]):
    ax.text(bar.get_width() + max(selected_scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}", va="center", fontsize=7.5)
ax.set_xlabel("ANOVA F-Score")
ax.set_title("Figure 4 — Top 30 Features Ranked by ANOVA F-test Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig04_feature_importance.png", bbox_inches="tight")
plt.close()
print("  Saved fig04_feature_importance.png")


###############################################################################
#                                                                             #
#   SECTION 7 — REPORT-READY VISUALISATIONS                                  #
#                                                                             #
#   All figures are saved at 300 dpi with tight bounding boxes, suitable     #
#   for direct insertion into a Word/LaTeX report.  Each figure has a        #
#   REPORT USAGE comment explaining what it shows and how to discuss it.     #
#                                                                             #
#   NOTE: Figures use the AVERAGED metrics for bar charts and trajectories,  #
#   and the LAST seed's predictions for confusion matrices and PCA scatter.  #
#                                                                             #
###############################################################################

print("\n" + "=" * 72)
print("  SECTION 7 — GENERATING REPORT FIGURES")
print("=" * 72)

phases      = list(results.keys())
short_names = ["Phase 1\nBaseline", "Phase 2\nImproved", "Phase 3\nOptimized"]
colors3     = ["#e74c3c", "#f39c12", "#27ae60"]
pred_list   = [pred_p1, pred_p2, pred_p3]
phase_titles = ["Phase 1: Baseline", "Phase 2: Improved", "Phase 3: Optimized"]


# ── Figure 5: Performance Metrics Side-by-Side + F1-Macro Trajectory ─────
# REPORT USAGE: The primary comparison figure — use in "Results" section.
# Left panel compares all four aggregate metrics across the three phases.
# Right panel zooms in on F1 (macro), the most challenging metric because
# it weights all classes equally — forcing the model to perform well even
# on MITM (26 samples) and flood (108 samples).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

metric_keys   = ["accuracy", "f1_weighted", "precision", "recall"]
metric_labels = ["Accuracy", "F1 (weighted)", "Precision", "Recall"]
x = np.arange(len(metric_keys))
w = 0.24

for i, (phase, lbl, clr) in enumerate(zip(phases, short_names, colors3)):
    vals = [results[phase][m] for m in metric_keys]
    ax1.bar(x + i * w, vals, w, label=lbl.replace("\n", " "), color=clr,
            edgecolor="white", linewidth=0.5)
    for xi, v in zip(x + i * w, vals):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5)

ax1.set_xticks(x + w)
ax1.set_xticklabels(metric_labels)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Score")
ax1.set_title("Figure 5a — Aggregate Metrics Across Phases")
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(axis="y", alpha=0.3)

f1_macros = [results[p]["f1_macro"] for p in phases]
ax2.plot(short_names, f1_macros, "o-", color="#2c3e50", lw=2.5, ms=12,
         markerfacecolor="#3498db", markeredgecolor="#2c3e50", markeredgewidth=1.5)
for i, v in enumerate(f1_macros):
    ax2.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("F1 Score (macro)")
ax2.set_title("Figure 5b — F1 (Macro) Improvement Trajectory")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig05_performance_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved fig05_performance_comparison.png")


# ── Figure 6: Confusion Matrices (normalised, one per phase) ────────────
# REPORT USAGE: Include in "Results" or a dedicated "Confusion Matrix
# Analysis" subsection.  Normalisation by true-class row sums means each
# cell shows "what percentage of [true class] was predicted as [predicted
# class]".  This makes classes of very different sizes comparable.
# Key observation: look at how the diagonal values for minority classes
# (flood, MITM) improve from Phase 1 → 3, and how off-diagonal "leaks"
# (e.g., flood misclassified as attackfree) shrink.
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, title, yp in zip(axes, phase_titles, pred_list):
    cm = confusion_matrix(y_test, yp)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, cbar_kws={"label": "% of true class"},
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

plt.suptitle("Figure 6 — Normalised Confusion Matrices (% of True Class)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig06_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved fig06_confusion_matrices.png")


# ── Figure 7: Per-Class F1 Score Comparison ──────────────────────────────
# REPORT USAGE: Shows which specific attack types benefited most from each
# improvement phase.  Typically the classes that gain the most are:
#   - floodattack:  from near-random in Phase 1 to 0.80+ in Phase 3
#   - mitmattack:   similarly dramatic improvement
# Meanwhile, fuzzyattack and portscan (large classes) are often already
# well-classified in Phase 1 and show marginal gains.
fig, ax = plt.subplots(figsize=(13, 6))
x_cls = np.arange(len(le.classes_))

for i, (phase, clr, lbl) in enumerate(zip(phases, colors3, short_names)):
    f1_per = f1_score(y_test, results[phase]["y_pred"],
                      average=None, zero_division=0)
    bars = ax.bar(x_cls + i * 0.25, f1_per, 0.25,
                  label=lbl.replace("\n", " "), color=clr,
                  edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, f1_per):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=7)

ax.set_xticks(x_cls + 0.25)
ax.set_xticklabels(le.classes_, rotation=35, ha="right")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.18)
ax.set_title("Figure 7 — Per-Class F1 Score Across Improvement Phases")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig07_per_class_f1.png", bbox_inches="tight")
plt.close()
print("  Saved fig07_per_class_f1.png")


# ── Figure 8: PCA Scatter — Predicted Labels in 2-D Space ───────────────
# REPORT USAGE: Provides visual intuition of how the SVM partitions the
# feature space when projected down to 2 principal components.  In Phase 1,
# many points are incorrectly coloured (misclassified); by Phase 3, clusters
# are tighter and correctly coloured.  NOTE: PCA is applied ONLY for
# visualisation — the SVM operates in the full 30-D feature space.
pca = PCA(n_components=2, random_state=42)
X_test_2d = pca.fit_transform(X_test_sel)
ev = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, title, yp in zip(axes, phase_titles, pred_list):
    ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=yp,
               cmap="tab10", alpha=0.55, s=14, edgecolors="none")
    ax.set_xlabel(f"PC 1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC 2 ({ev[1]*100:.1f}% var)")
    ax.set_title(title)

handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=plt.cm.tab10(i / 10), markersize=8, label=c)
           for i, c in enumerate(le.classes_)]
fig.legend(handles=handles, loc="lower center", ncol=len(le.classes_),
           fontsize=9, bbox_to_anchor=(0.5, -0.04))
plt.suptitle("Figure 8 — PCA Projection of Test Set (Predicted Labels)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig08_pca_scatter.png", bbox_inches="tight")
plt.close()
print("  Saved fig08_pca_scatter.png")


# ── Figure 9: GridSearchCV Heatmap (C vs Gamma) ─────────────────────────
# REPORT USAGE: Include in "Hyperparameter Tuning" subsection.  Shows the
# mean cross-validated F1 score for each (C, γ) combination under the best
# kernel.  A broad dark region indicates robustness (the model isn't overly
# sensitive to parameter choice).  An isolated dark cell would suggest the
# optimum is fragile and may not generalise.
cv_df = pd.DataFrame(grid.cv_results_)
best_kernel = grid.best_params_["kernel"]
kern_df = cv_df[cv_df["param_kernel"] == best_kernel].copy()

if len(kern_df) > 0:
    kern_df["param_gamma"] = kern_df["param_gamma"].astype(str)
    kern_df["param_C"]     = kern_df["param_C"].astype(str)
    pivot = kern_df.pivot_table(values="mean_test_score",
                                index="param_C", columns="param_gamma")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.8, linecolor="white", ax=ax)
    ax.set_title(f"Figure 9 — GridSearchCV F1 (weighted): {best_kernel.upper()} Kernel\n"
                 f"Rows = C  |  Columns = gamma")
    ax.set_ylabel("C (regularisation)")
    ax.set_xlabel("gamma (kernel bandwidth)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig09_gridsearch_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig09_gridsearch_heatmap.png")


# ── Figure 10: Learning Curve (Optimised Model) ─────────────────────────
# REPORT USAGE: Include in "Model Evaluation" or "Discussion" section.
# The learning curve plots training and validation scores as a function of
# training set size.  It answers two key questions:
#   1) Does the model suffer from high bias (both curves plateau low)?
#      → The model is too simple; consider more features or a different kernel.
#   2) Does the model suffer from high variance (large gap between curves)?
#      → The model is overfitting; consider more data or stronger regularisation.
# A well-tuned model shows both curves converging at a high score.
print("  Computing learning curve (may take a moment)...")
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_sel, y_train,
    cv=3,                       # 3-fold CV for speed
    scoring="f1_weighted",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
    random_state=42,
)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#27ae60",
        label="Training score", lw=2)
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color="#27ae60")

ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#e74c3c",
        label="Cross-validation score", lw=2)
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color="#e74c3c")

ax.set_xlabel("Training Set Size")
ax.set_ylabel("F1 Score (weighted)")
ax.set_title("Figure 10 — Learning Curve (Optimised SVM)")
ax.legend(loc="lower right")
ax.set_ylim(0.85, 1.02)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig10_learning_curve.png", bbox_inches="tight")
plt.close()
print("  Saved fig10_learning_curve.png")


###############################################################################
#                                                                             #
#   SECTION 8 — FINAL IMPROVEMENT SUMMARY                                    #
#                                                                             #
###############################################################################

print("\n" + "=" * 72)
print("  FINAL IMPROVEMENT SUMMARY")
print("=" * 72)

# ── Summary Table (AVERAGED across seeds) ────────────────────────────────
summary = pd.DataFrame({
    p: {k: v for k, v in m.items() if k != "y_pred" and not k.endswith("_std")}
    for p, m in results.items()
}).T
summary.index.name = "Phase"

print(f"\n  Averaged metrics across {N_SEEDS} seeds: {SEEDS}\n")
print(summary.to_string(float_format="{:.4f}".format))

# ── Improvement Deltas (Baseline → Optimized, averaged) ─────────────────
baseline  = results["Phase 1: Baseline"]
optimized = results["Phase 3: Optimized"]

print(f"\n  Improvement from Baseline to Optimized (averaged):")
print(f"  {'Metric':<20s}  {'Baseline':>10s}  {'Optimized':>10s}  {'Delta':>10s}  {'% Change':>10s}")
print(f"  {'─' * 65}")
for m in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
    b, o = baseline[m], optimized[m]
    d = o - b
    pct = d / max(b, 1e-9) * 100
    sign = "+" if d >= 0 else ""
    print(f"  {m:<20s}  {b:>10.4f}  {o:>10.4f}  {sign}{d:>9.4f}  {sign}{pct:>9.1f}%")

# ── Best Model Details ───────────────────────────────────────────────────
print(f"\n  Best model configuration:")
print(f"    Kernel             : {grid.best_params_['kernel']}")
print(f"    C                  : {grid.best_params_['C']}")
print(f"    gamma              : {grid.best_params_['gamma']}")
print(f"    class_weight       : balanced")
print(f"    Features used      : {X_train_sel.shape[1]} (ANOVA-selected)")
print(f"    Best CV F1 (wt)    : {grid.best_score_:.4f}")

# ── List all saved figures ───────────────────────────────────────────────
print(f"\n  All figures saved to: {OUTPUT_DIR}/")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith(".png"):
        print(f"    {fname}")

print("\n" + "=" * 72)
print("  DONE — Implementation complete")
print("=" * 72)
