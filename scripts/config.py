from __future__ import annotations

import os
from pathlib import Path

# -----------------------------
# Input / output paths
# -----------------------------
DATA_DIR = "data"
UKBANK_DIR = os.path.join(DATA_DIR, "example")
MODEL_DIR = os.path.join(DATA_DIR, "out")

INPUT_PROTEIN_FILE = os.path.join(UKBANK_DIR, "example_protein.csv")
PROTEIN_COLUMNS_FILE = None
DEFAULT_OUTPUT_DIR = os.path.join(MODEL_DIR, "results/mlp_demo")

# -----------------------------
# Column names
# -----------------------------
AGE_COLUMN = "Age"
SEX_COLUMN = "Sex"
DEMOGRAPHIC_COLS = ["Participant ID", "Age", "Month of Birth", "Sex"]

# -----------------------------
# Data filtering and preprocessing
# -----------------------------
SAMPLE_MISSING_THRESHOLD = 0.20
FEATURE_MISSING_THRESHOLD = 0.20
WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995
AGE_STRATIFY_BINS_MAX = 10

# 70/10/20 overall split.
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_DEV = 0.125  # 0.8 * 0.125 = 0.10 overall

# -----------------------------
# MLP architecture and training
# -----------------------------
RANDOM_SEED = 42
DEFAULT_HIDDEN_LAYERS = [128, 64, 32]
DEFAULT_DROPOUT = 0.20
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_DEVICE = "auto"

# -----------------------------
# Post-training analysis defaults
# -----------------------------
CV_FOLDS = 5
IG_STEPS = 100
BOOTSTRAP_RESAMPLES = 1000

# -----------------------------
# Output file names
# -----------------------------
MODEL_FILE = "model.pth"
PREPROCESSOR_FILE = "preprocessor.pkl"
SPLITS_FILE = "split_indices.csv"
FEATURE_NAMES_FILE = "feature_names.txt"
TRAIN_HISTORY_FILE = "training_history.csv"
TRAIN_METRICS_FILE = "test_metrics.json"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"
RUN_METADATA_FILE = "run_metadata.json"

# Held-out IG
IG_OUTPUT_FILE = "ig_global_importance.csv"
IG_PER_SAMPLE_FILE = "ig_per_sample.npy"
IG_PERMUTATION_FILE = "ig_permutation_summary.csv"
IG_HELDOUT_PLOT_FILE = "ig_top20_heldout.png"

# CV artifacts and plots
CV_DIRNAME = "cv_folds"
CV_METRICS_FILE = "cv_fold_metrics.csv"
CV_SUMMARY_FILE = "cv_summary.json"
CV_HISTORIES_FILE = "cv_fold_histories.csv"
CV_EPOCH_SUMMARY_FILE = "cv_epoch_summary.csv"
CV_MSE_PLOT_FILE = "cv_training_mse_curve.png"
CV_R2_PLOT_FILE = "cv_training_r2_curve.png"
CV_FOLD_SPLITS_FILE = "fold_row_indices.csv"
CV_FOLD_PREDICTIONS_FILE = "fold_eval_predictions.csv"

# CV-based IG outputs
IG_CV_OUTPUT_FILE = "ig_cv_global_importance.csv"
IG_CV_FOLD_FILE = "ig_cv_fold_importance.csv"
IG_CV_PLOT_FILE = "ig_top20_cv_errorbar.png"

# Bootstrap outputs
BOOTSTRAP_RAW_FILE = "bootstrap_metrics.csv"
BOOTSTRAP_SUMMARY_FILE = "bootstrap_summary.csv"


def ensure_output_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)
