from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Preprocessor:
    protein_cols: List[str]
    medians: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    means: pd.Series
    stds: pd.Series

    def save(self, path: str | os.PathLike) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | os.PathLike) -> "Preprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProteinExpressionNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (128, 64, 32),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def read_input_table(input_file: str) -> pd.DataFrame:
    try:
        return pd.read_csv(input_file)
    except Exception:
        return pd.read_csv(input_file, sep="\t")


def load_protein_columns(
    df: pd.DataFrame,
    demographic_cols: Sequence[str],
    protein_columns_file: Optional[str] = None,
) -> List[str]:
    if protein_columns_file:
        protein_cols = pd.read_csv(protein_columns_file, header=None)[0].astype(str).tolist()
        protein_cols = [col for col in protein_cols if col in df.columns]
    else:
        protein_cols = [col for col in df.columns if col not in set(demographic_cols)]
    if not protein_cols:
        raise ValueError("No valid protein columns were found.")
    return protein_cols


def clean_dataset(
    df: pd.DataFrame,
    protein_cols: Sequence[str],
    age_column: str,
    sex_column: Optional[str],
    sample_missing_threshold: float,
    feature_missing_threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    for col in protein_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if age_column not in df.columns:
        raise ValueError(f"Required age column '{age_column}' was not found.")
    df[age_column] = pd.to_numeric(df[age_column], errors="coerce")
    df = df.loc[df[age_column].notna()].copy()
    if df.empty:
        raise ValueError("No samples remain after dropping rows with missing age.")

    if sex_column and sex_column in df.columns:
        df[sex_column] = df[sex_column].fillna("UNK").astype(str)

    sample_missing = df[list(protein_cols)].isna().mean(axis=1)
    df = df.loc[sample_missing <= sample_missing_threshold].copy()
    if df.empty:
        raise ValueError("No samples remain after sample-level missingness filtering.")

    feature_missing = df[list(protein_cols)].isna().mean(axis=0)
    kept_proteins = feature_missing[feature_missing <= feature_missing_threshold].index.tolist()
    if not kept_proteins:
        raise ValueError("No protein features remain after feature-level missingness filtering.")

    return df, kept_proteins


def _build_combined_strata(
    df: pd.DataFrame,
    age_column: str,
    sex_column: Optional[str],
    n_bins: int,
) -> pd.Series:
    age = pd.to_numeric(df[age_column], errors="coerce")
    age_bins = pd.qcut(age, q=n_bins, duplicates="drop")
    age_bins = age_bins.astype(str)
    if sex_column and sex_column in df.columns:
        sex = df[sex_column].fillna("UNK").astype(str)
        return age_bins + "__" + sex
    return age_bins


def create_train_val_test_splits(
    df: pd.DataFrame,
    age_column: str,
    sex_column: Optional[str],
    test_size: float,
    val_size_within_dev: float,
    random_seed: int,
    max_age_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    index = df.index.to_numpy()
    for n_bins in range(max_age_bins, 1, -1):
        try:
            strata = _build_combined_strata(df, age_column, sex_column, n_bins)
            train_dev_idx, test_idx = train_test_split(
                index,
                test_size=test_size,
                random_state=random_seed,
                stratify=strata,
            )
            dev_strata = strata.loc[train_dev_idx]
            train_idx, val_idx = train_test_split(
                train_dev_idx,
                test_size=val_size_within_dev,
                random_state=random_seed,
                stratify=dev_strata,
            )
            return train_idx, val_idx, test_idx, n_bins
        except ValueError:
            continue

    train_dev_idx, test_idx = train_test_split(index, test_size=test_size, random_state=random_seed)
    train_idx, val_idx = train_test_split(train_dev_idx, test_size=val_size_within_dev, random_state=random_seed)
    return train_idx, val_idx, test_idx, 0


def fit_preprocessor(
    train_df: pd.DataFrame,
    protein_cols: Sequence[str],
    winsor_lower_quantile: float,
    winsor_upper_quantile: float,
) -> Preprocessor:
    x = train_df[list(protein_cols)].apply(pd.to_numeric, errors="coerce")
    medians = x.median(axis=0)
    x = x.fillna(medians)
    lower_bounds = x.quantile(winsor_lower_quantile, axis=0)
    upper_bounds = x.quantile(winsor_upper_quantile, axis=0)
    x = x.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0, 1.0)
    return Preprocessor(
        protein_cols=list(protein_cols),
        medians=medians,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        means=means,
        stds=stds,
    )


def transform_features(df: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    x = df[preprocessor.protein_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(preprocessor.medians)
    x = x.clip(
        lower=preprocessor.lower_bounds,
        upper=preprocessor.upper_bounds,
        axis=1,
    )
    x = (x - preprocessor.means) / preprocessor.stds
    return x.to_numpy(dtype=np.float32)


def build_tensor_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)).reshape(-1, 1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    truth: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().reshape(-1)
        preds.append(out)
        truth.append(yb.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(truth)


def compute_metrics(preds: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(truth, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(truth, preds))
    r2 = float(r2_score(truth, preds))
    if len(np.unique(truth)) > 1 and len(np.unique(preds)) > 1:
        pearson_r = float(np.corrcoef(truth, preds)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        train_preds: List[np.ndarray] = []
        train_truth: List[np.ndarray] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_n = xb.size(0)
            train_loss += loss.item() * batch_n
            n_train += batch_n
            train_preds.append(pred.detach().cpu().numpy().reshape(-1))
            train_truth.append(yb.detach().cpu().numpy().reshape(-1))
        train_loss /= max(n_train, 1)
        train_preds_arr = np.concatenate(train_preds) if train_preds else np.array([], dtype=float)
        train_truth_arr = np.concatenate(train_truth) if train_truth else np.array([], dtype=float)
        train_metrics = compute_metrics(train_preds_arr, train_truth_arr) if len(train_truth_arr) else {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
        }

        model.eval()
        val_loss = 0.0
        n_val = 0
        val_preds: List[np.ndarray] = []
        val_truth: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                batch_n = xb.size(0)
                val_loss += loss.item() * batch_n
                n_val += batch_n
                val_preds.append(pred.detach().cpu().numpy().reshape(-1))
                val_truth.append(yb.detach().cpu().numpy().reshape(-1))
        val_loss /= max(n_val, 1)
        val_preds_arr = np.concatenate(val_preds) if val_preds else np.array([], dtype=float)
        val_truth_arr = np.concatenate(val_truth) if val_truth else np.array([], dtype=float)
        val_metrics = compute_metrics(val_preds_arr, val_truth_arr) if len(val_truth_arr) else {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson_r": float("nan"),
        }

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "train_r2": train_metrics["r2"],
                "val_r2": val_metrics["r2"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "train_pearson_r": train_metrics["pearson_r"],
                "val_pearson_r": val_metrics["pearson_r"],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model, history

def save_history(history: Sequence[Dict[str, float]], path: str | os.PathLike) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_json(data: Dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_splits(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    path: str | os.PathLike,
) -> None:
    split_rows = (
        [(int(i), "train") for i in train_idx]
        + [(int(i), "val") for i in val_idx]
        + [(int(i), "test") for i in test_idx]
    )
    split_df = pd.DataFrame(split_rows, columns=["row_index", "split"])\
        .sort_values(["split", "row_index"])\
        .reset_index(drop=True)
    split_df.to_csv(path, index=False)


def load_splits(path: str | os.PathLike) -> Dict[str, List[int]]:
    split_df = pd.read_csv(path)
    return {
        split: split_df.loc[split_df["split"] == split, "row_index"].astype(int).tolist()
        for split in ["train", "val", "test"]
    }


def resolve_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_metrics(
    truth: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    n = len(truth)
    rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        rows.append(compute_metrics(preds[idx], truth[idx]))
    return pd.DataFrame(rows)


def summarize_bootstrap(boot_df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for metric in boot_df.columns:
        vals = boot_df[metric].dropna().to_numpy()
        summary.append(
            {
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
            }
        )
    return pd.DataFrame(summary)


def integrated_gradients_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 100,
) -> torch.Tensor:
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    assert baseline.shape == inputs.shape

    model.eval()
    total_gradients = torch.zeros_like(inputs)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=inputs.device)[1:]:
        x = baseline + alpha * (inputs - baseline)
        x.requires_grad_(True)
        outputs = model(x)
        gradients = torch.autograd.grad(outputs.sum(), x, retain_graph=False, create_graph=False)[0]
        total_gradients += gradients.detach()
    avg_gradients = total_gradients / steps
    return (inputs - baseline) * avg_gradients


def compute_global_ig(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    abs_sum = None
    sample_count = 0
    per_sample_scores: List[np.ndarray] = []

    for xb, _ in loader:
        xb = xb.to(device)
        baseline = torch.zeros_like(xb)
        ig = integrated_gradients_batch(model, xb, baseline=baseline, steps=steps)
        ig_cpu = ig.detach().cpu().numpy()
        per_sample_scores.append(ig_cpu)
        batch_abs = np.abs(ig_cpu).sum(axis=0)
        abs_sum = batch_abs if abs_sum is None else abs_sum + batch_abs
        sample_count += ig_cpu.shape[0]

    global_importance = abs_sum / max(sample_count, 1)
    per_sample = np.concatenate(per_sample_scores, axis=0)
    return global_importance, per_sample


def build_cv_labels(
    df: pd.DataFrame,
    age_column: str,
    sex_column: Optional[str],
    max_age_bins: int,
) -> Tuple[Optional[pd.Series], int]:
    for bins in range(max_age_bins, 1, -1):
        try:
            labels = _build_combined_strata(df, age_column, sex_column, bins)
            counts = labels.value_counts()
            if (counts >= 2).all():
                return labels, bins
        except ValueError:
            continue
    return None, 0


__all__ = [
    "Preprocessor",
    "ProteinExpressionNet",
    "bootstrap_metrics",
    "build_cv_labels",
    "build_tensor_loader",
    "clean_dataset",
    "compute_global_ig",
    "compute_metrics",
    "create_train_val_test_splits",
    "fit_preprocessor",
    "integrated_gradients_batch",
    "load_protein_columns",
    "load_splits",
    "predict",
    "read_input_table",
    "resolve_device",
    "save_history",
    "save_json",
    "save_splits",
    "set_seed",
    "summarize_bootstrap",
    "train_model",
    "transform_features",
]
