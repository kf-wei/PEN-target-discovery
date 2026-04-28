from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config_mlp_aligned as cfg
from mlp_common_aligned import (
    Preprocessor,
    ProteinExpressionNet,
    build_tensor_loader,
    clean_dataset,
    compute_global_ig,
    load_protein_columns,
    load_splits,
    read_input_table,
    resolve_device,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated Gradients for the aligned PEN MLP. Supports held-out test IG and "
            "CV-fold aggregated IG with Fig.4a-style error bars."
        )
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default=cfg.INPUT_PROTEIN_FILE)
    parser.add_argument("--protein_columns_file", type=str, default=cfg.PROTEIN_COLUMNS_FILE)
    parser.add_argument("--device", type=str, default=cfg.DEFAULT_DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=cfg.IG_STEPS)
    parser.add_argument("--batch_size", type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument("--mode", type=str, default="both", choices=["heldout", "cv", "both"])
    parser.add_argument("--cv_dir", type=str, default=None)
    parser.add_argument("--top_n_plot", type=int, default=20)
    parser.add_argument("--cv_error_bar", type=str, default="se", choices=["se", "sd"])
    parser.add_argument("--torch_num_threads", type=int, default=1)
    parser.add_argument("--run_permutation_test", action="store_true")
    parser.add_argument(
        "--permute_top_n",
        type=int,
        default=0,
        help="Optional held-out null IG for the top N proteins only. 0 = skip.",
    )
    parser.add_argument("--n_permutations", type=int, default=100)
    return parser.parse_args()


def load_run_metadata(run_dir: str) -> dict:
    with open(os.path.join(run_dir, cfg.RUN_METADATA_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_dataset(
    input_file: str,
    protein_columns_file: str | None,
) -> tuple[pd.DataFrame, List[str]]:
    df = read_input_table(input_file)
    protein_cols = load_protein_columns(
        df=df,
        demographic_cols=cfg.DEMOGRAPHIC_COLS,
        protein_columns_file=protein_columns_file,
    )
    df_clean, protein_cols = clean_dataset(
        df=df,
        protein_cols=protein_cols,
        age_column=cfg.AGE_COLUMN,
        sex_column=cfg.SEX_COLUMN,
        sample_missing_threshold=cfg.SAMPLE_MISSING_THRESHOLD,
        feature_missing_threshold=cfg.FEATURE_MISSING_THRESHOLD,
    )
    return df_clean, protein_cols


def reconstruct_test_arrays(
    df_clean: pd.DataFrame,
    run_dir: str,
) -> tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame, Preprocessor]:
    splits = load_splits(os.path.join(run_dir, cfg.SPLITS_FILE))
    test_df = df_clean.loc[splits["test"]].copy()
    preprocessor = Preprocessor.load(os.path.join(run_dir, cfg.PREPROCESSOR_FILE))
    x_test = transform_features(test_df, preprocessor)
    y_test = test_df[cfg.AGE_COLUMN].to_numpy(dtype="float32")
    return x_test, y_test, preprocessor.protein_cols, test_df, preprocessor


def plot_top_ig(
    ig_df: pd.DataFrame,
    save_path: str,
    top_n: int,
    title: str,
    error_col: str | None = None,
) -> None:
    plot_df = ig_df.head(top_n).copy().iloc[::-1]
    errors = plot_df[error_col].to_numpy(dtype=float) if error_col and error_col in plot_df.columns else None

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["mean_abs_ig"], xerr=errors)
    plt.xlabel("Mean absolute IG")
    plt.ylabel("Protein")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def permutation_test(
    model: ProteinExpressionNet,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    steps: int,
    batch_size: int,
    observed_df: pd.DataFrame,
    top_n: int,
    n_permutations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    if top_n <= 0:
        return pd.DataFrame()
    selected = observed_df.head(top_n).copy()
    summaries = []

    for _, row in selected.iterrows():
        feature = row["feature"]
        j = feature_names.index(feature)
        null_vals = []
        for _ in range(n_permutations):
            x_perm = x_test.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            perm_loader = build_tensor_loader(x_perm, y_test, batch_size=batch_size, shuffle=False)
            global_importance, _ = compute_global_ig(model, perm_loader, device=device, steps=steps)
            null_vals.append(float(global_importance[j]))
        null_vals = np.asarray(null_vals, dtype=float)
        summaries.append(
            {
                "feature": feature,
                "observed_mean_abs_ig": float(row["mean_abs_ig"]),
                "null_mean": float(null_vals.mean()),
                "null_std": float(null_vals.std(ddof=0)),
                "null_p999": float(np.quantile(null_vals, 0.999)) if len(null_vals) else np.nan,
                "empirical_p": float((np.sum(null_vals >= row["mean_abs_ig"]) + 1) / (len(null_vals) + 1)),
                "n_permutations": int(n_permutations),
            }
        )
        print(f"[INFO] Permutation test complete for {feature}")
    return pd.DataFrame(summaries)


def compute_heldout_ig(
    args: argparse.Namespace,
    run_meta: dict,
    df_clean: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    x_test, y_test, feature_names, _, _ = reconstruct_test_arrays(df_clean=df_clean, run_dir=args.run_dir)

    model = ProteinExpressionNet(
        input_size=len(feature_names),
        hidden_sizes=run_meta["hidden_layers"],
        dropout_rate=run_meta["dropout"],
    ).to(device)
    state = torch.load(os.path.join(args.run_dir, cfg.MODEL_FILE), map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = build_tensor_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False)
    global_importance, per_sample = compute_global_ig(
        model=model,
        loader=test_loader,
        device=device,
        steps=args.steps,
    )

    ig_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_ig": global_importance,
        }
    ).sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    ig_df.to_csv(os.path.join(args.run_dir, cfg.IG_OUTPUT_FILE), index=False)
    np.save(os.path.join(args.run_dir, cfg.IG_PER_SAMPLE_FILE), per_sample)
    plot_top_ig(
        ig_df=ig_df,
        save_path=os.path.join(args.run_dir, cfg.IG_HELDOUT_PLOT_FILE),
        top_n=args.top_n_plot,
        title="Held-out test IG (Top proteins)",
        error_col=None,
    )

    if args.run_permutation_test and args.permute_top_n > 0:
        perm_df = permutation_test(
            model=model,
            x_test=x_test,
            y_test=y_test,
            feature_names=feature_names,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            observed_df=ig_df,
            top_n=args.permute_top_n,
            n_permutations=args.n_permutations,
        )
        perm_df.to_csv(os.path.join(args.run_dir, cfg.IG_PERMUTATION_FILE), index=False)
        print(f"[INFO] Saved permutation summary for top {args.permute_top_n} held-out features.")

    print("\n[INFO] Held-out IG complete.")
    print(ig_df.head(args.top_n_plot))
    return ig_df


def compute_cv_ig(
    args: argparse.Namespace,
    run_meta: dict,
    df_clean: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    cv_dir = args.cv_dir or os.path.join(args.run_dir, cfg.CV_DIRNAME)
    if not os.path.isdir(cv_dir):
        raise FileNotFoundError(f"CV directory not found: {cv_dir}. Run cv_mlp_aligned.py first.")

    fold_dirs = [
        os.path.join(cv_dir, d)
        for d in sorted(os.listdir(cv_dir))
        if os.path.isdir(os.path.join(cv_dir, d)) and d.startswith("fold_")
    ]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {cv_dir}.")

    fold_rows = []

    for fold_dir in fold_dirs:
        fold_name = os.path.basename(fold_dir)
        fold_num = int(fold_name.split("_")[-1])
        split_df = pd.read_csv(os.path.join(fold_dir, cfg.CV_FOLD_SPLITS_FILE))
        eval_indices = split_df.loc[split_df["split"] == "eval", "row_index"].astype(int).tolist()
        fold_eval_df = df_clean.loc[eval_indices].copy()

        preprocessor = Preprocessor.load(os.path.join(fold_dir, cfg.PREPROCESSOR_FILE))
        x_eval = transform_features(fold_eval_df, preprocessor)
        y_eval = fold_eval_df[cfg.AGE_COLUMN].to_numpy(dtype="float32")
        eval_loader = build_tensor_loader(x_eval, y_eval, batch_size=args.batch_size, shuffle=False)

        model = ProteinExpressionNet(
            input_size=len(preprocessor.protein_cols),
            hidden_sizes=run_meta["hidden_layers"],
            dropout_rate=run_meta["dropout"],
        ).to(device)
        state = torch.load(os.path.join(fold_dir, cfg.MODEL_FILE), map_location=device)
        model.load_state_dict(state)
        model.eval()

        global_importance, per_sample = compute_global_ig(
            model=model,
            loader=eval_loader,
            device=device,
            steps=args.steps,
        )

        fold_ig_df = pd.DataFrame(
            {
                "feature": preprocessor.protein_cols,
                "mean_abs_ig": global_importance,
                "fold": fold_num,
            }
        ).sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
        fold_ig_df.to_csv(os.path.join(fold_dir, cfg.IG_OUTPUT_FILE), index=False)
        np.save(os.path.join(fold_dir, cfg.IG_PER_SAMPLE_FILE), per_sample)
        fold_rows.append(fold_ig_df)
        print(f"[INFO] CV IG complete for {fold_name}")

    fold_long_df = pd.concat(fold_rows, axis=0, ignore_index=True)
    fold_long_df.to_csv(os.path.join(args.run_dir, cfg.IG_CV_FOLD_FILE), index=False)

    summary_df = (
        fold_long_df.groupby("feature")["mean_abs_ig"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_ig", "std": "sd_abs_ig", "count": "n_folds"})
    )
    summary_df["sd_abs_ig"] = summary_df["sd_abs_ig"].fillna(0.0)
    summary_df["se_abs_ig"] = summary_df["sd_abs_ig"] / np.sqrt(summary_df["n_folds"].clip(lower=1))
    summary_df = summary_df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
    summary_df.to_csv(os.path.join(args.run_dir, cfg.IG_CV_OUTPUT_FILE), index=False)

    error_col = "se_abs_ig" if args.cv_error_bar == "se" else "sd_abs_ig"
    plot_top_ig(
        ig_df=summary_df,
        save_path=os.path.join(args.run_dir, cfg.IG_CV_PLOT_FILE),
        top_n=args.top_n_plot,
        title=f"CV-aggregated IG (Top proteins, error bars = {args.cv_error_bar})",
        error_col=error_col,
    )

    print("\n[INFO] CV-aggregated IG complete.")
    print(summary_df.head(args.top_n_plot))
    return summary_df


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    device = resolve_device(args.device)
    run_meta = load_run_metadata(args.run_dir)
    df_clean, _ = load_clean_dataset(
        input_file=args.input_file,
        protein_columns_file=args.protein_columns_file,
    )

    if args.mode in {"heldout", "both"}:
        compute_heldout_ig(args=args, run_meta=run_meta, df_clean=df_clean, device=device)

    if args.mode in {"cv", "both"}:
        compute_cv_ig(args=args, run_meta=run_meta, df_clean=df_clean, device=device)


if __name__ == "__main__":
    main()
