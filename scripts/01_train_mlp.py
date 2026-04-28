from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
import torch

import config_mlp_aligned as cfg
from mlp_common_aligned import (
    ProteinExpressionNet,
    build_tensor_loader,
    clean_dataset,
    compute_metrics,
    create_train_val_test_splits,
    fit_preprocessor,
    load_protein_columns,
    predict,
    read_input_table,
    resolve_device,
    save_history,
    save_json,
    save_splits,
    set_seed,
    train_model,
    transform_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned MLP age-training script for PEN manuscript methods."
    )
    parser.add_argument("--input_file", type=str, default=cfg.INPUT_PROTEIN_FILE)
    parser.add_argument("--protein_columns_file", type=str, default=cfg.PROTEIN_COLUMNS_FILE)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=cfg.DEFAULT_DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=cfg.DEFAULT_PATIENCE)
    parser.add_argument("--learning_rate", type=float, default=cfg.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default=",".join(map(str, cfg.DEFAULT_HIDDEN_LAYERS)),
        help="Comma-separated hidden layer sizes, default manuscript-aligned: 128,64,32",
    )
    parser.add_argument("--dropout", type=float, default=cfg.DEFAULT_DROPOUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(cfg.RANDOM_SEED)

    hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip()]
    run_dir = args.output_dir or f"{cfg.DEFAULT_OUTPUT_DIR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg.ensure_output_dir(run_dir)

    print(f"[INFO] Loading table: {args.input_file}")
    df = read_input_table(args.input_file)
    protein_cols = load_protein_columns(
        df=df,
        demographic_cols=cfg.DEMOGRAPHIC_COLS,
        protein_columns_file=args.protein_columns_file,
    )
    print(f"[INFO] Candidate protein columns: {len(protein_cols)}")

    df_clean, protein_cols = clean_dataset(
        df=df,
        protein_cols=protein_cols,
        age_column=cfg.AGE_COLUMN,
        sex_column=cfg.SEX_COLUMN,
        sample_missing_threshold=cfg.SAMPLE_MISSING_THRESHOLD,
        feature_missing_threshold=cfg.FEATURE_MISSING_THRESHOLD,
    )
    print(f"[INFO] Samples after QC filters: {len(df_clean)}")
    print(f"[INFO] Proteins after QC filters: {len(protein_cols)}")

    train_idx, val_idx, test_idx, used_bins = create_train_val_test_splits(
        df=df_clean,
        age_column=cfg.AGE_COLUMN,
        sex_column=cfg.SEX_COLUMN,
        test_size=cfg.TEST_SIZE,
        val_size_within_dev=cfg.VAL_SIZE_WITHIN_DEV,
        random_seed=cfg.RANDOM_SEED,
        max_age_bins=cfg.AGE_STRATIFY_BINS_MAX,
    )
    save_splits(train_idx, val_idx, test_idx, os.path.join(run_dir, cfg.SPLITS_FILE))

    train_df = df_clean.loc[train_idx].copy()
    val_df = df_clean.loc[val_idx].copy()
    test_df = df_clean.loc[test_idx].copy()

    preprocessor = fit_preprocessor(
        train_df=train_df,
        protein_cols=protein_cols,
        winsor_lower_quantile=cfg.WINSOR_LOWER_QUANTILE,
        winsor_upper_quantile=cfg.WINSOR_UPPER_QUANTILE,
    )
    preprocessor.save(os.path.join(run_dir, cfg.PREPROCESSOR_FILE))

    with open(os.path.join(run_dir, cfg.FEATURE_NAMES_FILE), "w", encoding="utf-8") as f:
        for col in protein_cols:
            f.write(f"{col}\n")

    x_train = transform_features(train_df, preprocessor)
    x_val = transform_features(val_df, preprocessor)
    x_test = transform_features(test_df, preprocessor)
    y_train = train_df[cfg.AGE_COLUMN].to_numpy(dtype="float32")
    y_val = val_df[cfg.AGE_COLUMN].to_numpy(dtype="float32")
    y_test = test_df[cfg.AGE_COLUMN].to_numpy(dtype="float32")

    train_loader = build_tensor_loader(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = build_tensor_loader(x_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_loader = build_tensor_loader(x_test, y_test, batch_size=args.batch_size, shuffle=False)

    model = ProteinExpressionNet(
        input_size=len(protein_cols),
        hidden_sizes=hidden_layers,
        dropout_rate=args.dropout,
    ).to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    torch.save(model.state_dict(), os.path.join(run_dir, cfg.MODEL_FILE))
    save_history(history, os.path.join(run_dir, cfg.TRAIN_HISTORY_FILE))

    preds, truth = predict(model, test_loader, device)
    metrics = compute_metrics(preds, truth)
    save_json(metrics, os.path.join(run_dir, cfg.TRAIN_METRICS_FILE))

    pred_df = pd.DataFrame(
        {
            "row_index": test_df.index.to_numpy(),
            "true_age": truth,
            "predicted_age": preds,
            "residual": preds - truth,
        }
    )
    pred_df.to_csv(os.path.join(run_dir, cfg.TEST_PREDICTIONS_FILE), index=False)

    save_json(
        {
            "input_file": args.input_file,
            "protein_columns_file": args.protein_columns_file,
            "n_samples_clean": int(len(df_clean)),
            "n_proteins": int(len(protein_cols)),
            "train_n": int(len(train_df)),
            "val_n": int(len(val_df)),
            "test_n": int(len(test_df)),
            "age_sex_stratify_bins_used": int(used_bins),
            "device": str(device),
            "hidden_layers": hidden_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "sample_missing_threshold": cfg.SAMPLE_MISSING_THRESHOLD,
            "feature_missing_threshold": cfg.FEATURE_MISSING_THRESHOLD,
            "winsor_lower_quantile": cfg.WINSOR_LOWER_QUANTILE,
            "winsor_upper_quantile": cfg.WINSOR_UPPER_QUANTILE,
        },
        os.path.join(run_dir, cfg.RUN_METADATA_FILE),
    )

    save_json(
        {
            "AGE_COLUMN": cfg.AGE_COLUMN,
            "SEX_COLUMN": cfg.SEX_COLUMN,
            "DEMOGRAPHIC_COLS": cfg.DEMOGRAPHIC_COLS,
            "TEST_SIZE": cfg.TEST_SIZE,
            "VAL_SIZE_WITHIN_DEV": cfg.VAL_SIZE_WITHIN_DEV,
            "SAMPLE_MISSING_THRESHOLD": cfg.SAMPLE_MISSING_THRESHOLD,
            "FEATURE_MISSING_THRESHOLD": cfg.FEATURE_MISSING_THRESHOLD,
            "WINSOR_LOWER_QUANTILE": cfg.WINSOR_LOWER_QUANTILE,
            "WINSOR_UPPER_QUANTILE": cfg.WINSOR_UPPER_QUANTILE,
            "RANDOM_SEED": cfg.RANDOM_SEED,
        },
        os.path.join(run_dir, cfg.CONFIG_SNAPSHOT_FILE),
    )

    print("\n[INFO] Run complete.")
    print(f"[INFO] Output directory: {run_dir}")
    print(f"[INFO] Test metrics: {metrics}")


if __name__ == "__main__":
    main()
