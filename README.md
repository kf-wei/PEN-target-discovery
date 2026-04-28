# PEN: Protein Expression Network for Target Discovery

## Overview

This repository implements the PEN framework for identifying therapeutic targets from proteomic data.

## Pipeline

1. MLP age prediction
2. Integrated Gradients biomarker discovery
3. GNN-based target prioritization
4. Mechanistic candidate discovery

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Train MLP

```bash
python scripts/01_train_mlp.py \
  --input_file data/example/example_protein.csv \
  --output_dir results/mlp_demo
```

### Step 2: IG analysis

```bash
python scripts/02_ig_analysis.py \
  --run_dir results/mlp_demo
```

### Step 3: Target discovery

```bash
python scripts/03_target_gnn.py \
  --diagnostic_file data/example/diagnostic_proteins.txt \
  --ppi_file data/example/example_ppi.csv \
  --output_dir results/gnn_demo
```

### Step 4: Mechanistic discovery

```bash
python scripts/04_mechanistic_discovery.py \
  --ranked_targets results/gnn_demo/final_target_scores.csv \
  --direct_pairs results/gnn_demo/diagnostic_target_pairs.csv \
  --output_dir results/final_demo
```

## Notes

Large datasets and trained models are not included.

