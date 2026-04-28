from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj: dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_table(path: Optional[str], required: bool = False) -> pd.DataFrame:
    if not path:
        if required:
            raise FileNotFoundError("Required input path is missing.")
        return pd.DataFrame()
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"File not found: {path}")
        return pd.DataFrame()
    lower = path.lower()
    sep = "\t" if lower.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def detect_gene_column(df: pd.DataFrame) -> str:
    for c in ["gene", "Gene", "symbol", "Symbol", "Target_Protein", "target", "Target"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find gene column in columns: {list(df.columns)}")


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.empty:
        return s
    vmin = float(s.min())
    vmax = float(s.max())
    if abs(vmax - vmin) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)


def load_gene_list(path: Optional[str]) -> List[str]:
    if not path:
        return []
    genes: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    return list(dict.fromkeys(genes))


# =============================================================================
# Annotation harmonization
# =============================================================================

def coerce_annotation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "gene",
            "is_secreted",
            "is_receptor",
            "is_enzyme_or_kinase",
            "known_drug_target_family",
            "aging_pathway_support",
            "module_support",
            "literature_support",
            "hub_penalty",
            "interpretability_support",
        ])

    gene_col = detect_gene_column(df)
    out = df.copy().rename(columns={gene_col: "gene"})
    out["gene"] = out["gene"].astype(str).str.strip()
    out = out.loc[out["gene"] != ""].copy()

    rename_map = {
        "secreted": "is_secreted",
        "receptor": "is_receptor",
        "enzyme_or_kinase": "is_enzyme_or_kinase",
        "drug_target_family": "known_drug_target_family",
        "pathway_support": "aging_pathway_support",
        "aging_support": "aging_pathway_support",
        "module_consistency": "module_support",
        "literature_score": "literature_support",
        "hub_score": "hub_penalty",
        "interpretability_score": "interpretability_support",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required_cols = [
        "is_secreted",
        "is_receptor",
        "is_enzyme_or_kinase",
        "known_drug_target_family",
        "aging_pathway_support",
        "module_support",
        "literature_support",
        "hub_penalty",
        "interpretability_support",
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0

    binary_like = [
        "is_secreted", "is_receptor", "is_enzyme_or_kinase", "known_drug_target_family"
    ]
    for c in binary_like:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[c] = (out[c] > 0).astype(float)

    numeric_like = [
        "aging_pathway_support", "module_support", "literature_support",
        "hub_penalty", "interpretability_support"
    ]
    for c in numeric_like:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out = out.groupby("gene", as_index=False).mean(numeric_only=True)
    return out


# =============================================================================
# Pair/path support features
# =============================================================================

def build_direct_pair_support(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=["gene", "n_direct_pairs", "max_interaction_weight", "mean_interaction_weight"])

    gene_col = "Target_Protein" if "Target_Protein" in pairs_df.columns else detect_gene_column(pairs_df)
    w_col = "Interaction_Weight" if "Interaction_Weight" in pairs_df.columns else None

    tmp = pairs_df.copy()
    tmp[gene_col] = tmp[gene_col].astype(str).str.strip()
    tmp = tmp.loc[tmp[gene_col] != ""].copy()

    if w_col and w_col in tmp.columns:
        tmp[w_col] = pd.to_numeric(tmp[w_col], errors="coerce").fillna(0.0)
        out = (
            tmp.groupby(gene_col, as_index=False)[w_col]
            .agg(["count", "mean", "max"])
            .reset_index()
            .rename(columns={gene_col: "gene", "count": "n_direct_pairs", "mean": "mean_interaction_weight", "max": "max_interaction_weight"})
        )
        return out

    out = tmp[[gene_col]].drop_duplicates().rename(columns={gene_col: "gene"})
    out["n_direct_pairs"] = 1
    out["mean_interaction_weight"] = 0.0
    out["max_interaction_weight"] = 0.0
    return out


def build_path_support(paths_df: pd.DataFrame) -> pd.DataFrame:
    if paths_df.empty:
        return pd.DataFrame(columns=["gene", "n_paths", "best_path_weight", "min_path_length"])

    gene_col = "Target_Protein" if "Target_Protein" in paths_df.columns else detect_gene_column(paths_df)
    tmp = paths_df.copy()
    tmp[gene_col] = tmp[gene_col].astype(str).str.strip()
    tmp = tmp.loc[tmp[gene_col] != ""].copy()

    if "Path_Weight_Product" in tmp.columns:
        tmp["Path_Weight_Product"] = pd.to_numeric(tmp["Path_Weight_Product"], errors="coerce").fillna(0.0)
    else:
        tmp["Path_Weight_Product"] = 0.0
    if "Path_Length" in tmp.columns:
        tmp["Path_Length"] = pd.to_numeric(tmp["Path_Length"], errors="coerce").fillna(99.0)
    else:
        tmp["Path_Length"] = 99.0

    out = (
        tmp.groupby(gene_col, as_index=False)
        .agg(
            n_paths=(gene_col, "count"),
            best_path_weight=("Path_Weight_Product", "max"),
            min_path_length=("Path_Length", "min"),
        )
        .rename(columns={gene_col: "gene"})
    )
    return out


# =============================================================================
# Discovery scoring
# =============================================================================

def compute_mechanistic_discovery(
    ranked_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    prior_genes: List[str],
    candidate_top_n: int,
    final_top_n: int,
    w_network: float,
    w_local_consistency: float,
    w_module: float,
    w_pathway: float,
    w_interpretability: float,
    w_novelty: float,
    w_hub_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranked_df.empty:
        raise ValueError("Ranked target table is empty.")

    gene_col = detect_gene_column(ranked_df)
    ranked = ranked_df.copy().rename(columns={gene_col: "gene"})
    ranked["gene"] = ranked["gene"].astype(str).str.strip()
    ranked = ranked.loc[ranked["gene"] != ""].copy()

    if "is_diagnostic_seed" in ranked.columns:
        ranked = ranked.loc[ranked["is_diagnostic_seed"] == 0].copy()

    sort_col = "final_score" if "final_score" in ranked.columns else ("score" if "score" in ranked.columns else None)
    if sort_col is None:
        raise ValueError("Ranked target table must contain 'final_score' or 'score'.")

    ranked = ranked.sort_values(sort_col, ascending=False).reset_index(drop=True)
    candidate_df = ranked.head(candidate_top_n).copy()

    ann = coerce_annotation_table(annotation_df)
    pair_support = build_direct_pair_support(pairs_df)
    path_support = build_path_support(paths_df)
    prior_set = set(prior_genes)

    merged = candidate_df.merge(ann, on="gene", how="left")
    merged = merged.merge(pair_support, on="gene", how="left")
    merged = merged.merge(path_support, on="gene", how="left")

    fill_zero_cols = [
        "is_secreted", "is_receptor", "is_enzyme_or_kinase", "known_drug_target_family",
        "aging_pathway_support", "module_support", "literature_support", "hub_penalty",
        "interpretability_support", "n_direct_pairs", "mean_interaction_weight", "max_interaction_weight",
        "n_paths", "best_path_weight", "min_path_length",
    ]
    for c in fill_zero_cols:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    # components
    merged["network_component"] = normalize_series(merged[sort_col])

    merged["local_consistency_raw"] = (
        normalize_series(merged["n_direct_pairs"]) * 0.35 +
        normalize_series(merged["max_interaction_weight"]) * 0.35 +
        normalize_series(merged["best_path_weight"]) * 0.20 +
        (1 - normalize_series(merged["min_path_length"])) * 0.10
    )
    merged["local_consistency_component"] = normalize_series(merged["local_consistency_raw"])

    merged["module_component"] = normalize_series(merged["module_support"])
    merged["pathway_component"] = normalize_series(merged["aging_pathway_support"] + 0.5 * merged["literature_support"])

    merged["interpretability_component"] = normalize_series(
        merged["interpretability_support"] +
        0.5 * merged["n_direct_pairs"] +
        0.5 * merged["best_path_weight"]
    )

    merged["prior_overlap"] = merged["gene"].isin(prior_set).astype(float)
    merged["novelty_component"] = 1.0 - merged["prior_overlap"]
    merged["hub_penalty_component"] = normalize_series(merged["hub_penalty"])

    merged["mechanistic_discovery_score"] = (
        w_network * merged["network_component"] +
        w_local_consistency * merged["local_consistency_component"] +
        w_module * merged["module_component"] +
        w_pathway * merged["pathway_component"] +
        w_interpretability * merged["interpretability_component"] +
        w_novelty * merged["novelty_component"] -
        w_hub_penalty * merged["hub_penalty_component"]
    )

    merged = merged.sort_values(
        ["mechanistic_discovery_score", "local_consistency_component", "network_component"],
        ascending=False,
    ).reset_index(drop=True)
    merged["discovery_rank"] = np.arange(1, len(merged) + 1)

    shortlist = merged.head(final_top_n).copy()
    return merged, shortlist


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a 'new, mechanistically coherent intervention-candidate' shortlist. "
            "This is discovery-oriented and explicitly rewards local mechanistic consistency and novelty."
        )
    )
    p.add_argument("--ranked_targets", required=True, help="CSV/TSV from target_gnn_noprior.py, e.g. final_target_scores.csv")
    p.add_argument("--direct_pairs", default=None, help="CSV/TSV from target_gnn_noprior.py, e.g. diagnostic_target_pairs.csv")
    p.add_argument("--path_table", default=None, help="Optional path export table, e.g. diagnostic_target_paths.csv")
    p.add_argument("--annotation_table", default=None, help="Optional annotation table")
    p.add_argument("--prior_file", default=None, help="Optional prior target list used only to score novelty (not to promote known targets)")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--candidate_top_n", type=int, default=50)
    p.add_argument("--final_top_n", type=int, default=10)

    p.add_argument("--w_network", type=float, default=0.30)
    p.add_argument("--w_local_consistency", type=float, default=0.25)
    p.add_argument("--w_module", type=float, default=0.15)
    p.add_argument("--w_pathway", type=float, default=0.10)
    p.add_argument("--w_interpretability", type=float, default=0.10)
    p.add_argument("--w_novelty", type=float, default=0.15)
    p.add_argument("--w_hub_penalty", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    ranked_df = load_table(args.ranked_targets, required=True)
    pairs_df = load_table(args.direct_pairs, required=False)
    paths_df = load_table(args.path_table, required=False)
    annotation_df = load_table(args.annotation_table, required=False)
    prior_genes = load_gene_list(args.prior_file)

    full_scored, shortlist = compute_mechanistic_discovery(
        ranked_df=ranked_df,
        annotation_df=annotation_df,
        pairs_df=pairs_df,
        paths_df=paths_df,
        prior_genes=prior_genes,
        candidate_top_n=args.candidate_top_n,
        final_top_n=args.final_top_n,
        w_network=args.w_network,
        w_local_consistency=args.w_local_consistency,
        w_module=args.w_module,
        w_pathway=args.w_pathway,
        w_interpretability=args.w_interpretability,
        w_novelty=args.w_novelty,
        w_hub_penalty=args.w_hub_penalty,
    )

    full_scored.to_csv(os.path.join(out_dir, "mechanistic_candidate_pool_scored.csv"), index=False)
    shortlist.to_csv(os.path.join(out_dir, "final_mechanistic_intervention_candidates.csv"), index=False)

    summary = {
        "candidate_top_n": args.candidate_top_n,
        "final_top_n": args.final_top_n,
        "weights": {
            "network": args.w_network,
            "local_consistency": args.w_local_consistency,
            "module": args.w_module,
            "pathway": args.w_pathway,
            "interpretability": args.w_interpretability,
            "novelty": args.w_novelty,
            "hub_penalty": args.w_hub_penalty,
        },
        "n_prior_supplied": len(prior_genes),
        "n_candidates_scored": int(len(full_scored)),
        "n_shortlist": int(len(shortlist)),
        "shortlist_preview": shortlist[[
            c for c in [
                "gene", "mechanistic_discovery_score", "network_component", "local_consistency_component",
                "module_component", "pathway_component", "interpretability_component", "novelty_component",
                "hub_penalty_component", "prior_overlap"
            ] if c in shortlist.columns
        ]].to_dict(orient="records"),
    }
    save_json(summary, os.path.join(out_dir, "mechanistic_discovery_summary.json"))

    print("\n[INFO] Mechanistic candidate discovery complete.")
    print(f"[INFO] Candidate pool scored: {len(full_scored)}")
    print(f"[INFO] Final shortlist size: {len(shortlist)}")
    print("\n[INFO] Final mechanistically coherent intervention candidates:")
    show_cols = [
        c for c in [
            "gene", "mechanistic_discovery_score", "network_component", "local_consistency_component",
            "module_component", "pathway_component", "interpretability_component", "novelty_component",
            "hub_penalty_component", "prior_overlap"
        ] if c in shortlist.columns
    ]
    print(shortlist[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()

