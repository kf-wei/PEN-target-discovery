from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# =============================================================================
# Reproducibility and utils
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def save_json(obj: dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Input readers
# =============================================================================

def read_diagnostic_proteins(path: str) -> List[str]:
    """
    Accepts:
      1) txt with one symbol per line
      2) txt/csv lines like SYMBOL;Description,score
      3) csv/tsv with one of columns:
         protein, Protein, gene, Gene, symbol, Symbol, Diagnostic_Protein
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Diagnostic protein file not found: {path}")

    lower = path.lower()
    if lower.endswith((".csv", ".tsv")):
        try:
            sep = "\t" if lower.endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep)
            for col in ["protein", "Protein", "gene", "Gene", "symbol", "Symbol", "Diagnostic_Protein"]:
                if col in df.columns:
                    vals = df[col].astype(str).str.strip()
                    vals = vals[vals.ne("")].tolist()
                    if vals:
                        return list(dict.fromkeys(vals))
        except Exception:
            pass

    proteins: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                proteins.append(line.split(";", 1)[0].replace('"', '').strip())
            else:
                proteins.append(line.split(",")[0].replace('"', '').strip())
    proteins = [p for p in proteins if p]
    if not proteins:
        raise ValueError("No diagnostic proteins were parsed from the input file.")
    return list(dict.fromkeys(proteins))


def read_ppi_edges(
    ppi_file: str,
    source_col: str,
    target_col: str,
    weight_col: str,
    min_weight: float,
) -> pd.DataFrame:
    if not os.path.exists(ppi_file):
        raise FileNotFoundError(f"PPI file not found: {ppi_file}")

    lower = ppi_file.lower()
    sep = "\t" if lower.endswith(".tsv") else ","
    df = pd.read_csv(ppi_file, sep=sep)

    missing = [c for c in [source_col, target_col, weight_col] if c not in df.columns]
    if missing:
        raise ValueError(f"PPI file missing required columns: {missing}")

    out = df[[source_col, target_col, weight_col]].copy()
    out.columns = ["source", "target", "weight"]
    out["source"] = out["source"].astype(str).str.strip()
    out["target"] = out["target"].astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["source", "target", "weight"])
    out = out.loc[(out["source"] != "") & (out["target"] != "")]
    out = out.loc[out["source"] != out["target"]]
    out = out.loc[out["weight"] >= min_weight].copy()
    if out.empty:
        raise ValueError("No PPI edges remain after filtering.")
    return out.reset_index(drop=True)


def merge_edge_tables(edge_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(edge_tables, axis=0, ignore_index=True)
    merged["pair_key"] = merged.apply(
        lambda r: tuple(sorted((str(r["source"]), str(r["target"])))), axis=1
    )
    merged = (
        merged.groupby("pair_key", as_index=False)
        .agg({"source": "first", "target": "first", "weight": "max"})
        .drop(columns=["pair_key"], errors="ignore")
    )
    return merged.reset_index(drop=True)


def load_feature_table(path: Optional[str], gene_col: str = "gene") -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature table not found: {path}")

    lower = path.lower()
    sep = "\t" if lower.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)

    if gene_col not in df.columns:
        for alt in ["gene", "Gene", "symbol", "Symbol", "protein", "Protein"]:
            if alt in df.columns:
                gene_col = alt
                break
    if gene_col not in df.columns:
        raise ValueError(f"Feature table {path} missing gene column.")

    df = df.copy()
    df[gene_col] = df[gene_col].astype(str).str.strip()
    df = df.loc[df[gene_col] != ""].copy()
    numeric_cols = [c for c in df.columns if c != gene_col]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.groupby(gene_col, as_index=False)[numeric_cols].mean()
    df = df.rename(columns={gene_col: "gene"})
    return df


# =============================================================================
# Graph construction
# =============================================================================

def build_local_subgraph(
    ppi_edges: pd.DataFrame,
    diagnostic_genes: Sequence[str],
    max_hops: int = 2,
    keep_largest_component: bool = True,
) -> nx.Graph:
    G_full = nx.Graph()
    for row in ppi_edges.itertuples(index=False):
        G_full.add_edge(row.source, row.target, weight=float(row.weight))

    diagnostic_genes = [g for g in diagnostic_genes if g in G_full]
    if not diagnostic_genes:
        raise ValueError("None of the diagnostic proteins were found in the PPI graph.")

    nodes_to_keep = set(diagnostic_genes)
    frontier = set(diagnostic_genes)
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G_full.neighbors(node))
        nodes_to_keep.update(next_frontier)
        frontier = next_frontier

    G = G_full.subgraph(sorted(nodes_to_keep)).copy()
    if G.number_of_nodes() == 0:
        raise ValueError("Local subgraph is empty after neighborhood expansion.")

    if keep_largest_component and G.number_of_nodes() > 0:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(sorted(largest)).copy()

    return G


# =============================================================================
# Features
# =============================================================================

def nx_feature_df(G: nx.Graph, diagnostic_genes: Sequence[str]) -> pd.DataFrame:
    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")

    # shortest distance to any diagnostic seed
    seed_in_graph = [g for g in diagnostic_genes if g in G]
    min_dist: Dict[str, int] = {g: 999 for g in G.nodes()}
    for s in seed_in_graph:
        sp = nx.single_source_shortest_path_length(G, s, cutoff=None)
        for node, dist in sp.items():
            if dist < min_dist[node]:
                min_dist[node] = dist

    rows = []
    for g in G.nodes():
        rows.append(
            {
                "gene": g,
                "degree_centrality": float(degree.get(g, 0.0)),
                "betweenness_centrality": float(betweenness.get(g, 0.0)),
                "closeness_centrality": float(closeness.get(g, 0.0)),
                "pagerank": float(pagerank.get(g, 0.0)),
                "min_seed_distance": float(min_dist.get(g, 999)),
                "seed_neighbor": 1.0 if min_dist.get(g, 999) == 1 else 0.0,
                "seed_second_neighbor": 1.0 if min_dist.get(g, 999) == 2 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_node_feature_matrix(
    G: nx.Graph,
    diagnostic_genes: Sequence[str],
    embedding_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> tuple[np.ndarray, List[str], pd.DataFrame]:
    nodes = list(G.nodes())

    merged = pd.DataFrame({"gene": nodes})

    if not embedding_df.empty:
        emb = embedding_df.copy().set_index("gene").reindex(nodes)
        emb.columns = [str(c) if str(c).startswith("emb_") else f"emb_{c}" for c in emb.columns]
        merged = merged.merge(emb.reset_index(), on="gene", how="left")

    if not pathway_df.empty:
        pw = pathway_df.copy().set_index("gene").reindex(nodes)
        pw.columns = [str(c) if str(c).startswith("path_") else f"path_{c}" for c in pw.columns]
        merged = merged.merge(pw.reset_index(), on="gene", how="left")

    if not clinical_df.empty:
        cl = clinical_df.copy().set_index("gene").reindex(nodes)
        cl.columns = [str(c) if str(c).startswith("clin_") else f"clin_{c}" for c in cl.columns]
        merged = merged.merge(cl.reset_index(), on="gene", how="left")

    net_df = nx_feature_df(G, diagnostic_genes)
    merged = merged.merge(net_df, on="gene", how="left")

    numeric_cols = [c for c in merged.columns if c != "gene"]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        fill_val = float(merged[c].median()) if merged[c].notna().any() else 0.0
        merged[c] = merged[c].fillna(fill_val)

    X = merged[numeric_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, nodes, merged


# =============================================================================
# Labels for no-prior semi-supervised training
# =============================================================================

def select_negative_nodes(
    G: nx.Graph,
    nodes: Sequence[str],
    diagnostic_genes: Sequence[str],
    n_neg_multiplier: float = 2.0,
    min_negative_distance: int = 3,
    seed: int = 42,
) -> List[str]:
    rng = np.random.default_rng(seed)
    diag = [g for g in diagnostic_genes if g in G]
    if not diag:
        raise ValueError("No diagnostic genes are present in the local graph.")

    min_dist: Dict[str, int] = {g: 999 for g in nodes}
    for s in diag:
        sp = nx.single_source_shortest_path_length(G, s, cutoff=None)
        for node, dist in sp.items():
            if dist < min_dist[node]:
                min_dist[node] = dist

    neg_candidates = [g for g in nodes if g not in set(diag) and min_dist.get(g, 999) >= min_negative_distance]
    if not neg_candidates:
        neg_candidates = [g for g in nodes if g not in set(diag)]

    n_pos = len(diag)
    n_neg = min(len(neg_candidates), max(int(np.ceil(n_pos * n_neg_multiplier)), n_pos))
    if n_neg == 0:
        return []
    chosen = rng.choice(neg_candidates, size=n_neg, replace=False)
    return sorted(chosen.tolist())


def build_pyg_data(
    G: nx.Graph,
    X: np.ndarray,
    nodes: Sequence[str],
    diagnostic_genes: Sequence[str],
    negative_genes: Optional[Sequence[str]] = None,
    n_neg_multiplier: float = 2.0,
    min_negative_distance: int = 3,
    seed: int = 42,
) -> tuple[Data, Dict[str, int], List[str], List[str]]:
    node_index = {g: i for i, g in enumerate(nodes)}

    edges: List[List[int]] = []
    edge_weight: List[float] = []
    for u, v, attr in G.edges(data=True):
        w = float(attr.get("weight", 1.0))
        ui, vi = node_index[u], node_index[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
        edge_weight.extend([w, w])

    if not edges:
        for i in range(len(nodes)):
            edges.append([i, i])
            edge_weight.append(1.0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weight, dtype=torch.float32)
    x = torch.tensor(X, dtype=torch.float32)

    pos_nodes = [g for g in diagnostic_genes if g in node_index]
    if not pos_nodes:
        raise ValueError("No diagnostic genes are present in the node index.")

    if negative_genes is None:
        neg_nodes = select_negative_nodes(
            G=G,
            nodes=nodes,
            diagnostic_genes=pos_nodes,
            n_neg_multiplier=n_neg_multiplier,
            min_negative_distance=min_negative_distance,
            seed=seed,
        )
    else:
        neg_nodes = [g for g in negative_genes if g in node_index and g not in set(pos_nodes)]

    y = torch.full((len(nodes),), -1.0, dtype=torch.float32)
    train_mask = torch.zeros(len(nodes), dtype=torch.bool)

    for g in pos_nodes:
        idx = node_index[g]
        y[idx] = 1.0
        train_mask[idx] = True

    for g in neg_nodes:
        idx = node_index[g]
        y[idx] = 0.0
        train_mask[idx] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask)
    return data, node_index, pos_nodes, neg_nodes


# =============================================================================
# GCN model
# =============================================================================

class TwoLayerGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)
        return x.squeeze(-1)


def train_gcn(
    data: Data,
    input_dim: int,
    device: torch.device,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    patience: int = 30,
) -> tuple[TwoLayerGCN, pd.DataFrame]:
    model = TwoLayerGCN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_loss = float("inf")
    wait = 0
    history: List[dict] = []

    labeled_mask = data.train_mask
    if int(labeled_mask.sum()) == 0:
        raise ValueError("No labeled nodes available for GCN training.")

    target = data.y[labeled_mask]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[labeled_mask], target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits[labeled_mask])
            pred_bin = (probs >= 0.5).float()
            acc = float((pred_bin == target).float().mean().item())

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "labeled_acc": acc,
                "n_labeled": int(labeled_mask.sum().item()),
            }
        )

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# =============================================================================
# Diffusion baseline and final scoring
# =============================================================================

def personalized_pagerank_scores(
    G: nx.Graph,
    diagnostic_genes: Sequence[str],
    alpha: float = 0.85,
) -> Dict[str, float]:
    seeds = [g for g in diagnostic_genes if g in G]
    if not seeds:
        return {g: 0.0 for g in G.nodes()}

    personalization = {g: 0.0 for g in G.nodes()}
    seed_weight = 1.0 / len(seeds)
    for g in seeds:
        personalization[g] = seed_weight

    return nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight")


def compute_final_scores(
    G: nx.Graph,
    nodes: Sequence[str],
    gnn_probs: np.ndarray,
    diagnostic_genes: Sequence[str],
    w_diffusion: float = 0.30,
    w_centrality: float = 0.05,
) -> pd.DataFrame:
    diffusion = personalized_pagerank_scores(G, diagnostic_genes)
    degree = nx.degree_centrality(G)

    diff_vals = np.array([float(diffusion.get(g, 0.0)) for g in nodes], dtype=float)
    cent_vals = np.array([float(degree.get(g, 0.0)) for g in nodes], dtype=float)

    # normalize helper
    def _normalize(x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    diff_norm = _normalize(diff_vals)
    cent_norm = _normalize(cent_vals)
    gnn_norm = _normalize(gnn_probs.astype(float))

    final_score = (1.0 - w_diffusion - w_centrality) * gnn_norm + w_diffusion * diff_norm + w_centrality * cent_norm

    df = pd.DataFrame(
        {
            "gene": list(nodes),
            "gnn_score": gnn_probs.astype(float),
            "gnn_score_norm": gnn_norm,
            "diffusion_score": diff_vals,
            "diffusion_score_norm": diff_norm,
            "centrality": cent_vals,
            "centrality_norm": cent_norm,
            "is_diagnostic_seed": [1 if g in set(diagnostic_genes) else 0 for g in nodes],
            "final_score": final_score,
        }
    )
    df = df.sort_values(["final_score", "gnn_score", "diffusion_score"], ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# =============================================================================
# Direct pairs and artifacts
# =============================================================================

def extract_direct_pairs_only(
    G: nx.Graph,
    diagnostic_genes: Sequence[str],
    ranked_targets_df: pd.DataFrame,
    n_top_targets_to_scan: int = 100,
) -> pd.DataFrame:
    ranked_targets = ranked_targets_df.loc[ranked_targets_df["is_diagnostic_seed"] == 0, "gene"].tolist()[:n_top_targets_to_scan]
    assigned_targets: set[str] = set()
    rows: List[dict] = []

    for d in diagnostic_genes:
        if d not in G:
            continue
        for t in ranked_targets:
            if t == d or t in assigned_targets or t not in G:
                continue
            if G.has_edge(d, t):
                w = float(G[d][t].get("weight", 0.0))
                rows.append(
                    {
                        "Diagnostic_Protein": d,
                        "Target_Protein": t,
                        "Interaction_Weight": w,
                        "Interaction_Level": 1,
                        "Regulation": "Up" if w >= 0.5 else "Down",
                    }
                )
                assigned_targets.add(t)
                break

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["Diagnostic_Protein", "Target_Protein", "Interaction_Weight", "Interaction_Level", "Regulation"])
    return out


# =============================================================================
# Main pipeline
# =============================================================================

@dataclass
class RunArtifacts:
    graph: nx.Graph
    feature_matrix: np.ndarray
    nodes: List[str]
    feature_table: pd.DataFrame
    ranked_scores: pd.DataFrame
    direct_pairs: pd.DataFrame
    training_history: pd.DataFrame
    positive_nodes: List[str]
    negative_nodes: List[str]


def run_pipeline(args: argparse.Namespace) -> RunArtifacts:
    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)

    diagnostic_genes = read_diagnostic_proteins(args.diagnostic_file)

    edge_tables = [
        read_ppi_edges(
            ppi_file=args.ppi_file,
            source_col=args.ppi_source_col,
            target_col=args.ppi_target_col,
            weight_col=args.ppi_weight_col,
            min_weight=args.min_edge_weight,
        )
    ]
    if args.omnipath_file:
        edge_tables.append(
            read_ppi_edges(
                ppi_file=args.omnipath_file,
                source_col=args.omnipath_source_col,
                target_col=args.omnipath_target_col,
                weight_col=args.omnipath_weight_col,
                min_weight=args.min_edge_weight,
            )
        )

    merged_edges = merge_edge_tables(edge_tables)
    merged_edges.to_csv(os.path.join(out_dir, "merged_ppi_edges.csv"), index=False)

    G = build_local_subgraph(
        ppi_edges=merged_edges,
        diagnostic_genes=diagnostic_genes,
        max_hops=args.max_hops,
        keep_largest_component=not args.keep_all_components,
    )

    embedding_df = load_feature_table(args.embedding_file, gene_col=args.embedding_gene_col)
    pathway_df = load_feature_table(args.pathway_file, gene_col=args.pathway_gene_col)
    clinical_df = load_feature_table(args.clinical_file, gene_col=args.clinical_gene_col)

    X, nodes, feature_table = build_node_feature_matrix(
        G=G,
        diagnostic_genes=diagnostic_genes,
        embedding_df=embedding_df,
        pathway_df=pathway_df,
        clinical_df=clinical_df,
    )
    feature_table.to_csv(os.path.join(out_dir, "graph_node_features.csv"), index=False)

    data, node_index, pos_nodes, neg_nodes = build_pyg_data(
        G=G,
        X=X,
        nodes=nodes,
        diagnostic_genes=diagnostic_genes,
        negative_genes=None,
        n_neg_multiplier=args.n_neg_multiplier,
        min_negative_distance=args.min_negative_distance,
        seed=args.seed,
    )

    model, history_df = train_gcn(
        data=data,
        input_dim=X.shape[1],
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )
    history_df.to_csv(os.path.join(out_dir, "gcn_training_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "target_gnn_noprior_model.pth"))

    model.eval()
    with torch.no_grad():
        logits = model(data.to(device)).detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

    ranked_df = compute_final_scores(
        G=G,
        nodes=nodes,
        gnn_probs=probs,
        diagnostic_genes=diagnostic_genes,
        w_diffusion=args.w_diffusion,
        w_centrality=args.w_centrality,
    )
    ranked_df.to_csv(os.path.join(out_dir, "final_target_scores.csv"), index=False)

    direct_pairs_df = extract_direct_pairs_only(
        G=G,
        diagnostic_genes=diagnostic_genes,
        ranked_targets_df=ranked_df,
        n_top_targets_to_scan=args.top_targets_to_scan,
    )
    direct_pairs_df.to_csv(os.path.join(out_dir, "diagnostic_target_pairs.csv"), index=False)

    pd.DataFrame({"positive_seed_nodes": pos_nodes}).to_csv(os.path.join(out_dir, "positive_seed_nodes.csv"), index=False)
    pd.DataFrame({"negative_training_nodes": neg_nodes}).to_csv(os.path.join(out_dir, "negative_training_nodes.csv"), index=False)
    pd.DataFrame({"gene": nodes}).to_csv(os.path.join(out_dir, "graph_nodes.csv"), index=False)
    pd.DataFrame(
        [{"source": u, "target": v, "weight": float(a.get("weight", 1.0))} for u, v, a in G.edges(data=True)]
    ).to_csv(os.path.join(out_dir, "graph_edges_local.csv"), index=False)

    summary = {
        "n_diagnostic_input": len(diagnostic_genes),
        "n_diagnostic_in_graph": len(pos_nodes),
        "n_negative_training_nodes": len(neg_nodes),
        "n_graph_nodes": int(G.number_of_nodes()),
        "n_graph_edges": int(G.number_of_edges()),
        "n_direct_pairs": int(len(direct_pairs_df)),
        "device": str(device),
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_hops": args.max_hops,
        "min_edge_weight": args.min_edge_weight,
        "n_neg_multiplier": args.n_neg_multiplier,
        "min_negative_distance": args.min_negative_distance,
        "w_diffusion": args.w_diffusion,
        "w_centrality": args.w_centrality,
        "top_ranked_preview": ranked_df.head(20).to_dict(orient="records"),
    }
    save_json(summary, os.path.join(out_dir, "run_summary.json"))
    save_json(vars(args), os.path.join(out_dir, "config_snapshot.json"))

    return RunArtifacts(
        graph=G,
        feature_matrix=X,
        nodes=nodes,
        feature_table=feature_table,
        ranked_scores=ranked_df,
        direct_pairs=direct_pairs_df,
        training_history=history_df,
        positive_nodes=pos_nodes,
        negative_nodes=neg_nodes,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "No-prior GCN target prioritization for PEN. "
            "Uses diagnostic proteins as seeds, real PPI edges, optional real node features, "
            "and negative nodes selected away from the seed neighborhood."
        )
    )

    p.add_argument("--diagnostic_file", required=True)
    p.add_argument("--ppi_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--omnipath_file", default=None)

    p.add_argument("--ppi_source_col", default="source")
    p.add_argument("--ppi_target_col", default="target")
    p.add_argument("--ppi_weight_col", default="weight")
    p.add_argument("--omnipath_source_col", default="source")
    p.add_argument("--omnipath_target_col", default="target")
    p.add_argument("--omnipath_weight_col", default="weight")

    p.add_argument("--min_edge_weight", type=float, default=0.7)
    p.add_argument("--max_hops", type=int, default=2)
    p.add_argument("--keep_all_components", action="store_true")

    p.add_argument("--embedding_file", default=None)
    p.add_argument("--embedding_gene_col", default="gene")
    p.add_argument("--pathway_file", default=None)
    p.add_argument("--pathway_gene_col", default="gene")
    p.add_argument("--clinical_file", default=None)
    p.add_argument("--clinical_gene_col", default="gene")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)

    p.add_argument("--n_neg_multiplier", type=float, default=2.0)
    p.add_argument("--min_negative_distance", type=int, default=3)

    p.add_argument("--w_diffusion", type=float, default=0.30)
    p.add_argument("--w_centrality", type=float, default=0.05)
    p.add_argument("--top_targets_to_scan", type=int, default=100)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_pipeline(args)

    print("\n[INFO] No-prior PEN target GNN run complete.")
    print(f"[INFO] Graph nodes: {artifacts.graph.number_of_nodes()}")
    print(f"[INFO] Graph edges: {artifacts.graph.number_of_edges()}")
    print(f"[INFO] Positive seed nodes used: {len(artifacts.positive_nodes)}")
    print(f"[INFO] Negative training nodes used: {len(artifacts.negative_nodes)}")

    print("\n[INFO] Top ranked targets:")
    print(
        artifacts.ranked_scores.loc[artifacts.ranked_scores["is_diagnostic_seed"] == 0]
        .head(10)
        [["gene", "gnn_score", "diffusion_score", "centrality", "final_score"]]
        .to_string(index=False)
    )

    print("\n[INFO] Direct diagnostic-target pairs (Supplementary Table S1 style):")
    if artifacts.direct_pairs.empty:
        print("No level-1 direct pairs found under the current graph and ranking settings.")
    else:
        print(artifacts.direct_pairs.to_string(index=False))


if __name__ == "__main__":
    main()

