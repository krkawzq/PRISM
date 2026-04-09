#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from scipy import sparse

try:
    from hpdex import parallel_differential_expression
    from prism.io import read_gene_list
except ImportError as exc:
    raise ImportError(
        "scripts/analysis/calc_degs.py requires both `hpdex` and the installed "
        "`prism` package in the active environment. Install dependencies instead "
        "of relying on a hard-coded local path."
    ) from exc

console = Console()
install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute differential expression with hpdex."
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("--groupby-key", required=True, type=str)
    parser.add_argument("--reference", required=True, type=str)
    parser.add_argument(
        "--groups",
        action="append",
        default=[],
        help="Optional target groups to compare against reference. Repeatable.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional text/JSON gene list restricting features.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Optional AnnData layer to use instead of X.",
    )
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--clip-value", type=float, default=20.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-group DEG gene lists and summary JSON.",
    )
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--log2fc-threshold", type=float, default=0.25)
    parser.add_argument(
        "--plot-fg-format",
        action="store_true",
        help="Export plot-fg compatible columns: gene, label, log2_fold_change, fdr.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k genes per group when exporting filtered lists.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show hpdex internal progress bar.",
    )
    return parser.parse_args()


def rename_deg_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"target": "perturbation", "feature": "gene"}).copy()
    return renamed


def to_plot_fg_format(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"target": "label", "feature": "gene"}).copy()
    return pd.DataFrame(renamed[["gene", "label", "log2_fold_change", "fdr"]].copy())


def subset_genes(adata: ad.AnnData, gene_list_path: Path | None) -> ad.AnnData:
    if gene_list_path is None:
        return adata
    requested = read_gene_list(gene_list_path)
    lookup = {str(gene): idx for idx, gene in enumerate(adata.var_names.tolist())}
    selected = [gene for gene in requested if gene in lookup]
    missing = [gene for gene in requested if gene not in lookup]
    if missing:
        console.print(f"[yellow]Skipped[/yellow] {len(missing)} missing genes")
    if not selected:
        raise ValueError("gene list has no overlap with the dataset")
    return adata[:, [lookup[gene] for gene in selected]].copy()


def select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def ensure_x_matrix(adata: ad.AnnData, layer: str | None) -> ad.AnnData:
    matrix = select_matrix(adata, layer)
    if sparse.issparse(matrix):
        x_matrix = sparse.csc_matrix(matrix, dtype=np.float32)
    else:
        x_matrix = np.asarray(matrix, dtype=np.float32)
    prepared = ad.AnnData(
        X=x_matrix,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy(),
    )
    prepared.var_names = adata.var_names.copy()
    prepared.obs_names = adata.obs_names.copy()
    return prepared


def export_group_lists(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    filter_significant: bool,
    fdr_threshold: float,
    log2fc_threshold: float,
    top_k: int | None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    group_column = "label" if "label" in df.columns else "perturbation"
    gene_column = "gene" if "gene" in df.columns else "feature"
    for label, group_df in df.groupby(group_column, sort=True):
        filtered = group_df.copy()
        if filter_significant:
            filtered = filtered[
                (filtered["fdr"] <= fdr_threshold)
                & (np.abs(filtered["log2_fold_change"]) >= log2fc_threshold)
            ].copy()
        filtered = filtered.sort_values(by=["fdr", gene_column])
        if top_k is not None:
            filtered = filtered.head(top_k)
        counts[str(label)] = int(filtered.shape[0])
        (output_dir / f"{label}.csv").write_text(
            filtered[[gene_column, "log2_fold_change", "fdr"]].to_csv(index=False),
            encoding="utf-8",
        )
        (output_dir / f"{label}.txt").write_text(
            "\n".join(filtered[gene_column].astype(str).tolist())
            + ("\n" if filtered.shape[0] > 0 else ""),
            encoding="utf-8",
        )
    return counts


def main() -> None:
    args = parse_args()
    if args.threads == 0:
        raise ValueError("--threads cannot be 0")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be positive")
    if args.clip_value <= 0:
        raise ValueError("--clip-value must be positive")
    if not (0.0 < args.fdr_threshold <= 1.0):
        raise ValueError("--fdr-threshold must be in (0, 1]")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be positive when provided")

    input_path = args.input_h5ad.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    gene_list_path = (
        None if args.gene_list is None else args.gene_list.expanduser().resolve()
    )

    intro = Table(show_header=False, box=None)
    intro.add_row("Input", str(input_path))
    intro.add_row("Groupby", args.groupby_key)
    intro.add_row("Reference", args.reference)
    intro.add_row("Layer", args.layer or "X")
    intro.add_row("Gene list", str(gene_list_path) if gene_list_path else "None")
    intro.add_row("Threads", str(args.threads))
    intro.add_row("Output CSV", str(output_csv))
    intro.add_row("Output dir", str(output_dir) if output_dir else "None")
    console.print(Panel(intro, title="Calc DEGs", border_style="cyan"))

    start_time = perf_counter()
    with console.status("Loading AnnData and preparing matrix..."):
        adata = ad.read_h5ad(input_path)
        if args.groupby_key not in adata.obs:
            raise KeyError(f"obs column {args.groupby_key!r} not found")
        adata = subset_genes(adata, gene_list_path)
        prepared = ensure_x_matrix(adata, args.layer)
        groups = None if not args.groups else list(dict.fromkeys(args.groups))

    obs_labels = prepared.obs[args.groupby_key].astype(str)
    if args.reference not in set(obs_labels.tolist()):
        raise ValueError(
            f"reference group {args.reference!r} not found in {args.groupby_key!r}"
        )
    if groups is not None:
        missing_groups = [
            group for group in groups if group not in set(obs_labels.tolist())
        ]
        if missing_groups:
            raise ValueError(f"requested groups missing from dataset: {missing_groups}")

    with console.status("Running hpdex differential expression..."):
        df = parallel_differential_expression(
            prepared,
            groupby_key=args.groupby_key,
            reference=args.reference,
            groups=groups,
            tie_correction=True,
            use_continuity=True,
            min_samples=args.min_samples,
            threads=args.threads,
            clip_value=args.clip_value,
            show_progress=args.show_progress,
        )

    output_df = to_plot_fg_format(df) if args.plot_fg_format else rename_deg_columns(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    exported_counts: dict[str, int] | None = None
    if output_dir is not None:
        exported_counts = export_group_lists(
            output_df,
            output_dir,
            filter_significant=not args.plot_fg_format,
            fdr_threshold=args.fdr_threshold,
            log2fc_threshold=args.log2fc_threshold,
            top_k=args.top_k,
        )
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "input_h5ad": str(input_path),
                    "groupby_key": args.groupby_key,
                    "reference": args.reference,
                    "groups": groups,
                    "layer": args.layer,
                    "n_cells": int(prepared.n_obs),
                    "n_genes": int(prepared.n_vars),
                    "threads": int(args.threads),
                    "min_samples": int(args.min_samples),
                    "clip_value": float(args.clip_value),
                    "filter_significant": bool(not args.plot_fg_format),
                    "fdr_threshold": float(args.fdr_threshold),
                    "log2fc_threshold": float(args.log2fc_threshold),
                    "top_k": args.top_k,
                    "exported_deg_counts": exported_counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = Table(title="DEG Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Cells", str(prepared.n_obs))
    summary.add_row("Genes", str(prepared.n_vars))
    summary.add_row("Rows", str(output_df.shape[0]))
    summary.add_row(
        "Output format",
        "plot-fg" if args.plot_fg_format else "full-renamed",
    )
    summary.add_row(
        "Groups",
        str(
            output_df["label"].nunique()
            if "label" in output_df.columns
            else output_df["perturbation"].nunique()
        ),
    )
    summary.add_row("Output CSV", str(output_csv))
    if output_dir is not None:
        summary.add_row("Output dir", str(output_dir))
        if exported_counts is not None:
            summary.add_row(
                "Exported DEG counts",
                ", ".join(f"{k}:{v}" for k, v in list(exported_counts.items())[:10]),
            )
    summary.add_row("Elapsed", f"{perf_counter() - start_time:.2f}s")
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(Panel(str(exc), title="calc_degs failed", border_style="red"))
        raise SystemExit(1) from exc
