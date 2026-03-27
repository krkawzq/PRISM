from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy import sparse

from prism.model import load_checkpoint

genes_app = typer.Typer(help="Gene list utilities.", no_args_is_help=True)
console = Console()
EPS = 1e-12


@genes_app.command("intersect")
def intersect_genes_command(
    input_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Two or more h5ad datasets."
    ),
    output_genes: Path = typer.Option(
        ..., "--output-genes", help="Output text file with one gene per line."
    ),
    output_json: Path | None = typer.Option(
        None, "--output-json", help="Optional JSON summary path."
    ),
    sort: str = typer.Option(
        "first", help="Ordering of overlapping genes: first or alpha."
    ),
) -> int:
    if len(input_paths) < 2:
        raise ValueError("intersect requires at least two datasets")
    ordered_lists = [
        load_var_names(path.expanduser().resolve()) for path in input_paths
    ]
    overlap = set(ordered_lists[0])
    for genes in ordered_lists[1:]:
        overlap &= set(genes)
    if sort == "first":
        ordered = [gene for gene in ordered_lists[0] if gene in overlap]
    elif sort == "alpha":
        ordered = sorted(overlap)
    else:
        raise ValueError("sort must be 'first' or 'alpha'")
    output_genes = output_genes.expanduser().resolve()
    output_genes.parent.mkdir(parents=True, exist_ok=True)
    output_genes.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    payload = {
        "inputs": [str(path.expanduser().resolve()) for path in input_paths],
        "n_overlap": len(ordered),
        "sort": sort,
        "first10": ordered[:10],
    }
    if output_json is not None:
        output_json = output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = Table(title="Gene Intersection")
    table.add_column("Datasets", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_row(str(len(input_paths)), str(len(ordered)))
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_genes}")
    return 0


@genes_app.command("subset")
def subset_genes_command(
    input_genes: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input gene list text file."
    ),
    output_genes: Path = typer.Option(
        ..., "--output", "-o", help="Output gene list text file."
    ),
    top_k: int | None = typer.Option(
        None, min=1, help="Keep only the first K genes after filtering."
    ),
    start: int = typer.Option(0, min=0, help="Start index after filtering."),
    end: int | None = typer.Option(
        None, min=0, help="Exclusive end index after filtering."
    ),
    intersect_with: Path | None = typer.Option(
        None,
        "--intersect",
        exists=True,
        dir_okay=False,
        help="Keep only genes also present in this file.",
    ),
    exclude_with: Path | None = typer.Option(
        None,
        "--exclude",
        exists=True,
        dir_okay=False,
        help="Drop genes present in this file.",
    ),
) -> int:
    genes = read_gene_list(input_genes.expanduser().resolve())
    if intersect_with is not None:
        keep = set(read_gene_list(intersect_with.expanduser().resolve()))
        genes = [gene for gene in genes if gene in keep]
    if exclude_with is not None:
        drop = set(read_gene_list(exclude_with.expanduser().resolve()))
        genes = [gene for gene in genes if gene not in drop]
    resolved_end = len(genes) if end is None else min(end, len(genes))
    if resolved_end < start:
        raise ValueError("end must be >= start")
    genes = genes[start:resolved_end]
    if top_k is not None:
        genes = genes[:top_k]
    output_genes = output_genes.expanduser().resolve()
    output_genes.parent.mkdir(parents=True, exist_ok=True)
    output_genes.write_text("\n".join(genes) + "\n", encoding="utf-8")
    console.print(f"[bold green]Saved[/bold green] {output_genes} ({len(genes)} genes)")
    return 0


@genes_app.command("rank")
def rank_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad or checkpoint path."
    ),
    method: str = typer.Option(..., "--method", help="Ranking method."),
    output_ranked_genes: Path = typer.Option(
        ..., "--output-ranked-genes", help="Output text file with ranked genes."
    ),
    output_json: Path | None = typer.Option(
        None, "--output-json", help="Optional JSON summary path."
    ),
    top_k: int | None = typer.Option(
        None, min=1, help="Optional top-k summary for the JSON payload."
    ),
    hvg_flavor: str = typer.Option(
        "seurat_v3", help="Scanpy HVG flavor for HVG-based methods."
    ),
) -> int:
    input_path = input_path.expanduser().resolve()
    descending = True
    metadata: dict[str, Any] = {}
    if method == "prior-entropy":
        checkpoint = load_checkpoint(input_path)
        gene_names = np.asarray(checkpoint.gene_names)
        weights = np.asarray(checkpoint.priors.batched().weights, dtype=np.float64)
        scores = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
        metadata["score_definition"] = "prior entropy of F_g"
    elif method == "prior-entropy-reverse":
        checkpoint = load_checkpoint(input_path)
        gene_names = np.asarray(checkpoint.gene_names)
        weights = np.asarray(checkpoint.priors.batched().weights, dtype=np.float64)
        scores = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
        metadata["score_definition"] = "prior entropy of F_g"
        descending = False
    else:
        adata = ad.read_h5ad(input_path)
        if method == "hvg":
            gene_names, scores = compute_hvg_ranking_from_adata(
                adata, flavor=hvg_flavor
            )
            metadata["hvg_flavor"] = hvg_flavor
        elif method == "lognorm-variance":
            gene_names, scores = compute_lognorm_ranking(adata, dispersion=False)
            metadata["score_definition"] = "variance(log1p(normalize_total(X)))"
        elif method == "lognorm-dispersion":
            gene_names, scores = compute_lognorm_ranking(adata, dispersion=True)
            metadata["score_definition"] = (
                "variance(log1p(normalize_total(X))) / mean(log1p(normalize_total(X)))"
            )
        else:
            raise ValueError(f"unsupported method: {method}")
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    ranked = [str(gene_names[idx]) for idx in order.tolist()]
    output_ranked_genes = output_ranked_genes.expanduser().resolve()
    output_ranked_genes.parent.mkdir(parents=True, exist_ok=True)
    output_ranked_genes.write_text("\n".join(ranked) + "\n", encoding="utf-8")
    if output_json is not None:
        output_json = output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        preview_order = (
            order[:top_k] if top_k is not None else order[: min(50, len(order))]
        )
        payload = {
            "source_path": str(input_path),
            "method": method,
            "top_k": None if top_k is None else int(top_k),
            "gene_names": [str(gene_names[idx]) for idx in preview_order.tolist()],
            "scores": [float(scores[idx]) for idx in preview_order.tolist()],
            "metadata": metadata,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved[/bold green] {output_ranked_genes}")
    return 0


def read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_var_names(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def compute_hvg_ranking_from_adata(
    adata: ad.AnnData, *, flavor: str
) -> tuple[np.ndarray, np.ndarray]:
    import scanpy as sc

    try:
        sc.pp.highly_variable_genes(adata, flavor=cast(Any, flavor), inplace=True)
    except ImportError:
        if flavor not in {"seurat_v3", "seurat_v3_paper"}:
            raise
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=True)
    if "highly_variable_rank" in adata.var:
        score = -np.nan_to_num(
            np.asarray(adata.var["highly_variable_rank"], dtype=np.float64), nan=np.inf
        )
    elif "dispersions_norm" in adata.var:
        score = np.asarray(adata.var["dispersions_norm"], dtype=np.float64)
    else:
        raise ValueError("scanpy did not produce an HVG ranking field")
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
    )


def compute_lognorm_ranking(
    adata: ad.AnnData, *, dispersion: bool
) -> tuple[np.ndarray, np.ndarray]:
    matrix = adata.X
    if matrix is None:
        raise ValueError("input h5ad has empty X")
    if sparse.issparse(matrix):
        counts = np.asarray(cast(Any, matrix).toarray(), dtype=np.float64)
    else:
        counts = np.asarray(matrix, dtype=np.float64)
    totals = counts.sum(axis=1)
    target = float(np.median(totals))
    values = np.log1p(counts * (target / np.maximum(totals, 1.0))[:, None])
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    score = var / np.maximum(mean, EPS) if dispersion else var
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
    )
