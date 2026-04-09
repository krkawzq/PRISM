from __future__ import annotations

from pathlib import Path

import anndata as ad
import typer

from prism.cli.common import console, print_key_value_table, resolve_bool
from prism.io import read_gene_list, write_h5ad


def _resolve_gene_indices(
    adata: ad.AnnData,
    gene_names: list[str],
    *,
    allow_missing: bool,
) -> tuple[list[int], list[str], list[str]]:
    if not gene_names:
        raise ValueError("gene list is empty")
    lookup = {str(name): idx for idx, name in enumerate(adata.var_names.tolist())}
    selected_names: list[str] = []
    selected_indices: list[int] = []
    missing: list[str] = []
    for gene in gene_names:
        idx = lookup.get(gene)
        if idx is None:
            missing.append(gene)
            continue
        selected_names.append(gene)
        selected_indices.append(idx)
    if missing and not allow_missing:
        raise ValueError(
            f"{len(missing)} genes are missing from the dataset, for example: {missing[:5]}"
        )
    if not selected_indices:
        raise ValueError("gene list has no overlap with the dataset")
    return selected_indices, selected_names, missing


def subset_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input AnnData file."
    ),
    genes_path: Path = typer.Option(
        ..., "--genes", exists=True, dir_okay=False, help="Gene-list text file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output subset AnnData file."
    ),
    allow_missing: bool = typer.Option(
        False,
        "--allow-missing/--strict",
        help="Skip genes absent from the dataset instead of failing.",
    ),
) -> int:
    input_path = input_path.expanduser().resolve()
    genes_path = genes_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    allow_missing = resolve_bool(allow_missing)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)
    gene_names = read_gene_list(genes_path)
    indices, selected_names, missing = _resolve_gene_indices(
        adata,
        list(gene_names),
        allow_missing=allow_missing,
    )

    subset = adata[:, indices].copy()
    subset.uns["gene_subset"] = {
        "source_h5ad": str(input_path),
        "gene_list_path": str(genes_path),
        "gene_list_format": "text",
        "allow_missing": bool(allow_missing),
        "n_obs": int(subset.n_obs),
        "n_vars_before": int(adata.n_vars),
        "n_vars_after": int(subset.n_vars),
        "n_requested_genes": int(len(gene_names)),
        "n_selected_genes": int(len(selected_names)),
        "n_missing_genes": int(len(missing)),
        "missing_genes_preview": missing[:10],
    }
    write_h5ad(subset, output_path)

    print_key_value_table(
        console,
        title="Subset Genes",
        values={
            "Input": input_path,
            "Gene list": genes_path,
            "Cells": f"{adata.n_obs} -> {subset.n_obs}",
            "Genes": f"{adata.n_vars} -> {subset.n_vars}",
            "Missing": len(missing),
            "Output": output_path,
        },
    )
    return 0


__all__ = ["subset_genes_command"]
