from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import (
    print_saved_path,
    resolve_int,
    resolve_optional_int,
    resolve_optional_path,
)

from .common import console, print_gene_summary, read_gene_list, subset_gene_list, write_gene_list


def subset_genes_command(
    input_genes: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input gene list text file."
    ),
    output_path: Path = typer.Option(
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
    input_genes = input_genes.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    top_k = resolve_optional_int(top_k)
    start = resolve_int(start)
    end = resolve_optional_int(end)
    intersect_with = resolve_optional_path(intersect_with)
    exclude_with = resolve_optional_path(exclude_with)

    genes = read_gene_list(input_genes)
    intersect_genes = None if intersect_with is None else read_gene_list(intersect_with)
    exclude_genes = None if exclude_with is None else read_gene_list(exclude_with)
    selected, metadata = subset_gene_list(
        genes,
        start=start,
        end=end,
        top_k=top_k,
        intersect_genes=intersect_genes,
        exclude_genes=exclude_genes,
    )
    write_gene_list(output_path, selected)

    print_gene_summary(
        "Subset Genes",
        Input=input_genes,
        GenesIn=metadata["n_input_genes"],
        GenesFiltered=metadata["n_after_filtering"],
        GenesOut=metadata["n_output_genes"],
        Output=output_path,
    )
    print_saved_path(console, output_path)
    return 0


__all__ = ["subset_genes_command"]
