from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import print_saved_path, resolve_str

from .common import (
    SUPPORTED_GENE_SET_MODES,
    SUPPORTED_MERGE_METHODS,
    console,
    merge_gene_lists,
    print_gene_summary,
    read_gene_list,
    write_gene_list,
)


def merge_genes_command(
    input_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Two or more input gene-list files."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output text file with merged genes."
    ),
    method: str = typer.Option("rank-sum", help="Merge method: rank-sum."),
    gene_set_mode: str = typer.Option(
        "exact", help="Gene-set policy: exact, intersection, union."
    ),
) -> int:
    resolved_inputs = [path.expanduser().resolve() for path in input_paths]
    output_path = output_path.expanduser().resolve()
    method = resolve_str(method)
    gene_set_mode = resolve_str(gene_set_mode)

    inputs = [read_gene_list(path) for path in resolved_inputs]
    ordered, metadata = merge_gene_lists(
        inputs,
        method=method,
        gene_set_mode=gene_set_mode,
    )
    write_gene_list(output_path, ordered)

    print_gene_summary(
        "Merge Genes",
        Inputs=len(resolved_inputs),
        Method=metadata["merge_method"],
        GeneSetMode=metadata["gene_set_mode"],
        GenesOut=metadata["n_output_genes"],
        Output=output_path,
    )
    print_saved_path(console, output_path)
    return 0


__all__ = [
    "SUPPORTED_GENE_SET_MODES",
    "SUPPORTED_MERGE_METHODS",
    "merge_genes_command",
]
