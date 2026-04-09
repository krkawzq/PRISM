from __future__ import annotations

from pathlib import Path

import typer

from prism.cli.common import print_saved_path, resolve_optional_path, resolve_str

from .common import (
    console,
    intersect_gene_lists,
    load_var_names,
    print_gene_summary,
    write_gene_list,
    write_json,
)


def intersect_genes_command(
    input_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Two or more h5ad datasets."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output text file with one gene per line."
    ),
    summary_json_path: Path | None = typer.Option(
        None, "--summary-json", help="Optional JSON summary path."
    ),
    sort: str = typer.Option(
        "first", help="Ordering of overlapping genes: first or alpha."
    ),
) -> int:
    resolved_inputs = [path.expanduser().resolve() for path in input_paths]
    output_path = output_path.expanduser().resolve()
    summary_json_path = resolve_optional_path(summary_json_path)
    sort = resolve_str(sort)

    ordered_lists = [load_var_names(path) for path in resolved_inputs]
    ordered, metadata = intersect_gene_lists(ordered_lists, sort=sort)
    write_gene_list(output_path, ordered)
    if summary_json_path is not None:
        write_json(
            summary_json_path,
            {
                "inputs": [str(path) for path in resolved_inputs],
                "sort": metadata["sort"],
                "n_overlap": metadata["n_overlap"],
                "input_sizes": metadata["input_sizes"],
                "first10": ordered[:10],
            },
        )

    print_gene_summary(
        "Intersect Genes",
        Inputs=len(resolved_inputs),
        Overlap=metadata["n_overlap"],
        Sort=metadata["sort"],
        Output=output_path,
    )
    print_saved_path(console, output_path)
    if summary_json_path is not None:
        print_saved_path(console, summary_json_path)
    return 0


__all__ = ["intersect_genes_command"]
