from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from .common import console, load_var_names, write_json


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
        write_json(output_json.expanduser().resolve(), payload)
    table = Table(title="Gene Intersection")
    table.add_column("Datasets", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_row(str(len(input_paths)), str(len(ordered)))
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_genes}")
    return 0


__all__ = ["intersect_genes_command"]
