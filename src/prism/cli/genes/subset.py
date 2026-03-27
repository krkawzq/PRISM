from __future__ import annotations

from pathlib import Path

import typer

from .common import console, read_gene_list


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


__all__ = ["subset_genes_command"]
