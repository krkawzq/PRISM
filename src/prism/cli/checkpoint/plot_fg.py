from __future__ import annotations

from pathlib import Path

import typer

from prism.model import load_checkpoint

from .common import (
    console,
    curve_sets_to_dataframe,
    load_annotation_tables,
    load_gene_list_file,
    plot_fg_facet_figure,
    plot_fg_figure,
    resolve_plot_curve_sets,
    resolve_x_axis,
    plt,
)


def plot_fg_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    gene_names: list[str] = typer.Option(
        None, "--gene", help="Repeatable gene name to plot."
    ),
    gene_list_path: Path | None = typer.Option(
        None,
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional gene list text/JSON file. Uses the first top-n genes.",
    ),
    top_n: int | None = typer.Option(
        None,
        "--top-n",
        min=1,
        help="Number of leading genes to use with --gene-list.",
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV path for exported curve coordinates and weights.",
    ),
    annot_csv_paths: list[Path] | None = typer.Option(
        None,
        "--annot-csv",
        exists=True,
        dir_okay=False,
        help="Optional repeatable CSV files with columns gene,label,... for subplot annotations.",
    ),
    annot_names: list[str] | None = typer.Option(
        None,
        "--annot-name",
        help="Optional repeatable names for annotation CSVs, aligned with --annot-csv.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable label priors to include. If omitted, include all label priors.",
    ),
    x_axis: str = typer.Option("mu", help="x axis: mu or p."),
    mass_quantile: float = typer.Option(
        0.995,
        min=0.5,
        max=1.0,
        help="Upper cumulative mass used to truncate the displayed axis.",
    ),
    include_global: bool = typer.Option(
        True,
        "--include-global/--no-include-global",
        help="Include the global prior when present.",
    ),
    layout: str = typer.Option(
        "overlay",
        help="Plot layout: overlay or facet. Gene-list mode defaults to facet.",
    ),
    show_subplot_labels: bool = typer.Option(
        False,
        "--show-subplot-labels/--no-show-subplot-labels",
        help="Render gene/source labels inside every subplot.",
    ),
) -> int:
    if gene_list_path is not None and gene_names:
        raise ValueError("--gene and --gene-list are mutually exclusive")
    if gene_list_path is None and not gene_names:
        raise ValueError("provide either --gene or --gene-list")
    resolved_layout = layout.strip().lower()
    if resolved_layout not in {"overlay", "facet"}:
        raise ValueError("layout must be overlay or facet")

    if gene_list_path is not None:
        all_genes = load_gene_list_file(gene_list_path.expanduser().resolve())
        limit = len(all_genes) if top_n is None else min(top_n, len(all_genes))
        gene_names = all_genes[:limit]
        resolved_layout = "facet"
    annotation_tables = None
    if annot_csv_paths:
        if resolved_layout != "facet":
            raise ValueError("--annot-csv is only supported in facet layout")
        annotation_tables = load_annotation_tables(
            [path.expanduser().resolve() for path in annot_csv_paths],
            None if not annot_names else annot_names,
        )

    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    curve_sets = resolve_plot_curve_sets(
        checkpoint, gene_names=gene_names, labels=labels, include_global=include_global
    )
    resolved_x_axis = resolve_x_axis(x_axis)
    fig = (
        plot_fg_facet_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=float(mass_quantile),
            show_subplot_labels=show_subplot_labels,
            annotation_tables=annotation_tables,
        )
        if resolved_layout == "facet"
        else plot_fg_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=float(mass_quantile),
        )
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    if output_csv_path is not None:
        output_csv_path = output_csv_path.expanduser().resolve()
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis).to_csv(
            output_csv_path, index=False
        )
        console.print(f"[bold green]Saved[/bold green] {output_csv_path}")
    return 0


__all__ = ["plot_fg_command"]
