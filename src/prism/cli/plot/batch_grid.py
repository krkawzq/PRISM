from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prism.cli.common import ensure_mutually_exclusive
from prism.io import read_gene_list, read_string_list
from prism.model import load_checkpoint
from prism.plotting import (
    batch_grid_curve_sets_to_dataframe,
    batch_grid_summary_dataframe,
    load_label_grid_entries,
    parse_batch_grid_entries,
    plot_batch_grid_figure,
    plt,
    resolve_batch_grid_curve_sets,
    resolve_curve_mode,
    resolve_stat_fields,
    resolve_x_axis,
    resolve_y_scale,
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
)
from .common import option_sequence, option_value

console = Console()


def _resolve_genes(
    *,
    gene_names: list[str] | None,
    genes_path: Path | None,
    top_n: int | None,
) -> list[str]:
    gene_names = option_sequence(gene_names)
    genes_path = option_value(genes_path)
    top_n = option_value(top_n)
    ensure_mutually_exclusive(("--gene", gene_names), ("--genes", genes_path))
    if genes_path is not None:
        values = read_gene_list(genes_path.expanduser().resolve())
        limit = len(values) if top_n is None else min(top_n, len(values))
        return values[:limit]
    if not gene_names:
        raise ValueError("provide either --gene or --genes")
    return list(dict.fromkeys(gene_names))


def _resolve_label_entries(
    *,
    checkpoint_path: Path,
    labels: list[str] | None,
    labels_path: Path | None,
    label_grid_csv_path: Path | None,
):
    checkpoint_path = option_value(checkpoint_path)
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    label_grid_csv_path = option_value(label_grid_csv_path)
    if label_grid_csv_path is not None:
        entries = load_label_grid_entries(label_grid_csv_path.expanduser().resolve())
        if labels is None and labels_path is None:
            return entries
        requested = (
            list(dict.fromkeys(labels))
            if labels_path is None
            else list(dict.fromkeys(read_string_list(labels_path.expanduser().resolve())))
        )
        requested_set = set(requested)
        return [entry for entry in entries if entry.label in requested_set]

    ensure_mutually_exclusive(("--label", labels), ("--labels", labels_path))
    resolved_labels = (
        sorted(load_checkpoint(checkpoint_path).label_priors)
        if labels_path is None and not labels
        else (
            list(dict.fromkeys(labels))
            if labels_path is None
            else list(dict.fromkeys(read_string_list(labels_path.expanduser().resolve())))
        )
    )
    return parse_batch_grid_entries(resolved_labels)


def plot_batch_grid_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    gene_names: list[str] = typer.Option(
        None, "--gene", help="Repeatable gene name to plot."
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional gene list text/JSON file. Uses the first top-n genes.",
    ),
    top_n: int | None = typer.Option(
        None,
        "--top-n",
        min=1,
        help="Number of leading genes to use with --genes.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable labels to include.",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON file listing labels to include.",
    ),
    label_grid_csv_path: Path | None = typer.Option(
        None,
        "--label-grid-csv",
        exists=True,
        dir_okay=False,
        help="Optional CSV with columns label,batch,perturbation.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV exporting all batch-grid curves.",
    ),
    summary_csv_path: Path | None = typer.Option(
        None,
        "--summary-csv",
        help="Optional CSV exporting per-cell summary statistics.",
    ),
    x_axis: str = typer.Option("mu", help="x axis: mu or p."),
    curve_mode: str = typer.Option(
        "density",
        help="Curve mode: " + ", ".join(SUPPORTED_CURVE_MODES) + ".",
    ),
    y_scale: str = typer.Option(
        "linear",
        help="y axis scale: " + ", ".join(SUPPORTED_Y_SCALES) + ".",
    ),
    mass_quantile: float = typer.Option(
        0.995,
        min=0.5,
        max=1.0,
        help="Upper cumulative mass used to truncate the displayed axis.",
    ),
    image_format: str = typer.Option(
        "svg",
        help="Vector image format for per-gene figures: svg, pdf, or eps.",
    ),
    dpi: int = typer.Option(180, min=1, help="Output figure DPI."),
    stat_fields: list[str] = typer.Option(
        None,
        "--stat",
        help=(
            "Repeatable per-cell stats to annotate: "
            + ", ".join(SUPPORTED_STAT_FIELDS)
            + "."
        ),
    ),
    hide_empty: bool = typer.Option(
        False,
        "--hide-empty/--show-empty",
        help="Hide empty batch-grid cells with no matching prior.",
    ),
    show_axis_ticks: bool = typer.Option(
        True,
        "--show-axis-ticks/--hide-axis-ticks",
        help="Show axis tick labels inside batch-grid cells.",
    ),
    panel_width: float = typer.Option(0.0, min=0.0, help="Optional panel width override."),
    panel_height: float = typer.Option(0.0, min=0.0, help="Optional panel height override."),
) -> int:
    checkpoint_path = option_value(checkpoint_path)
    gene_names = option_sequence(gene_names)
    genes_path = option_value(genes_path)
    top_n = option_value(top_n)
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    label_grid_csv_path = option_value(label_grid_csv_path)
    output_dir = option_value(output_dir)
    output_csv_path = option_value(output_csv_path)
    summary_csv_path = option_value(summary_csv_path)
    x_axis = str(option_value(x_axis))
    curve_mode = str(option_value(curve_mode))
    y_scale = str(option_value(y_scale))
    mass_quantile = float(option_value(mass_quantile))
    image_format = str(option_value(image_format))
    dpi = int(option_value(dpi))
    stat_fields = option_sequence(stat_fields)
    hide_empty = bool(option_value(hide_empty))
    show_axis_ticks = bool(option_value(show_axis_ticks))
    panel_width = float(option_value(panel_width))
    panel_height = float(option_value(panel_height))

    image_format_resolved = image_format.strip().lower()
    if image_format_resolved not in {"svg", "pdf", "eps"}:
        raise ValueError("--image-format must be one of: svg, pdf, eps")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    resolved_genes = _resolve_genes(
        gene_names=gene_names,
        genes_path=genes_path,
        top_n=top_n,
    )
    entries = _resolve_label_entries(
        checkpoint_path=checkpoint_path,
        labels=labels,
        labels_path=labels_path,
        label_grid_csv_path=label_grid_csv_path,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    curve_sets, batches, perturbations = resolve_batch_grid_curve_sets(
        checkpoint,
        gene_names=resolved_genes,
        entries=entries,
    )
    resolved_x_axis = resolve_x_axis(x_axis)
    resolved_curve_mode = resolve_curve_mode(curve_mode)
    resolved_y_scale = resolve_y_scale(y_scale)
    resolved_stat_fields = resolve_stat_fields(stat_fields)
    resolved_panel_width = 2.2 if panel_width <= 0 else float(panel_width)
    resolved_panel_height = 1.9 if panel_height <= 0 else float(panel_height)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = Table(show_header=False, box=None)
    summary.add_row("Checkpoint", str(checkpoint_path))
    summary.add_row("Genes", str(len(resolved_genes)))
    summary.add_row("Batches", ", ".join(batches))
    summary.add_row("Perturbations", str(len(perturbations)))
    summary.add_row("Output dir", str(output_dir))
    console.print(summary)

    for gene_name, curve_map in curve_sets.items():
        if not curve_map:
            console.print(f"[yellow]Skipped[/yellow] {gene_name}: no matching label priors")
            continue
        fig = plot_batch_grid_figure(
            gene_name,
            curve_map,
            batches=batches,
            perturbations=perturbations,
            x_axis=resolved_x_axis,
            mass_quantile=mass_quantile,
            curve_mode=resolved_curve_mode,
            y_scale=resolved_y_scale,
            stats_fields=resolved_stat_fields,
            hide_empty=hide_empty,
            show_axis_ticks=show_axis_ticks,
            panel_width=resolved_panel_width,
            panel_height=resolved_panel_height,
        )
        output_path = output_dir / f"{gene_name}.{image_format_resolved}"
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            format=image_format_resolved,
        )
        plt.close(fig)
        console.print(f"[bold green]Saved[/bold green] {output_path}")

    if output_csv_path is not None:
        output_csv_path = output_csv_path.expanduser().resolve()
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        batch_grid_curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis).to_csv(
            output_csv_path, index=False
        )
        console.print(f"[bold green]Saved[/bold green] {output_csv_path}")
    if summary_csv_path is not None:
        summary_csv_path = summary_csv_path.expanduser().resolve()
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        batch_grid_summary_dataframe(curve_sets).to_csv(summary_csv_path, index=False)
        console.print(f"[bold green]Saved[/bold green] {summary_csv_path}")
    return 0


__all__ = ["plot_batch_grid_command"]
