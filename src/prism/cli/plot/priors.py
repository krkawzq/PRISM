from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from prism.cli.common import ensure_mutually_exclusive
from prism.io import read_gene_list, read_string_list
from prism.model import load_checkpoint
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution
from prism.plotting import (
    curve_sets_to_dataframe,
    curve_sets_summary_dataframe,
    load_annotation_tables,
    plot_prior_facet_figure,
    plot_prior_overlay_figure,
    plt,
    resolve_curve_mode,
    resolve_prior_curve_sets,
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


def _resolve_labels(
    *,
    labels: list[str] | None,
    labels_path: Path | None,
) -> list[str] | None:
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    ensure_mutually_exclusive(("--label", labels), ("--labels", labels_path))
    if labels_path is None:
        return None if not labels else list(dict.fromkeys(labels))
    return list(dict.fromkeys(read_string_list(labels_path.expanduser().resolve())))


def plot_priors_command(
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
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV path for exported curve coordinates and weights.",
    ),
    summary_csv_path: Path | None = typer.Option(
        None,
        "--summary-csv",
        help="Optional CSV path for per-curve summary statistics.",
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
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON file listing labels to include.",
    ),
    x_axis: str = typer.Option(
        "auto",
        help="x axis: auto, mu, p, or rate. auto chooses rate for rate-grid priors, otherwise mu.",
    ),
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
    show_legend: bool = typer.Option(
        True,
        "--show-legend/--no-show-legend",
        help="Show legend in overlay layout.",
    ),
    stat_fields: list[str] = typer.Option(
        None,
        "--stat",
        help=(
            "Repeatable per-curve stats to annotate: "
            + ", ".join(SUPPORTED_STAT_FIELDS)
            + "."
        ),
    ),
    panel_width: float = typer.Option(
        0.0, min=0.0, help="Optional panel width override."
    ),
    panel_height: float = typer.Option(
        0.0, min=0.0, help="Optional panel height override."
    ),
) -> int:
    gene_names = option_sequence(gene_names)
    genes_path = option_value(genes_path)
    top_n = option_value(top_n)
    output_path = option_value(output_path)
    output_csv_path = option_value(output_csv_path)
    summary_csv_path = option_value(summary_csv_path)
    annot_csv_paths = option_sequence(annot_csv_paths)
    annot_names = option_sequence(annot_names)
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    x_axis = str(option_value(x_axis))
    curve_mode = str(option_value(curve_mode))
    y_scale = str(option_value(y_scale))
    mass_quantile = float(option_value(mass_quantile))
    include_global = bool(option_value(include_global))
    layout = str(option_value(layout))
    show_subplot_labels = bool(option_value(show_subplot_labels))
    show_legend = bool(option_value(show_legend))
    stat_fields = option_sequence(stat_fields)
    panel_width = float(option_value(panel_width))
    panel_height = float(option_value(panel_height))

    resolved_layout = layout.strip().lower()
    if resolved_layout not in {"overlay", "facet"}:
        raise ValueError("layout must be overlay or facet")
    resolved_genes = _resolve_genes(
        gene_names=gene_names,
        genes_path=genes_path,
        top_n=top_n,
    )
    if genes_path is not None:
        resolved_layout = "facet"
    resolved_labels = _resolve_labels(labels=labels, labels_path=labels_path)
    resolved_curve_mode = resolve_curve_mode(curve_mode)
    resolved_y_scale = resolve_y_scale(y_scale)
    resolved_stat_fields = resolve_stat_fields(stat_fields)
    overlay_panel_width = 8.2 if panel_width <= 0 else float(panel_width)
    overlay_panel_height = 4.8 if panel_height <= 0 else float(panel_height)
    facet_panel_width = 4.6 if panel_width <= 0 else float(panel_width)
    facet_panel_height = 3.2 if panel_height <= 0 else float(panel_height)

    annotation_tables = None
    if annot_csv_paths:
        if resolved_layout != "facet":
            raise ValueError("--annot-csv is only supported in facet layout")
        annotation_tables = load_annotation_tables(
            [path.expanduser().resolve() for path in annot_csv_paths],
            None if not annot_names else annot_names,
        )

    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    resolve_cli_checkpoint_distribution(
        checkpoint,
        command_name="prism plot priors",
    )
    curve_sets = resolve_prior_curve_sets(
        checkpoint,
        gene_names=resolved_genes,
        labels=resolved_labels,
        include_global=include_global,
    )
    if x_axis.strip().lower() == "auto":
        first_curve = next(iter(next(iter(curve_sets.values()))))
        resolved_x_axis = "rate" if first_curve.grid_domain == "rate" else "mu"
    else:
        resolved_x_axis = resolve_x_axis(x_axis)
    fig = (
        plot_prior_facet_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=mass_quantile,
            show_subplot_labels=show_subplot_labels,
            annotation_tables=annotation_tables,
            curve_mode=resolved_curve_mode,
            y_scale=resolved_y_scale,
            stats_fields=resolved_stat_fields,
            panel_width=facet_panel_width,
            panel_height=facet_panel_height,
        )
        if resolved_layout == "facet"
        else plot_prior_overlay_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=mass_quantile,
            curve_mode=resolved_curve_mode,
            y_scale=resolved_y_scale,
            show_legend=show_legend,
            stats_fields=resolved_stat_fields,
            panel_width=overlay_panel_width,
            panel_height=overlay_panel_height,
        )
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if output_csv_path is not None:
        output_csv_path = output_csv_path.expanduser().resolve()
        df = curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
    if summary_csv_path is not None:
        summary_csv_path = summary_csv_path.expanduser().resolve()
        summary_df = curve_sets_summary_dataframe(curve_sets)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False)
    return 0


__all__ = ["plot_priors_command"]
