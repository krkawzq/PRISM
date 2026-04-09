from __future__ import annotations

from pathlib import Path
from time import perf_counter

import typer

from prism.cli.common import (
    console,
    print_elapsed,
    print_key_value_table,
    print_saved_path,
    resolve_bool,
    resolve_float,
    resolve_int,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_path,
    resolve_str,
)
from prism.model import load_checkpoint
from prism.plotting import (
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
    batch_grid_curve_sets_to_dataframe,
    batch_grid_summary_dataframe,
    plot_batch_grid_figure,
    plt,
    resolve_batch_grid_curve_sets,
    resolve_curve_mode,
    resolve_stat_fields,
    resolve_x_axis,
    resolve_y_scale,
)

from .common import (
    normalize_image_format,
    resolve_gene_names,
    resolve_label_entries,
    resolve_optional_list,
)


def plot_batch_grid_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    gene_names: list[str] | None = typer.Option(
        None, "--gene", help="Repeatable gene name to plot."
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional gene list file. Uses the first --top-n genes when provided.",
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
        help="Optional file listing labels to include.",
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
    x_axis: str = typer.Option(
        "scaled",
        help="x axis: scaled, support, or rate.",
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
    image_format: str = typer.Option(
        "svg",
        help="Per-gene image format: svg, pdf, or eps.",
    ),
    dpi: int = typer.Option(180, min=1, help="Output figure DPI."),
    stat_fields: list[str] | None = typer.Option(
        None,
        "--stat",
        help="Repeatable per-cell stats to annotate: "
        + ", ".join(SUPPORTED_STAT_FIELDS)
        + ".",
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
    panel_width: float | None = typer.Option(
        None, min=0.0, help="Optional panel width override."
    ),
    panel_height: float | None = typer.Option(
        None, min=0.0, help="Optional panel height override."
    ),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    genes_path = resolve_optional_path(genes_path)
    top_n = resolve_optional_int(top_n)
    labels_path = resolve_optional_path(labels_path)
    label_grid_csv_path = resolve_optional_path(label_grid_csv_path)
    output_dir = output_dir.expanduser().resolve()
    output_csv_path = resolve_optional_path(output_csv_path)
    summary_csv_path = resolve_optional_path(summary_csv_path)
    x_axis = resolve_str(x_axis)
    curve_mode = resolve_str(curve_mode)
    y_scale = resolve_str(y_scale)
    image_format = resolve_str(image_format)
    mass_quantile = resolve_float(mass_quantile)
    dpi = resolve_int(dpi)
    hide_empty = resolve_bool(hide_empty)
    show_axis_ticks = resolve_bool(show_axis_ticks)
    panel_width = resolve_optional_float(panel_width)
    panel_height = resolve_optional_float(panel_height)

    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint.has_label_priors:
        raise ValueError("checkpoint has no label priors")
    resolved_genes = resolve_gene_names(
        gene_names=resolve_optional_list(gene_names),
        genes_path=genes_path,
        top_n=top_n,
    )
    entries = resolve_label_entries(
        checkpoint=checkpoint,
        labels=resolve_optional_list(labels),
        labels_path=labels_path,
        label_grid_csv_path=label_grid_csv_path,
    )
    curve_sets, batches, perturbations = resolve_batch_grid_curve_sets(
        checkpoint,
        gene_names=resolved_genes,
        entries=entries,
    )
    resolved_x_axis = resolve_x_axis(x_axis)
    resolved_curve_mode = resolve_curve_mode(curve_mode)
    resolved_y_scale = resolve_y_scale(y_scale)
    resolved_image_format = normalize_image_format(image_format)
    resolved_stat_fields = resolve_stat_fields(resolve_optional_list(stat_fields))
    resolved_panel_width = 2.2 if panel_width is None or panel_width <= 0 else panel_width
    resolved_panel_height = (
        1.9 if panel_height is None or panel_height <= 0 else panel_height
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print_key_value_table(
        console,
        title="Batch Grid Plot",
        values={
            "Checkpoint": checkpoint_path,
            "Genes": len(resolved_genes),
            "Batches": ", ".join(batches),
            "Perturbations": len(perturbations),
            "Output dir": output_dir,
        },
    )
    for gene_name, curve_map in curve_sets.items():
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
        output_path = output_dir / f"{gene_name}.{resolved_image_format}"
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            format=resolved_image_format,
        )
        plt.close(fig)
        print_saved_path(console, output_path)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        batch_grid_curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis).to_csv(
            output_csv_path, index=False
        )
        print_saved_path(console, output_csv_path)
    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        batch_grid_summary_dataframe(curve_sets).to_csv(summary_csv_path, index=False)
        print_saved_path(console, summary_csv_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


__all__ = ["plot_batch_grid_command"]
