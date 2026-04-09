from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import cast

import typer

from prism.cli.common import (
    console,
    print_elapsed,
    print_saved_path,
    resolve_bool,
    resolve_float,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_path,
    resolve_str,
    unwrap_typer_value,
)
from prism.model import ModelCheckpoint, load_checkpoint
from prism.plotting import (
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
    curve_sets_summary_dataframe,
    curve_sets_to_dataframe,
    default_checkpoint_name,
    load_annotation_tables,
    plot_prior_facet_figure,
    plot_prior_overlay_figure,
    plt,
    resolve_curve_mode,
    resolve_multi_checkpoint_prior_curve_sets,
    resolve_stat_fields,
    resolve_x_axis,
    resolve_y_scale,
)

from .common import (
    normalize_layout,
    resolve_gene_names,
    resolve_label_names,
    resolve_optional_list,
)


def _resolve_annotation_names(
    value: list[str] | None | object,
) -> list[str] | None:
    resolved = resolve_optional_list(value)
    if resolved is None:
        return None
    return list(resolved)


def _resolve_checkpoint_name_values(
    value: list[str] | None | object,
) -> list[str] | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return [str(item).strip() for item in cast(list[str], resolved)]


def _normalize_missing_policy(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"error", "drop"}:
        raise ValueError("missing_policy must be one of: error, drop")
    return resolved


def _resolve_checkpoint_specs(
    checkpoint_paths: list[Path],
    checkpoint_name_values: list[str] | None,
) -> list[tuple[str, Path, ModelCheckpoint]]:
    if checkpoint_name_values is not None and len(checkpoint_name_values) != len(
        checkpoint_paths
    ):
        raise ValueError("--checkpoint-name count must match checkpoint paths")
    resolved_paths = [path.expanduser().resolve() for path in checkpoint_paths]
    checkpoints = [(path, load_checkpoint(path)) for path in resolved_paths]
    explicit_names = checkpoint_name_values is not None
    names = (
        checkpoint_name_values
        if checkpoint_name_values is not None
        else [
            default_checkpoint_name(checkpoint, checkpoint_path=path)
            for path, checkpoint in checkpoints
        ]
    )
    if any(not name for name in names):
        raise ValueError("checkpoint names cannot be blank")
    if explicit_names and len(names) != len(set(names)):
        raise ValueError("--checkpoint-name values must be unique")
    if not explicit_names:
        seen: dict[str, int] = {}
        deduped_names: list[str] = []
        for name in names:
            next_index = seen.get(name, 0) + 1
            seen[name] = next_index
            deduped_names.append(name if next_index == 1 else f"{name}#{next_index}")
        names = deduped_names
    return [
        (name, path, checkpoint)
        for name, (path, checkpoint) in zip(names, checkpoints, strict=True)
    ]


def _resolve_x_axis_auto(curve_sets: dict[str, list[object]], *, x_axis: str) -> str:
    if x_axis.strip().lower() != "auto":
        return resolve_x_axis(x_axis)
    support_domains = {
        str(getattr(curve, "support_domain"))
        for curves in curve_sets.values()
        for curve in curves
    }
    if len(support_domains) != 1:
        raise ValueError(
            "x_axis='auto' requires all selected curves to share the same support domain"
        )
    support_domain = next(iter(support_domains))
    return "rate" if support_domain == "rate" else "scaled"


def plot_priors_command(
    checkpoint_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="One or more checkpoint paths."
    ),
    checkpoint_names: list[str] | None = typer.Option(
        None,
        "--checkpoint-name",
        help="Optional repeatable display names aligned with checkpoint paths.",
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
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV path for exported curve coordinates.",
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
        help="Optional repeatable CSV files with columns gene,label,... for facet annotations.",
    ),
    annot_names: list[str] | None = typer.Option(
        None,
        "--annot-name",
        help="Optional repeatable names for --annot-csv, aligned by position.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable label priors to include. Defaults to all label priors.",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional file listing labels to include.",
    ),
    x_axis: str = typer.Option(
        "auto",
        help="x axis: auto, scaled, support, or rate. auto chooses rate for rate-domain priors, otherwise scaled.",
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
        help="Plot layout: overlay or facet. Using --genes forces facet layout.",
    ),
    show_subplot_labels: bool = typer.Option(
        False,
        "--show-subplot-labels/--no-show-subplot-labels",
        help="Render gene/source labels inside every facet subplot.",
    ),
    show_legend: bool = typer.Option(
        True,
        "--show-legend/--no-show-legend",
        help="Show legend in overlay layout.",
    ),
    missing_policy: str = typer.Option(
        "error",
        help="How to handle missing genes or labels across checkpoints: error or drop.",
    ),
    stat_fields: list[str] | None = typer.Option(
        None,
        "--stat",
        help="Repeatable per-curve stats to annotate: "
        + ", ".join(SUPPORTED_STAT_FIELDS)
        + ".",
    ),
    panel_width: float | None = typer.Option(
        None, min=0.0, help="Optional panel width override."
    ),
    panel_height: float | None = typer.Option(
        None, min=0.0, help="Optional panel height override."
    ),
) -> int:
    start_time = perf_counter()
    genes_path = resolve_optional_path(genes_path)
    top_n = resolve_optional_int(top_n)
    output_path = output_path.expanduser().resolve()
    output_csv_path = resolve_optional_path(output_csv_path)
    summary_csv_path = resolve_optional_path(summary_csv_path)
    labels_path = resolve_optional_path(labels_path)
    x_axis = resolve_str(x_axis)
    curve_mode = resolve_str(curve_mode)
    y_scale = resolve_str(y_scale)
    layout = resolve_str(layout)
    annot_csv_resolved = (
        None
        if unwrap_typer_value(annot_csv_paths) is None
        else [
            Path(path).expanduser().resolve()
            for path in unwrap_typer_value(annot_csv_paths)
        ]
    )
    annot_names_resolved = _resolve_annotation_names(annot_names)
    mass_quantile = resolve_float(mass_quantile)
    include_global = resolve_bool(include_global)
    show_subplot_labels = resolve_bool(show_subplot_labels)
    show_legend = resolve_bool(show_legend)
    missing_policy = _normalize_missing_policy(resolve_str(missing_policy))
    panel_width = resolve_optional_float(panel_width)
    panel_height = resolve_optional_float(panel_height)

    resolved_layout = normalize_layout(layout)
    resolved_genes = resolve_gene_names(
        gene_names=resolve_optional_list(gene_names),
        genes_path=genes_path,
        top_n=top_n,
    )
    if genes_path is not None:
        resolved_layout = "facet"
    resolved_labels = resolve_label_names(
        labels=resolve_optional_list(labels),
        labels_path=labels_path,
    )
    resolved_curve_mode = resolve_curve_mode(curve_mode)
    resolved_y_scale = resolve_y_scale(y_scale)
    resolved_stat_fields = resolve_stat_fields(resolve_optional_list(stat_fields))

    annotation_tables = None
    if annot_csv_resolved:
        if resolved_layout != "facet":
            raise ValueError("--annot-csv is only supported with --layout facet")
        annotation_tables = load_annotation_tables(annot_csv_resolved, annot_names_resolved)

    checkpoint_specs = _resolve_checkpoint_specs(
        checkpoint_paths,
        _resolve_checkpoint_name_values(checkpoint_names),
    )
    curve_sets = resolve_multi_checkpoint_prior_curve_sets(
        checkpoint_specs,
        gene_names=resolved_genes,
        labels=resolved_labels,
        include_global=include_global,
        missing_policy=missing_policy,
    )
    resolved_x_axis = _resolve_x_axis_auto(curve_sets, x_axis=x_axis)
    overlay_panel_width = 8.2 if panel_width is None or panel_width <= 0 else panel_width
    overlay_panel_height = (
        4.8 if panel_height is None or panel_height <= 0 else panel_height
    )
    facet_panel_width = 4.6 if panel_width is None or panel_width <= 0 else panel_width
    facet_panel_height = 3.2 if panel_height is None or panel_height <= 0 else panel_height

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_saved_path(console, output_path)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis).to_csv(
            output_csv_path, index=False
        )
        print_saved_path(console, output_csv_path)
    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        curve_sets_summary_dataframe(curve_sets).to_csv(summary_csv_path, index=False)
        print_saved_path(console, summary_csv_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


__all__ = ["plot_priors_command"]
