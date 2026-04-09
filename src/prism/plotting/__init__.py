from __future__ import annotations

from ._shared import resolve_plot_export
from .prior_annotations import (
    LabelGridEntry,
    load_annotation_tables,
    load_label_grid_entries,
    parse_batch_grid_entries,
)
from .prior_curves import (
    PriorCurve,
    batch_grid_curve_sets_to_dataframe,
    batch_grid_summary_dataframe,
    curve_sets_summary_dataframe,
    curve_sets_to_dataframe,
    default_checkpoint_name,
    resolve_batch_grid_curve_sets,
    resolve_multi_checkpoint_prior_curve_sets,
    resolve_prior_curve_sets,
)
from .prior_options import (
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
    resolve_curve_mode,
    resolve_stat_fields,
    resolve_x_axis,
    resolve_y_scale,
)
from .prior_stats import (
    display_cutoff,
    format_curve_stats,
    resolve_curve_y_values,
    resolve_x_values,
    summarize_prior_curve,
)


def __getattr__(name: str) -> object:
    try:
        return resolve_plot_export(name, package=__name__)
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc


__all__ = [
    "LabelGridEntry",
    "PriorCurve",
    "SUPPORTED_CURVE_MODES",
    "SUPPORTED_STAT_FIELDS",
    "SUPPORTED_Y_SCALES",
    "batch_grid_curve_sets_to_dataframe",
    "batch_grid_summary_dataframe",
    "curve_sets_summary_dataframe",
    "curve_sets_to_dataframe",
    "default_checkpoint_name",
    "display_cutoff",
    "format_curve_stats",
    "load_annotation_tables",
    "load_label_grid_entries",
    "parse_batch_grid_entries",
    "plot_batch_grid_figure",
    "plot_prior_facet_figure",
    "plot_prior_overlay_figure",
    "plt",
    "resolve_batch_grid_curve_sets",
    "resolve_curve_mode",
    "resolve_multi_checkpoint_prior_curve_sets",
    "resolve_curve_y_values",
    "resolve_prior_curve_sets",
    "resolve_stat_fields",
    "resolve_x_axis",
    "resolve_x_values",
    "resolve_y_scale",
    "summarize_prior_curve",
]
