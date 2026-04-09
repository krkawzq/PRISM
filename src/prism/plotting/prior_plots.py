from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

from .prior_stats import (
    display_cutoff,
    format_curve_stats,
    resolve_curve_y_values,
    resolve_x_values,
)

if TYPE_CHECKING:
    from .prior_curves import PriorCurve


COLOR_CYCLE = ["#1d4ed8", "#0f766e", "#c2410c", "#7c3aed", "#be123c", "#0891b2"]
LINE_STYLE_CYCLE = ["-", "--", ":", "-."]
GUI_BACKENDS = frozenset(
    {
        "gtk3agg",
        "gtk3cairo",
        "gtk4agg",
        "gtk4cairo",
        "macosx",
        "nbagg",
        "qtagg",
        "qtcairo",
        "qt5agg",
        "qt5cairo",
        "tkagg",
        "tkcairo",
        "webagg",
        "wx",
        "wxagg",
        "wxcairo",
    }
)


def _get_pyplot():
    if "matplotlib.pyplot" not in sys.modules:
        backend = str(matplotlib.get_backend()).lower()
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if backend in GUI_BACKENDS and not has_display:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def __getattr__(name: str) -> object:
    if name == "plt":
        return _get_pyplot()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _curve_y_for_plot(
    curve: "PriorCurve", *, curve_mode: str, y_scale: str
) -> np.ndarray:
    values = resolve_curve_y_values(curve, curve_mode=curve_mode)
    if y_scale == "log":
        return np.clip(values, 1e-12, None)
    return values


def _require_curve_sets(
    curve_sets: dict[str, list["PriorCurve"]],
) -> list[str]:
    genes = list(curve_sets)
    if not genes:
        raise ValueError("curve_sets cannot be empty")
    for gene_name, curves in curve_sets.items():
        if not curves:
            raise ValueError(f"gene {gene_name!r} has no curves to plot")
    return genes


def _display_max_for_curves(
    curves: list["PriorCurve"], *, x_axis: str, mass_quantile: float
) -> float:
    return max(
        max(
            display_cutoff(
                resolve_x_values(curve, x_axis=x_axis),
                curve.prior_probabilities,
                mass_quantile,
            ),
            1e-12,
        )
        for curve in curves
    )


def _curve_style_lookup(
    curve_sets: dict[str, list["PriorCurve"]],
) -> dict[str, tuple[str, str]]:
    curves = [curve for gene_curves in curve_sets.values() for curve in gene_curves]
    checkpoint_names = list(dict.fromkeys(curve.checkpoint_name for curve in curves))
    scope_names = list(dict.fromkeys(curve.scope_name for curve in curves))
    if len(checkpoint_names) > 1:
        checkpoint_colors = {
            checkpoint_name: COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            for idx, checkpoint_name in enumerate(checkpoint_names)
        }
        scope_styles = {
            scope_name: LINE_STYLE_CYCLE[idx % len(LINE_STYLE_CYCLE)]
            for idx, scope_name in enumerate(scope_names)
        }
        return {
            curve.source: (
                checkpoint_colors[curve.checkpoint_name],
                scope_styles[curve.scope_name],
            )
            for curve in curves
        }
    source_names = list(dict.fromkeys(curve.source for curve in curves))
    return {
        source_name: (COLOR_CYCLE[idx % len(COLOR_CYCLE)], "-")
        for idx, source_name in enumerate(source_names)
    }


def _plot_curve(
    ax,
    curve: "PriorCurve",
    *,
    x_axis: str,
    display_max: float,
    curve_mode: str,
    y_scale: str,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None = None,
) -> None:
    x_values = resolve_x_values(curve, x_axis=x_axis)
    y_values = _curve_y_for_plot(curve, curve_mode=curve_mode, y_scale=y_scale)
    mask = x_values <= display_max + 1e-12
    if not np.any(mask):
        mask = np.ones_like(x_values, dtype=bool)
    ax.plot(
        x_values[mask],
        y_values[mask],
        lw=linewidth,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def plot_prior_overlay_figure(
    curve_sets: dict[str, list["PriorCurve"]],
    *,
    x_axis: str,
    mass_quantile: float,
    curve_mode: str = "density",
    y_scale: str = "linear",
    show_legend: bool = True,
    stats_fields: tuple[str, ...] = (),
    panel_width: float = 8.2,
    panel_height: float = 4.8,
):
    plt = _get_pyplot()
    genes = _require_curve_sets(curve_sets)
    n_genes = len(genes)
    cols = min(2, max(n_genes, 1))
    rows = (n_genes + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(panel_width * cols, panel_height * rows), squeeze=False
    )
    style_lookup = _curve_style_lookup(curve_sets)
    for ax in axes.ravel():
        ax.set_visible(False)
    for panel_idx, gene_name in enumerate(genes):
        ax = axes.ravel()[panel_idx]
        ax.set_visible(True)
        curves = curve_sets[gene_name]
        display_max = _display_max_for_curves(
            curves,
            x_axis=x_axis,
            mass_quantile=mass_quantile,
        )
        for curve in curves:
            color, linestyle = style_lookup[curve.source]
            _plot_curve(
                ax,
                curve,
                x_axis=x_axis,
                display_max=display_max,
                curve_mode=curve_mode,
                y_scale=y_scale,
                color=color,
                linestyle=linestyle,
                label=curve.source,
                linewidth=2.0,
            )
        ax.set_title(gene_name)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("prior mass" if curve_mode == "density" else "prior CDF")
        ax.set_yscale(y_scale)
        if show_legend:
            ax.legend(frameon=False)
        if stats_fields:
            lines = [
                f"{curve.source}: {format_curve_stats(curve, fields=stats_fields)}"
                for curve in curves
            ]
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
        ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def plot_prior_facet_figure(
    curve_sets: dict[str, list["PriorCurve"]],
    *,
    x_axis: str,
    mass_quantile: float,
    show_subplot_labels: bool = False,
    annotation_tables: dict[str, dict[tuple[str, str], str]] | None = None,
    curve_mode: str = "density",
    y_scale: str = "linear",
    stats_fields: tuple[str, ...] = (),
    panel_width: float = 4.6,
    panel_height: float = 3.2,
):
    plt = _get_pyplot()
    genes = _require_curve_sets(curve_sets)
    column_labels: list[str] = []
    seen: set[str] = set()
    for curves in curve_sets.values():
        for curve in curves:
            if curve.source not in seen:
                seen.add(curve.source)
                column_labels.append(curve.source)
    n_rows = len(genes)
    n_cols = len(column_labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    style_lookup = _curve_style_lookup(curve_sets)
    y_label = "prior mass" if curve_mode == "density" else "prior CDF"
    for row_idx, gene_name in enumerate(genes):
        curve_map = {curve.source: curve for curve in curve_sets[gene_name]}
        row_display_max = _display_max_for_curves(
            list(curve_map.values()),
            x_axis=x_axis,
            mass_quantile=mass_quantile,
        )
        for col_idx, column_label in enumerate(column_labels):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(column_label)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(f"{gene_name}\n{y_label}" if col_idx == 0 else y_label)
            curve = curve_map.get(column_label)
            if curve is None:
                ax.set_xlim(0.0, row_display_max)
                ax.set_visible(False)
                continue
            color, linestyle = style_lookup[curve.source]
            _plot_curve(
                ax,
                curve,
                x_axis=x_axis,
                display_max=row_display_max,
                curve_mode=curve_mode,
                y_scale=y_scale,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
            )
            ax.set_xlim(0.0, row_display_max)
            ax.set_yscale(y_scale)
            ax.grid(alpha=0.18)
            if show_subplot_labels:
                ax.text(
                    0.98,
                    0.98,
                    f"{gene_name}\n{column_label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1.5,
                    },
                )
            if stats_fields:
                ax.text(
                    0.02,
                    0.02,
                    format_curve_stats(curve, fields=stats_fields),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "none",
                        "pad": 1.0,
                    },
                )
            if annotation_tables:
                lines = [
                    f"{table_name}: {mapping[(gene_name, column_label)]}"
                    for table_name, mapping in annotation_tables.items()
                    if (gene_name, column_label) in mapping
                ]
                if lines:
                    ax.text(
                        0.5,
                        -0.24,
                        "\n".join(lines),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                    )
    fig.tight_layout()
    return fig


def plot_batch_grid_figure(
    gene_name: str,
    curve_map: dict[tuple[str, str], "PriorCurve"],
    *,
    batches: list[str],
    perturbations: list[str],
    x_axis: str,
    mass_quantile: float,
    curve_mode: str = "density",
    y_scale: str = "linear",
    stats_fields: tuple[str, ...] = (),
    hide_empty: bool = False,
    show_axis_ticks: bool = True,
    panel_width: float = 2.2,
    panel_height: float = 1.9,
):
    plt = _get_pyplot()
    if not curve_map:
        raise ValueError("curve_map cannot be empty")
    if not batches:
        raise ValueError("batches cannot be empty")
    if not perturbations:
        raise ValueError("perturbations cannot be empty")
    row_display_max = _display_max_for_curves(
        list(curve_map.values()),
        x_axis=x_axis,
        mass_quantile=mass_quantile,
    )
    fig, axes = plt.subplots(
        len(batches),
        len(perturbations),
        figsize=(panel_width * len(perturbations), panel_height * len(batches)),
        squeeze=False,
    )
    for row_idx, batch in enumerate(batches):
        for col_idx, perturbation in enumerate(perturbations):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(perturbation, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(batch, fontsize=8)
            curve = curve_map.get((batch, perturbation))
            if curve is None:
                ax.set_xlim(0.0, row_display_max)
                if hide_empty:
                    ax.set_visible(False)
                    continue
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            _plot_curve(
                ax,
                curve,
                x_axis=x_axis,
                display_max=row_display_max,
                curve_mode=curve_mode,
                y_scale=y_scale,
                color="#1d4ed8",
                linestyle="-",
                linewidth=1.3,
            )
            ax.set_xlim(0.0, row_display_max)
            ax.set_yscale(y_scale)
            if show_axis_ticks:
                ax.tick_params(axis="both", labelsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if stats_fields:
                ax.text(
                    0.02,
                    0.02,
                    format_curve_stats(curve, fields=stats_fields),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "none",
                        "pad": 0.8,
                    },
                )
            ax.grid(alpha=0.16)
    fig.suptitle(gene_name, fontsize=12)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_batch_grid_figure",
    "plot_prior_facet_figure",
    "plot_prior_overlay_figure",
    "plt",
]
