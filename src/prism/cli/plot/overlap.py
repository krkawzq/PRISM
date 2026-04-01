from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from prism.cli.common import ensure_mutually_exclusive
from prism.io import read_gene_list, read_string_list
from prism.model import load_checkpoint
from prism.plotting import (
    compute_overlap_dataframe,
    SUPPORTED_OVERLAP_METRICS,
    plot_overlap_heatmap_figure,
    plt,
    resolve_overlap_metric,
)
from .common import option_sequence, option_value, resolve_order_mode

console = Console()


def _resolve_gene_names(
    *,
    checkpoint,
    gene_names: list[str] | None,
    genes_path: Path | None,
) -> list[str]:
    gene_names = option_sequence(gene_names)
    genes_path = option_value(genes_path)
    ensure_mutually_exclusive(("--gene", gene_names), ("--genes", genes_path))
    if genes_path is not None:
        return list(dict.fromkeys(read_gene_list(genes_path.expanduser().resolve())))
    if not gene_names:
        return list(checkpoint.gene_names)
    return list(dict.fromkeys(gene_names))


def _resolve_labels(
    *,
    checkpoint,
    control_label: str,
    labels: list[str] | None,
    labels_path: Path | None,
) -> list[str]:
    control_label = str(option_value(control_label))
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    ensure_mutually_exclusive(("--label", labels), ("--labels", labels_path))
    default = sorted(
        label for label in checkpoint.label_priors.keys() if label != control_label
    )
    if labels_path is not None:
        return list(dict.fromkeys(read_string_list(labels_path.expanduser().resolve())))
    if not labels:
        return default
    return list(dict.fromkeys(labels))


def _metric_sort_key(values: pd.Series, *, metric: str) -> pd.Series:
    if metric == "overlap":
        return -values
    if metric == "best_scale":
        return (values - 1.0).abs()
    return values


def _sort_order(
    df: pd.DataFrame,
    *,
    axis: str,
    order_mode: str,
    metric: str,
    fallback_order: list[str],
) -> list[str]:
    if order_mode == "input":
        return [value for value in fallback_order if value in set(df[axis])]
    if order_mode == "alpha":
        return sorted(set(df[axis].astype(str).tolist()))
    grouped = df.groupby(axis, sort=False)[metric].mean()
    sort_key = _metric_sort_key(grouped, metric=metric)
    return sort_key.sort_values().index.astype(str).tolist()


def plot_overlap_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output heatmap path."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV path for exported overlap metrics.",
    ),
    control_label: str = typer.Option(..., "--control-label", help="Reference control label."),
    gene_names: list[str] = typer.Option(
        None,
        "--gene",
        help="Optional repeatable genes. Defaults to all checkpoint genes.",
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON gene list.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Optional top-k genes selected by the chosen metric.",
    ),
    labels: list[str] = typer.Option(
        None,
        "--label",
        help="Optional repeatable perturbation labels.",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON file listing labels.",
    ),
    metric: str = typer.Option(
        "overlap",
        "--metric",
        help="Metric: " + ", ".join(SUPPORTED_OVERLAP_METRICS) + ".",
    ),
    gene_order: str = typer.Option(
        "metric",
        "--gene-order",
        help="Gene ordering: input, alpha, or metric.",
    ),
    label_order: str = typer.Option(
        "input",
        "--label-order",
        help="Label ordering: input, alpha, or metric.",
    ),
    scale_min: float = typer.Option(0.25, min=1e-12, help="Minimum scale factor."),
    scale_max: float = typer.Option(4.0, min=1e-12, help="Maximum scale factor."),
    scale_grid_size: int = typer.Option(
        201, min=3, help="Number of log-scale factors to search."
    ),
    interp_points: int = typer.Option(
        2048, min=128, help="Interpolation points for overlap metrics."
    ),
    annotate_cells: bool = typer.Option(
        False,
        "--annotate-cells/--no-annotate-cells",
        help="Render numeric values inside heatmap cells.",
    ),
    cmap: str | None = typer.Option(None, "--cmap", help="Optional matplotlib colormap."),
    panel_width: float = typer.Option(0.45, min=0.1, help="Heatmap width per label."),
    panel_height: float = typer.Option(0.35, min=0.1, help="Heatmap height per gene."),
) -> int:
    checkpoint_path = option_value(checkpoint_path).expanduser().resolve()
    output_path = option_value(output_path)
    output_csv_path = option_value(output_csv_path)
    control_label = str(option_value(control_label))
    gene_names = option_sequence(gene_names)
    genes_path = option_value(genes_path)
    top_k = option_value(top_k)
    labels = option_sequence(labels)
    labels_path = option_value(labels_path)
    metric = str(option_value(metric))
    scale_min = float(option_value(scale_min))
    scale_max = float(option_value(scale_max))
    scale_grid_size = int(option_value(scale_grid_size))
    interp_points = int(option_value(interp_points))
    annotate_cells = bool(option_value(annotate_cells))
    cmap = option_value(cmap)
    panel_width = float(option_value(panel_width))
    panel_height = float(option_value(panel_height))

    resolved_metric = resolve_overlap_metric(metric)
    resolved_gene_order_mode = resolve_order_mode(gene_order, name="--gene-order")
    resolved_label_order_mode = resolve_order_mode(label_order, name="--label-order")
    checkpoint = load_checkpoint(checkpoint_path)
    resolved_genes = _resolve_gene_names(
        checkpoint=checkpoint,
        gene_names=gene_names,
        genes_path=genes_path,
    )
    resolved_labels = _resolve_labels(
        checkpoint=checkpoint,
        control_label=control_label,
        labels=labels,
        labels_path=labels_path,
    )
    control_priors = checkpoint.label_priors.get(control_label)
    if control_priors is None:
        raise ValueError(f"control label {control_label!r} not found in checkpoint label priors")
    selected_genes = [
        gene for gene in resolved_genes if gene in control_priors.gene_names
    ]
    if not selected_genes:
        raise ValueError("no selected genes remain after intersecting with control priors")
    df = compute_overlap_dataframe(
        checkpoint,
        control_label=control_label,
        gene_names=selected_genes,
        labels=resolved_labels,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_grid_size=scale_grid_size,
        interp_points=interp_points,
    )
    if top_k is not None:
        grouped = df.groupby("gene", sort=False)[resolved_metric].mean()
        sort_key = _metric_sort_key(grouped, metric=resolved_metric)
        selected_top = sort_key.sort_values().index.astype(str).tolist()[:top_k]
        df = df[df["gene"].isin(selected_top)].copy()
        selected_genes = [gene for gene in selected_genes if gene in set(selected_top)]
    resolved_gene_order = _sort_order(
        df,
        axis="gene",
        order_mode=resolved_gene_order_mode,
        metric=resolved_metric,
        fallback_order=selected_genes,
    )
    resolved_label_order = _sort_order(
        df,
        axis="label",
        order_mode=resolved_label_order_mode,
        metric=resolved_metric,
        fallback_order=resolved_labels,
    )
    fig = plot_overlap_heatmap_figure(
        df,
        metric=resolved_metric,
        gene_order=resolved_gene_order,
        label_order=resolved_label_order,
        annotate_cells=annotate_cells,
        panel_width=panel_width,
        panel_height=panel_height,
        cmap=cmap,
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    if output_csv_path is not None:
        output_csv_path = output_csv_path.expanduser().resolve()
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        console.print(f"[bold green]Saved[/bold green] {output_csv_path}")
    return 0


__all__ = ["plot_overlap_command"]
