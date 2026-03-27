from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from rich.console import Console
from rich.table import Table

from prism.model import ModelCheckpoint, PriorGrid, ScaleMetadata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()


def safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return list(value)


def checkpoint_gene_names(
    global_priors: PriorGrid | None,
    label_priors: dict[str, PriorGrid],
    fallback_gene_names: list[str],
) -> list[str]:
    if global_priors is not None:
        return list(global_priors.gene_names)
    if label_priors:
        return list(next(iter(label_priors.values())).gene_names)
    return list(fallback_gene_names)


def validate_shared_metadata(
    checkpoints: list[ModelCheckpoint], paths: list[Path]
) -> None:
    first = checkpoints[0]
    for path, checkpoint in zip(paths[1:], checkpoints[1:], strict=True):
        if checkpoint.fit_config != first.fit_config:
            raise ValueError(f"{path} has a different fit configuration")
        for key in (
            "source_h5ad_path",
            "layer",
            "reference_gene_names",
            "requested_fit_gene_names",
            "fit_mode",
            "label_key",
        ):
            if checkpoint.metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"{path} has different metadata for {key!r}")


def merge_prior_scope(
    priors_list: list[PriorGrid | None],
    scales: list[ScaleMetadata | None],
    *,
    requested_gene_names: list[str],
    allow_partial: bool,
    scope_name: str,
) -> tuple[PriorGrid | None, ScaleMetadata | None]:
    present = [
        (priors, scale)
        for priors, scale in zip(priors_list, scales, strict=True)
        if priors is not None
    ]
    if not present:
        return None, None
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first_priors, first_scale = present[0]
    assert first_priors is not None and first_scale is not None
    for priors, _scale in present:
        batched = priors.batched()
        if priors.S != first_priors.S:
            raise ValueError(f"{scope_name} has inconsistent S across checkpoints")
        for idx, gene_name in enumerate(batched.gene_names):
            if gene_name in rows:
                raise ValueError(
                    f"duplicate fitted gene across checkpoints in {scope_name}: {gene_name}"
                )
            rows[gene_name] = (
                np.asarray(batched.p_grid[idx], dtype=np.float64),
                np.asarray(batched.weights[idx], dtype=np.float64),
            )
    ordered_gene_names = (
        requested_gene_names if requested_gene_names else list(rows.keys())
    )
    missing = [name for name in ordered_gene_names if name not in rows]
    if missing and not allow_partial:
        raise ValueError(
            f"{scope_name} is missing {len(missing)} genes, e.g. {missing[:5]}"
        )
    merged_gene_names = [name for name in ordered_gene_names if name in rows]
    if not merged_gene_names:
        return None, None
    p_grid = np.stack([rows[name][0] for name in merged_gene_names], axis=0)
    weights = np.stack([rows[name][1] for name in merged_gene_names], axis=0)
    return PriorGrid(
        gene_names=list(merged_gene_names),
        p_grid=p_grid,
        weights=weights,
        S=float(first_priors.S),
    ), first_scale


def print_merge_summary(
    output_path: Path, checkpoints: list[ModelCheckpoint], merged: ModelCheckpoint
) -> None:
    table = Table(title="Merged Checkpoint")
    table.add_column("Input genes", justify="right")
    table.add_column("Merged genes", justify="right")
    table.add_column("Label priors", justify="right")
    table.add_row(
        str(sum(len(checkpoint.gene_names) for checkpoint in checkpoints)),
        str(len(merged.gene_names)),
        str(len(merged.label_priors)),
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")


def resolve_x_axis(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"mu", "p"}:
        raise ValueError("x_axis must be either 'mu' or 'p'")
    return resolved


def resolve_plot_curve_sets(
    checkpoint: ModelCheckpoint,
    *,
    gene_names: list[str],
    labels: list[str] | None,
    include_global: bool,
) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    requested = list(dict.fromkeys(gene_names))
    if not requested:
        raise ValueError("at least one --gene is required")
    selected_labels = (
        sorted(checkpoint.label_priors) if not labels else list(dict.fromkeys(labels))
    )
    unknown = [
        label for label in selected_labels if label not in checkpoint.label_priors
    ]
    if unknown:
        raise ValueError(f"unknown label priors: {unknown}")
    curve_sets: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for gene_name in requested:
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        if (
            include_global
            and checkpoint.priors is not None
            and gene_name in checkpoint.priors.gene_names
        ):
            prior = checkpoint.priors.subset(gene_name)
            curves.append(
                (
                    "global",
                    np.asarray(prior.mu_grid, dtype=np.float64).reshape(-1),
                    np.asarray(prior.weights, dtype=np.float64).reshape(-1),
                )
            )
        for label in selected_labels:
            priors = checkpoint.label_priors[label]
            if gene_name not in priors.gene_names:
                continue
            prior = priors.subset(gene_name)
            curves.append(
                (
                    f"label:{label}",
                    np.asarray(prior.mu_grid, dtype=np.float64).reshape(-1),
                    np.asarray(prior.weights, dtype=np.float64).reshape(-1),
                )
            )
        if not curves:
            raise ValueError(
                f"gene {gene_name!r} is not present in the selected prior sources"
            )
        curve_sets[gene_name] = curves
    return curve_sets


def display_cutoff(grid: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    grid_np = np.asarray(grid, dtype=np.float64).reshape(-1)
    weight_np = np.asarray(weights, dtype=np.float64).reshape(-1)
    if grid_np.size == 0:
        return 1.0
    mass = np.maximum(np.sum(weight_np), 1e-12)
    cdf = np.cumsum(weight_np / mass)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), grid_np.size - 1)
    return float(grid_np[idx])


def mu_to_p(mu_grid: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_grid, dtype=np.float64)
    max_mu = float(np.max(mu))
    return mu / max(max_mu, 1e-12)


def plot_fg_figure(
    curve_sets: dict[str, list[tuple[str, np.ndarray, np.ndarray]]],
    *,
    x_axis: str,
    mass_quantile: float,
):
    genes = list(curve_sets)
    n_genes = len(genes)
    cols = min(2, n_genes)
    rows = (n_genes + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(8.2 * cols, 4.8 * rows), squeeze=False
    )
    color_cycle = ["#1d4ed8", "#0f766e", "#c2410c", "#7c3aed", "#be123c", "#0891b2"]
    for ax in axes.ravel():
        ax.set_visible(False)
    for panel_idx, gene_name in enumerate(genes):
        ax = axes.ravel()[panel_idx]
        ax.set_visible(True)
        curves = curve_sets[gene_name]
        x_sets = (
            [(label, mu_to_p(mu), weights) for label, mu, weights in curves]
            if x_axis == "p"
            else curves
        )
        display_max = max(
            display_cutoff(x, weights, mass_quantile) for _, x, weights in x_sets
        )
        display_max = max(display_max, 1e-12)
        for curve_idx, (label, x_values, weights) in enumerate(x_sets):
            mask = x_values <= display_max + 1e-12
            if not np.any(mask):
                mask = np.ones_like(x_values, dtype=bool)
            ax.plot(
                x_values[mask],
                weights[mask],
                lw=2.0,
                color=color_cycle[curve_idx % len(color_cycle)],
                label=label,
            )
        ax.set_title(gene_name)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("prior mass")
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig
