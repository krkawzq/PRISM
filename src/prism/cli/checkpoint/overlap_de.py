from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from prism.model import load_checkpoint

from .common import console, load_gene_list_file


def _resolve_name_list(
    explicit: list[str] | None, file_path: Path | None, *, default: list[str]
) -> list[str]:
    values: list[str] = []
    if explicit:
        values.extend(explicit)
    if file_path is not None:
        values.extend(load_gene_list_file(file_path.expanduser().resolve()))
    resolved = list(dict.fromkeys(values))
    return default if not resolved else resolved


def _prepare_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid_min: float,
    grid_max: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.linspace(grid_min, grid_max, n_points, dtype=np.float64)
    values = np.interp(support, x, y, left=0.0, right=0.0)
    values = np.clip(values, 0.0, None)
    area = float(np.trapezoid(values, support))
    if area <= 0:
        raise ValueError("density area must be positive")
    values /= area
    return support, values


def _distribution_metrics(
    support: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float, float]:
    dx = float(support[1] - support[0]) if support.size > 1 else 1.0
    overlap = float(np.sum(np.minimum(p, q)) * dx)
    overlap = min(max(overlap, 0.0), 1.0)
    p_mass = p / max(float(np.sum(p)), 1e-12)
    q_mass = q / max(float(np.sum(q)), 1e-12)
    m_mass = 0.5 * (p_mass + q_mass)
    eps = 1e-12
    jsd = 0.5 * float(np.sum(p_mass * np.log((p_mass + eps) / (m_mass + eps))))
    jsd += 0.5 * float(np.sum(q_mass * np.log((q_mass + eps) / (m_mass + eps))))
    cdf_diff = np.cumsum(p_mass - q_mass)
    wasserstein = float(np.sum(np.abs(cdf_diff)) * dx)
    return overlap, jsd, wasserstein


def _best_scaled_metrics(
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    pert_x: np.ndarray,
    pert_y: np.ndarray,
    *,
    scale_min: float,
    scale_max: float,
    scale_grid_size: int,
    interp_points: int,
) -> tuple[float, float, float, float]:
    best_scale = 1.0
    best_overlap = -1.0
    best_jsd = float("inf")
    best_wasserstein = float("inf")
    scale_grid = np.exp(
        np.linspace(
            np.log(scale_min), np.log(scale_max), scale_grid_size, dtype=np.float64
        )
    )
    for scale in scale_grid:
        scaled_x = pert_x * scale
        grid_min = float(min(np.min(ctrl_x), np.min(scaled_x)))
        grid_max = float(max(np.max(ctrl_x), np.max(scaled_x)))
        support, ctrl_density = _prepare_density(
            ctrl_x,
            ctrl_y,
            grid_min=grid_min,
            grid_max=grid_max,
            n_points=interp_points,
        )
        _, pert_density = _prepare_density(
            scaled_x,
            pert_y,
            grid_min=grid_min,
            grid_max=grid_max,
            n_points=interp_points,
        )
        overlap, jsd, wasserstein = _distribution_metrics(
            support,
            ctrl_density,
            pert_density,
        )
        if overlap > best_overlap:
            best_scale = float(scale)
            best_overlap = overlap
            best_jsd = jsd
            best_wasserstein = wasserstein
    return best_scale, best_overlap, best_jsd, best_wasserstein


def overlap_de_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    output_csv_path: Path = typer.Option(..., "--output-csv", help="Output CSV path."),
    control_label: str = typer.Option(..., help="Control label used as reference."),
    gene_names: list[str] | None = typer.Option(
        None,
        "--gene",
        help="Optional repeatable genes. Defaults to all checkpoint genes.",
    ),
    gene_list_path: Path | None = typer.Option(
        None,
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON gene list.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable perturbation labels. Defaults to all non-control labels.",
    ),
    label_list_path: Path | None = typer.Option(
        None,
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text file with one perturbation label per line.",
    ),
    scale_min: float = typer.Option(0.25, min=1e-12, help="Minimum scale factor."),
    scale_max: float = typer.Option(4.0, min=1e-12, help="Maximum scale factor."),
    scale_grid_size: int = typer.Option(
        201, min=3, help="Number of log-scale factors to search."
    ),
    interp_points: int = typer.Option(
        2048, min=128, help="Interpolation points for overlap metrics."
    ),
) -> int:
    if scale_min > scale_max:
        raise ValueError("--scale-min must be <= --scale-max")
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    if not checkpoint.label_priors:
        raise ValueError("checkpoint has no label priors")
    if control_label not in checkpoint.label_priors:
        raise ValueError(
            f"control label {control_label!r} not found in checkpoint label priors"
        )

    selected_genes = _resolve_name_list(
        gene_names,
        gene_list_path,
        default=list(checkpoint.gene_names),
    )
    available_labels = sorted(
        label for label in checkpoint.label_priors if label != control_label
    )
    selected_labels = _resolve_name_list(
        labels, label_list_path, default=available_labels
    )
    unknown_labels = [
        label for label in selected_labels if label not in checkpoint.label_priors
    ]
    if unknown_labels:
        raise ValueError(f"unknown labels: {unknown_labels[:10]}")

    control_priors = checkpoint.label_priors[control_label]
    missing_genes = [
        gene for gene in selected_genes if gene not in control_priors.gene_names
    ]
    if missing_genes:
        console.print(
            f"[yellow]Skipped[/yellow] {len(missing_genes)} genes missing from control priors"
        )
    selected_genes = [
        gene for gene in selected_genes if gene in control_priors.gene_names
    ]
    if not selected_genes:
        raise ValueError(
            "no selected genes remain after intersecting with control priors"
        )

    rows: list[dict[str, object]] = []
    for label in selected_labels:
        if label == control_label:
            continue
        priors = checkpoint.label_priors[label]
        common_genes = [gene for gene in selected_genes if gene in priors.gene_names]
        for gene in common_genes:
            ctrl_prior = control_priors.subset(gene)
            pert_prior = priors.subset(gene)
            ctrl_x = np.asarray(ctrl_prior.mu_grid, dtype=np.float64).reshape(-1)
            ctrl_y = np.asarray(ctrl_prior.weights, dtype=np.float64).reshape(-1)
            pert_x = np.asarray(pert_prior.mu_grid, dtype=np.float64).reshape(-1)
            pert_y = np.asarray(pert_prior.weights, dtype=np.float64).reshape(-1)
            best_scale, overlap, jsd, wasserstein = _best_scaled_metrics(
                ctrl_x,
                ctrl_y,
                pert_x,
                pert_y,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_grid_size=scale_grid_size,
                interp_points=interp_points,
            )
            rows.append(
                {
                    "gene": gene,
                    "label": label,
                    "overlap": overlap,
                    "jsd": jsd,
                    "wasserstein": wasserstein,
                    "best_scale": best_scale,
                }
            )

    if not rows:
        raise ValueError("no overlap-DE rows were produced")
    df = pd.DataFrame(rows).sort_values(["label", "gene"]).reset_index(drop=True)
    output_csv_path = output_csv_path.expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    table = Table(title="Overlap-DE")
    table.add_column("Genes", justify="right")
    table.add_column("Labels", justify="right")
    table.add_column("Output")
    table.add_row(
        str(df["gene"].nunique()), str(df["label"].nunique()), str(output_csv_path)
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_csv_path}")
    return 0


__all__ = ["overlap_de_command"]
