#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()
install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot batch x perturbation prior grids for multiple genes."
    )
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gene-list", required=True, type=Path)
    parser.add_argument("--label-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--x-axis", choices=("mu", "p"), default="mu")
    parser.add_argument("--mass-quantile", type=float, default=0.995)
    parser.add_argument(
        "--image-format",
        choices=("svg", "pdf", "eps"),
        default="svg",
        help="Vector image format for per-gene figures.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def display_cutoff(grid: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mass = max(float(np.sum(weights)), 1e-12)
    cdf = np.cumsum(weights / mass)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), grid.size - 1)
    return float(grid[idx])


def mu_to_p(mu_grid: np.ndarray) -> np.ndarray:
    return mu_grid / max(float(np.max(mu_grid)), 1e-12)


def parse_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    batches: list[str] = []
    perturbations: list[str] = []
    for value in labels:
        if not value.startswith("batch") or "_" not in value:
            raise ValueError(f"invalid batch_x_perturbation label: {value}")
        batch, perturbation = value.split("_", 1)
        if batch not in batches:
            batches.append(batch)
        if perturbation not in perturbations:
            perturbations.append(perturbation)
    return batches, perturbations


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    gene_list_path = args.gene_list.expanduser().resolve()
    label_list_path = args.label_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    genes = read_list(gene_list_path)
    if args.top_n is not None:
        genes = genes[: args.top_n]
    labels = read_list(label_list_path)
    batches, perturbations = parse_labels(labels)

    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint.label_priors:
        raise ValueError("checkpoint has no label priors")

    intro = Table(show_header=False, box=None)
    intro.add_row("Checkpoint", str(checkpoint_path))
    intro.add_row("Genes", str(len(genes)))
    intro.add_row("Batches", ", ".join(batches))
    intro.add_row("Perturbations", str(len(perturbations)))
    intro.add_row("Output dir", str(output_dir))
    console.print(Panel(intro, title="Batch Grid Plot", border_style="cyan"))

    output_dir.mkdir(parents=True, exist_ok=True)

    for gene in genes:
        curve_map: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for batch in batches:
            for perturbation in perturbations:
                label = f"{batch}_{perturbation}"
                if label not in checkpoint.label_priors:
                    continue
                priors = checkpoint.label_priors[label]
                if gene not in priors.gene_names:
                    continue
                prior = priors.subset(gene)
                mu = np.asarray(prior.mu_grid, dtype=np.float64).reshape(-1)
                weights = np.asarray(prior.weights, dtype=np.float64).reshape(-1)
                curve_map[(batch, perturbation)] = (mu, weights)
        if not curve_map:
            console.print(f"[yellow]Skipped[/yellow] {gene}: no matching label priors")
            continue

        row_display_max = max(
            display_cutoff(
                mu_to_p(mu) if args.x_axis == "p" else mu, weights, args.mass_quantile
            )
            for mu, weights in curve_map.values()
        )
        row_display_max = max(row_display_max, 1e-12)

        fig, axes = plt.subplots(
            len(batches),
            len(perturbations),
            figsize=(1.8 * len(perturbations), 1.6 * len(batches)),
            squeeze=False,
        )
        for row_idx, batch in enumerate(batches):
            for col_idx, perturbation in enumerate(perturbations):
                ax = axes[row_idx][col_idx]
                if row_idx == 0:
                    ax.set_title(perturbation, fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(batch, fontsize=8)
                key = (batch, perturbation)
                if key not in curve_map:
                    ax.set_xlim(0.0, row_display_max)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                mu, weights = curve_map[key]
                x = mu_to_p(mu) if args.x_axis == "p" else mu
                mask = x <= row_display_max + 1e-12
                if not np.any(mask):
                    mask = np.ones_like(x, dtype=bool)
                ax.plot(x[mask], weights[mask], lw=1.3, color="#1d4ed8")
                ax.set_xlim(0.0, row_display_max)
                ax.tick_params(axis="both", labelsize=6)
                ax.grid(alpha=0.16)
        fig.suptitle(gene, fontsize=12)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{gene}.{args.image_format}",
            dpi=args.dpi,
            bbox_inches="tight",
            format=args.image_format,
        )
        plt.close(fig)
        console.print(f"[green]Saved[/green] {gene}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(
                str(exc),
                title="plot_batch_perturbation_grid failed",
                border_style="red",
            )
        )
        raise SystemExit(1) from exc
