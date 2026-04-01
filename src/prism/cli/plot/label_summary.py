"""Plot label-summary: label similarity heatmap and per-label prior summary from a checkpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prism.cli.common import print_elapsed, print_saved_path
from prism.model import load_checkpoint
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution

from rich.console import Console

console = Console()


def plot_label_summary_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path with label priors."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output figure path (svg/png/pdf)."
    ),
    gene_names: list[str] | None = typer.Option(
        None, "--gene", help="Repeatable gene names to include. Defaults to all."
    ),
    max_genes: int = typer.Option(
        50, min=1, help="Maximum number of genes to include."
    ),
    metric: str = typer.Option("jsd", help="Similarity metric: jsd or overlap."),
    figsize_w: float = typer.Option(10.0, help="Figure width in inches."),
    figsize_h: float = typer.Option(10.0, help="Figure height in inches."),
    palette: str | None = typer.Option(None, help="Optional matplotlib colormap name."),
) -> int:
    from time import perf_counter

    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    resolve_cli_checkpoint_distribution(
        checkpoint,
        command_name="prism plot label-summary",
        require_label_priors=True,
        require_grid_domains={"p"},
    )
    if not checkpoint.label_priors:
        raise ValueError("checkpoint does not contain label-specific priors")

    labels = sorted(checkpoint.label_priors)
    n_labels = len(labels)

    common_genes: set[str] | None = None
    for label in labels:
        prior = checkpoint.label_priors[label]
        names_set = set(prior.gene_names)
        common_genes = names_set if common_genes is None else common_genes & names_set

    if common_genes is None or not common_genes:
        raise ValueError("no common genes found across label priors")

    if gene_names:
        selected = [g for g in gene_names if g in common_genes]
    else:
        selected = sorted(common_genes)

    if len(selected) > max_genes:
        selected = selected[:max_genes]

    if not selected:
        raise ValueError("no genes to plot after filtering")

    similarity = np.zeros((n_labels, n_labels), dtype=np.float64)
    for i, label_i in enumerate(labels):
        prior_i = checkpoint.label_priors[label_i].subset(selected).batched()
        w_i = np.asarray(prior_i.weights, dtype=np.float64)
        for j, label_j in enumerate(labels):
            if i == j:
                similarity[i, j] = 1.0 if metric == "overlap" else 0.0
                continue
            prior_j = checkpoint.label_priors[label_j].subset(selected).batched()
            w_j = np.asarray(prior_j.weights, dtype=np.float64)
            if metric == "jsd":
                similarity[i, j] = float(_mean_jsd(w_i, w_j))
            else:
                similarity[i, j] = float(_mean_overlap(w_i, w_j))

    cmap_name = palette or ("viridis_r" if metric == "jsd" else "viridis")

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    im = ax.imshow(similarity, cmap=cmap_name, aspect="auto")
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Label {metric.upper()} ({len(selected)} genes)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print_saved_path(console, output_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


def _mean_jsd(w1: np.ndarray, w2: np.ndarray) -> float:
    eps = 1e-12
    m = 0.5 * (w1 + w2)
    kl1 = np.sum(w1 * np.log(np.clip(w1, eps, None) / np.clip(m, eps, None)), axis=-1)
    kl2 = np.sum(w2 * np.log(np.clip(w2, eps, None) / np.clip(m, eps, None)), axis=-1)
    jsd = 0.5 * (kl1 + kl2)
    return float(np.mean(jsd))


def _mean_overlap(w1: np.ndarray, w2: np.ndarray) -> float:
    overlap = np.sum(np.minimum(w1, w2), axis=-1)
    return float(np.mean(overlap))


__all__ = ["plot_label_summary_command"]
