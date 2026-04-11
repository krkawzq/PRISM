from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import typer

from prism.cli.common import (
    console,
    print_elapsed,
    print_saved_path,
    resolve_float,
    resolve_int,
    resolve_optional_str,
    resolve_str,
)
from prism.model import PriorGrid, load_checkpoint
from prism.plotting import plt

from .common import (
    normalize_label_summary_metric,
    resolve_label_names,
    resolve_optional_list,
)


def _select_gene_names(
    checkpoint_labels: list[str],
    priors: dict[str, PriorGrid],
    *,
    gene_names: list[str] | None,
    max_genes: int,
) -> list[str]:
    common_genes: set[str] | None = None
    for label in checkpoint_labels:
        label_genes = set(priors[label].gene_names)
        common_genes = (
            label_genes if common_genes is None else common_genes & label_genes
        )
    if common_genes is None or not common_genes:
        raise ValueError("no common genes found across label priors")
    selected = (
        [gene for gene in gene_names if gene in common_genes]
        if gene_names is not None
        else sorted(common_genes)
    )
    if not selected:
        raise ValueError("no genes to plot after filtering")
    return selected[:max_genes]


def _select_probabilities(
    prior: PriorGrid, gene_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    selected = prior.select_genes(gene_names).as_gene_specific()
    support = np.asarray(selected.support, dtype=np.float64)
    probabilities = np.asarray(selected.prior_probabilities, dtype=np.float64)
    if support.ndim == 1:
        support = support[None, :]
        probabilities = probabilities[None, :]
    return support, probabilities


def _align_probability_rows(
    support_a: np.ndarray,
    probabilities_a: np.ndarray,
    support_b: np.ndarray,
    probabilities_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    union_support = np.union1d(support_a, support_b)
    aligned_a = np.zeros(union_support.shape[0], dtype=np.float64)
    aligned_b = np.zeros(union_support.shape[0], dtype=np.float64)
    aligned_a[np.searchsorted(union_support, support_a)] = probabilities_a
    aligned_b[np.searchsorted(union_support, support_b)] = probabilities_b
    return aligned_a, aligned_b


def _mean_jsd(
    support_a: np.ndarray,
    probabilities_a: np.ndarray,
    support_b: np.ndarray,
    probabilities_b: np.ndarray,
) -> float:
    eps = 1e-12
    scores: list[float] = []
    for support_row_a, probability_row_a, support_row_b, probability_row_b in zip(
        support_a,
        probabilities_a,
        support_b,
        probabilities_b,
        strict=True,
    ):
        aligned_a, aligned_b = _align_probability_rows(
            support_row_a, probability_row_a, support_row_b, probability_row_b
        )
        midpoint = 0.5 * (aligned_a + aligned_b)
        kl_a = np.sum(
            aligned_a * np.log(np.clip(aligned_a, eps, None) / np.clip(midpoint, eps, None))
        )
        kl_b = np.sum(
            aligned_b * np.log(np.clip(aligned_b, eps, None) / np.clip(midpoint, eps, None))
        )
        scores.append(float(0.5 * (kl_a + kl_b)))
    return float(np.mean(scores))


def _mean_overlap(
    support_a: np.ndarray,
    probabilities_a: np.ndarray,
    support_b: np.ndarray,
    probabilities_b: np.ndarray,
) -> float:
    scores: list[float] = []
    for support_row_a, probability_row_a, support_row_b, probability_row_b in zip(
        support_a,
        probabilities_a,
        support_b,
        probabilities_b,
        strict=True,
    ):
        aligned_a, aligned_b = _align_probability_rows(
            support_row_a, probability_row_a, support_row_b, probability_row_b
        )
        scores.append(float(np.sum(np.minimum(aligned_a, aligned_b))))
    return float(np.mean(scores))


def plot_label_summary_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path with label priors."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    gene_names: list[str] | None = typer.Option(
        None,
        "--gene",
        help="Repeatable gene names to include. Defaults to all common genes.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable label names to include. Defaults to all label priors.",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional file listing labels to include.",
    ),
    max_genes: int = typer.Option(
        50, min=1, help="Maximum number of genes to include."
    ),
    metric: str = typer.Option("jsd", help="Similarity metric: jsd or overlap."),
    figsize_w: float = typer.Option(10.0, help="Figure width in inches."),
    figsize_h: float = typer.Option(10.0, help="Figure height in inches."),
    palette: str | None = typer.Option(None, help="Optional matplotlib colormap name."),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    labels_path = None if labels_path is None else labels_path.expanduser().resolve()
    max_genes = resolve_int(max_genes)
    metric = normalize_label_summary_metric(resolve_str(metric))
    figsize_w = resolve_float(figsize_w)
    figsize_h = resolve_float(figsize_h)
    palette = resolve_optional_str(palette)

    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint.has_label_priors:
        raise ValueError("checkpoint has no label priors")
    resolved_labels = resolve_label_names(
        labels=resolve_optional_list(labels),
        labels_path=labels_path,
        default=list(checkpoint.available_labels),
    )
    assert resolved_labels is not None
    if len(resolved_labels) < 2:
        raise ValueError("label-summary requires at least two label priors")
    selected_genes = _select_gene_names(
        resolved_labels,
        checkpoint.label_priors,
        gene_names=resolve_optional_list(gene_names),
        max_genes=max_genes,
    )

    similarity = np.zeros(
        (len(resolved_labels), len(resolved_labels)), dtype=np.float64
    )
    for row_idx, label_a in enumerate(resolved_labels):
        support_a, probabilities_a = _select_probabilities(
            checkpoint.label_priors[label_a],
            selected_genes,
        )
        for col_idx, label_b in enumerate(resolved_labels):
            if row_idx == col_idx:
                similarity[row_idx, col_idx] = 0.0 if metric == "jsd" else 1.0
                continue
            support_b, probabilities_b = _select_probabilities(
                checkpoint.label_priors[label_b],
                selected_genes,
            )
            similarity[row_idx, col_idx] = (
                _mean_jsd(support_a, probabilities_a, support_b, probabilities_b)
                if metric == "jsd"
                else _mean_overlap(support_a, probabilities_a, support_b, probabilities_b)
            )

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    cmap_name = palette or ("viridis_r" if metric == "jsd" else "viridis")
    image = ax.imshow(similarity, cmap=cmap_name, aspect="auto")
    ax.set_xticks(range(len(resolved_labels)))
    ax.set_yticks(range(len(resolved_labels)))
    ax.set_xticklabels(resolved_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(resolved_labels, fontsize=8)
    ax.set_title(f"Label {metric.upper()} ({len(selected_genes)} genes)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_saved_path(console, output_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


__all__ = ["plot_label_summary_command"]
