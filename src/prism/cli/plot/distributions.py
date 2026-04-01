"""Plot distributions from extracted h5ad layers (signal, entropy, MI, etc.)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prism.cli.common import print_elapsed, print_saved_path
from prism.cli.plot.common import resolve_order_mode  # noqa: F401

from rich.console import Console

console = Console()


def plot_distributions_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Extracted h5ad file with signal layers."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output figure path (svg/png/pdf)."
    ),
    layers: list[str] | None = typer.Option(
        None,
        "--layer",
        help="Repeatable layer names to plot. Defaults to signal, posterior_entropy, mutual_information.",
    ),
    group_key: str | None = typer.Option(
        None,
        "--group-key",
        help="Optional obs column to group cells for per-group distributions.",
    ),
    plot_type: str = typer.Option(
        "violin",
        help="Plot type: violin, box, or hist.",
    ),
    max_genes: int = typer.Option(
        20, min=1, help="Maximum number of genes to include in the figure."
    ),
    figsize_w: float = typer.Option(12.0, help="Figure width in inches."),
    figsize_h: float = typer.Option(4.0, help="Figure height per layer in inches."),
    seed: int = typer.Option(0, min=0, help="Random seed for gene sampling."),
    palette: str | None = typer.Option(
        None, help="Optional matplotlib colormap name for group coloring."
    ),
) -> int:
    from time import perf_counter

    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    adata = ad.read_h5ad(h5ad_path)
    default_layers = ["signal", "posterior_entropy", "mutual_information"]
    selected_layers = (
        list(layers)
        if layers
        else [name for name in default_layers if name in adata.layers]
    )
    if not selected_layers:
        raise ValueError(
            f"no matching layers found. Available: {list(adata.layers.keys())}"
        )
    missing = [name for name in selected_layers if name not in adata.layers]
    if missing:
        raise ValueError(f"layers not found in h5ad: {missing}")

    gene_names = [str(name) for name in adata.var_names.tolist()]
    if len(gene_names) > max_genes:
        rng = np.random.default_rng(seed)
        chosen_idx = np.sort(rng.choice(len(gene_names), size=max_genes, replace=False))
        gene_names = [gene_names[i] for i in chosen_idx]
        gene_positions = chosen_idx.tolist()
    else:
        gene_positions = list(range(len(gene_names)))

    plot_type_resolved = plot_type.strip().lower()
    if plot_type_resolved not in {"violin", "box", "hist"}:
        raise ValueError("plot_type must be one of: violin, box, hist")

    n_layers = len(selected_layers)
    fig, axes = plt.subplots(
        n_layers,
        1,
        figsize=(figsize_w, figsize_h * n_layers),
        squeeze=False,
    )

    groups = None
    group_labels = None
    if group_key is not None:
        if group_key not in adata.obs.columns:
            raise KeyError(f"obs column {group_key!r} does not exist")
        group_labels_arr = np.asarray(adata.obs[group_key].astype(str)).reshape(-1)
        unique_labels = sorted(np.unique(group_labels_arr).tolist())
        groups = {
            label: np.flatnonzero(group_labels_arr == label) for label in unique_labels
        }
        group_labels = unique_labels

    cmap = None
    if palette is not None:
        cmap = plt.get_cmap(palette)

    for layer_idx, layer_name in enumerate(selected_layers):
        ax = axes[layer_idx, 0]
        layer_data = np.asarray(adata.layers[layer_name], dtype=np.float64)

        if groups is not None and group_labels is not None:
            _plot_grouped(
                ax,
                layer_data,
                gene_names,
                gene_positions,
                groups,
                group_labels,
                plot_type_resolved,
                cmap,
            )
        else:
            _plot_ungrouped(
                ax,
                layer_data,
                gene_names,
                gene_positions,
                plot_type_resolved,
            )

        ax.set_title(layer_name)
        if layer_idx == n_layers - 1:
            ax.set_xlabel("Gene")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print_saved_path(console, output_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


def _plot_ungrouped(
    ax,
    layer_data: np.ndarray,
    gene_names: list[str],
    gene_positions: list[int],
    plot_type: str,
) -> None:
    data = [layer_data[:, pos] for pos in gene_positions]
    data = [np.asarray(d[np.isfinite(d)], dtype=np.float64) for d in data]
    positions = list(range(len(gene_names)))

    if plot_type == "violin":
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for pc in parts.get("bodies", []):
            pc.set_alpha(0.7)
    elif plot_type == "box":
        ax.boxplot(data, positions=positions, widths=0.6)
    else:
        for i, d in enumerate(data):
            ax.hist(d, bins=30, alpha=0.5, label=gene_names[i])
        ax.legend(fontsize=6, ncol=2)
        return

    ax.set_xticks(positions)
    ax.set_xticklabels(gene_names, rotation=45, ha="right", fontsize=7)


def _plot_grouped(
    ax,
    layer_data: np.ndarray,
    gene_names: list[str],
    gene_positions: list[int],
    groups: dict[str, np.ndarray],
    group_labels: list[str],
    plot_type: str,
    cmap,
) -> None:
    n_groups = len(group_labels)
    n_genes = len(gene_names)
    width = 0.8 / max(n_groups, 1)
    colors = None
    if cmap is not None:
        colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]

    for g_idx, label in enumerate(group_labels):
        cell_indices = groups[label]
        offset = (g_idx - n_groups / 2 + 0.5) * width
        data = []
        positions = []
        for gene_idx, pos in enumerate(gene_positions):
            values = layer_data[cell_indices, pos]
            values = values[np.isfinite(values)]
            data.append(np.asarray(values, dtype=np.float64))
            positions.append(gene_idx + offset)

        color = colors[g_idx] if colors is not None else None

        if plot_type == "violin":
            if all(len(d) > 0 for d in data):
                parts = ax.violinplot(
                    data, positions=positions, widths=width * 0.9, showmedians=True
                )
                if color is not None:
                    for pc in parts.get("bodies", []):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.6)
        elif plot_type == "box":
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                manage_ticks=False,
            )
            if color is not None:
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
        else:
            for d in data:
                ax.hist(d, bins=20, alpha=0.4, color=color, label=label)
            break

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_names, rotation=45, ha="right", fontsize=7)
    if n_groups <= 10:
        from matplotlib.patches import Patch

        handles = []
        for g_idx, label in enumerate(group_labels):
            c = colors[g_idx] if colors is not None else f"C{g_idx}"
            handles.append(Patch(facecolor=c, alpha=0.6, label=label))
        ax.legend(handles=handles, fontsize=6, loc="upper right")


__all__ = ["plot_distributions_command"]
