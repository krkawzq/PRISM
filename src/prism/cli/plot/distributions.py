from __future__ import annotations

from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import typer

from prism.cli.common import (
    console,
    print_elapsed,
    print_saved_path,
    resolve_float,
    resolve_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_str,
)
from prism.io import read_gene_list
from prism.plotting import plt

from .common import normalize_distribution_plot_type, resolve_optional_list


def _plot_ungrouped(
    ax,
    layer_data: np.ndarray,
    gene_names: list[str],
    gene_positions: list[int],
    plot_type: str,
) -> None:
    data = [layer_data[:, pos] for pos in gene_positions]
    data = [
        np.asarray(values[np.isfinite(values)], dtype=np.float64) for values in data
    ]
    positions = list(range(len(gene_names)))
    if plot_type == "violin":
        parts = ax.violinplot(data, positions=positions, showmedians=True)
        for body in parts.get("bodies", []):
            body.set_alpha(0.7)
    elif plot_type == "box":
        ax.boxplot(data, positions=positions, widths=0.6)
    else:
        for idx, values in enumerate(data):
            ax.hist(values, bins=30, alpha=0.5, label=gene_names[idx])
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
    palette: str | None,
) -> None:
    n_groups = len(group_labels)
    n_genes = len(gene_names)
    width = 0.8 / max(n_groups, 1)
    colors = None
    if palette is not None:
        cmap = plt.get_cmap(palette)
        colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    for group_idx, label in enumerate(group_labels):
        cell_indices = groups[label]
        offset = (group_idx - n_groups / 2 + 0.5) * width
        data = []
        positions = []
        for gene_idx, pos in enumerate(gene_positions):
            values = np.asarray(layer_data[cell_indices, pos], dtype=np.float64)
            data.append(values[np.isfinite(values)])
            positions.append(gene_idx + offset)
        color = None if colors is None else colors[group_idx]
        if plot_type == "violin":
            if all(len(values) > 0 for values in data):
                parts = ax.violinplot(
                    data,
                    positions=positions,
                    widths=width * 0.9,
                    showmedians=True,
                )
                if color is not None:
                    for body in parts.get("bodies", []):
                        body.set_facecolor(color)
                        body.set_alpha(0.6)
        elif plot_type == "box":
            boxes = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                manage_ticks=False,
            )
            if color is not None:
                for patch in boxes["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
        else:
            ax.hist(data[0], bins=20, alpha=0.4, color=color, label=label)
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_names, rotation=45, ha="right", fontsize=7)
    if plot_type == "hist":
        ax.set_xticks([])
        ax.set_xlabel(gene_names[0])
        ax.legend(fontsize=6, loc="upper right")
        return
    if n_groups <= 10:
        from matplotlib.patches import Patch

        handles = []
        for group_idx, label in enumerate(group_labels):
            color = f"C{group_idx}" if colors is None else colors[group_idx]
            handles.append(Patch(facecolor=color, alpha=0.6, label=label))
        ax.legend(handles=handles, fontsize=6, loc="upper right")


def _resolve_distribution_gene_names(
    adata: ad.AnnData,
    *,
    gene_names: list[str] | None,
    genes_path: Path | None,
    max_genes: int,
    sample_genes: bool,
    seed: int,
) -> tuple[list[str], list[int]]:
    available_gene_names = [str(name) for name in adata.var_names.tolist()]
    if gene_names is not None and genes_path is not None:
        raise ValueError("--gene and --genes are mutually exclusive")
    if gene_names is not None or genes_path is not None:
        requested = gene_names if gene_names is not None else read_gene_list(genes_path)
        assert requested is not None
        selected_names: list[str] = []
        selected_positions: list[int] = []
        lookup = {gene_name: idx for idx, gene_name in enumerate(available_gene_names)}
        missing: list[str] = []
        for gene_name in requested:
            resolved = str(gene_name).strip()
            if not resolved:
                continue
            idx = lookup.get(resolved)
            if idx is None:
                missing.append(resolved)
                continue
            selected_names.append(resolved)
            selected_positions.append(idx)
        if missing:
            raise ValueError(
                f"{len(missing)} requested genes are missing from the dataset; examples: {missing[:5]}"
            )
        if not selected_names:
            raise ValueError("no genes remain after filtering the requested gene list")
        limit = min(max_genes, len(selected_names))
        return selected_names[:limit], selected_positions[:limit]
    if len(available_gene_names) <= max_genes:
        return available_gene_names, list(range(len(available_gene_names)))
    if sample_genes:
        rng = np.random.default_rng(seed)
        chosen_idx = np.sort(
            rng.choice(len(available_gene_names), size=max_genes, replace=False)
        )
        return [available_gene_names[idx] for idx in chosen_idx], chosen_idx.tolist()
    selected_positions = list(range(max_genes))
    return [available_gene_names[idx] for idx in selected_positions], selected_positions


def plot_distributions_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Extracted h5ad file with signal layers."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    gene_names: list[str] | None = typer.Option(
        None,
        "--gene",
        help="Repeatable gene name to plot. Defaults to the first --max-genes genes.",
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional gene-list file used instead of random or leading-gene selection.",
    ),
    layers: list[str] | None = typer.Option(
        None,
        "--layer",
        help="Repeatable layer names to plot. Defaults to signal, posterior_entropy, mutual_information when present.",
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
    sample_genes: bool = typer.Option(
        False,
        "--sample-genes/--no-sample-genes",
        help="Randomly sample genes when no explicit gene selection is provided.",
    ),
    palette: str | None = typer.Option(
        None, help="Optional matplotlib colormap name for group coloring."
    ),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    genes_path = resolve_optional_path(genes_path)
    group_key = resolve_optional_str(group_key)
    max_genes = resolve_int(max_genes)
    figsize_w = resolve_float(figsize_w)
    figsize_h = resolve_float(figsize_h)
    seed = resolve_int(seed)
    sample_genes = bool(sample_genes)
    palette = resolve_optional_str(palette)
    resolved_plot_type = normalize_distribution_plot_type(resolve_str(plot_type))

    adata = ad.read_h5ad(h5ad_path)
    requested_layers = resolve_optional_list(layers)
    default_layers = ["signal", "posterior_entropy", "mutual_information"]
    selected_layers = (
        requested_layers
        if requested_layers is not None
        else [layer for layer in default_layers if layer in adata.layers]
    )
    if not selected_layers:
        raise ValueError(
            "no matching layers found; available layers: "
            + ", ".join(sorted(adata.layers.keys()))
        )
    missing_layers = [layer for layer in selected_layers if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"layers not found in h5ad: {missing_layers}")

    selected_gene_names, gene_positions = _resolve_distribution_gene_names(
        adata,
        gene_names=resolve_optional_list(gene_names),
        genes_path=genes_path,
        max_genes=max_genes,
        sample_genes=sample_genes,
        seed=seed,
    )

    groups = None
    group_labels = None
    if group_key is not None:
        if group_key not in adata.obs.columns:
            raise KeyError(f"obs column {group_key!r} does not exist")
        labels_array = np.asarray(adata.obs[group_key].astype(str)).reshape(-1)
        group_labels = sorted(np.unique(labels_array).tolist())
        groups = {
            label: np.flatnonzero(labels_array == label) for label in group_labels
        }
        if resolved_plot_type == "hist" and len(selected_gene_names) != 1:
            raise ValueError(
                "grouped histogram plots require exactly one gene; use --gene or choose --plot-type violin/box"
            )

    fig, axes = plt.subplots(
        len(selected_layers),
        1,
        figsize=(figsize_w, figsize_h * len(selected_layers)),
        squeeze=False,
    )
    for layer_idx, layer_name in enumerate(selected_layers):
        ax = axes[layer_idx, 0]
        layer_data = np.asarray(adata.layers[layer_name], dtype=np.float64)
        if groups is None or group_labels is None:
            _plot_ungrouped(
                ax,
                layer_data,
                selected_gene_names,
                gene_positions,
                resolved_plot_type,
            )
        else:
            _plot_grouped(
                ax,
                layer_data,
                selected_gene_names,
                gene_positions,
                groups,
                group_labels,
                resolved_plot_type,
                palette,
            )
        ax.set_title(layer_name)
        if layer_idx == len(selected_layers) - 1:
            ax.set_xlabel("Gene")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_saved_path(console, output_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


__all__ = ["plot_distributions_command"]
