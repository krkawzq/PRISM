#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.model import (
    ObservationBatch,
    PriorFitConfig,
    fit_gene_priors,
    fit_gene_priors_em,
    mean_reference_count,
)

console = Console()
install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gradient-based and EM prior fitting visually."
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gene", action="append", default=[])
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--reference-genes", type=Path, default=None)
    parser.add_argument("--label-key", type=str, default=None)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include a global fit alongside the selected labels.",
    )
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--S", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--sigma-bins", type=float, default=1.0)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cell-chunk-size", type=int, default=512)
    parser.add_argument(
        "--em-tol",
        type=float,
        default=1e-6,
        help="Early-stop tolerance for EM max absolute update.",
    )
    parser.add_argument(
        "--torch-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        gene_names = payload.get("gene_names")
        if not isinstance(gene_names, list) or not all(
            isinstance(x, str) for x in gene_names
        ):
            raise ValueError(f"invalid gene_names in {path}")
        return [gene for gene in gene_names if gene]
    return [line.strip() for line in text.splitlines() if line.strip()]


def resolve_selected_genes(args: argparse.Namespace) -> list[str]:
    if args.gene and args.gene_list is not None:
        raise ValueError("--gene and --gene-list are mutually exclusive")
    if args.gene:
        return list(dict.fromkeys(args.gene))
    if args.gene_list is not None:
        genes = read_gene_list(args.gene_list.expanduser().resolve())
        return list(dict.fromkeys(genes[: args.top_n]))
    raise ValueError("provide either --gene or --gene-list")


def select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def ensure_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def subsample_cells(
    matrix: np.ndarray, *, max_cells: int | None, seed: int
) -> np.ndarray:
    n_cells = matrix.shape[0]
    if max_cells is None or n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cells, size=max_cells, replace=False)).astype(np.int64)


def build_observation_batch(
    adata: ad.AnnData,
    *,
    selected_genes: list[str],
    reference_genes: list[str] | None,
    layer: str | None,
    max_cells: int | None,
    seed: int,
    cell_indices: np.ndarray | None = None,
) -> tuple[ObservationBatch, float, dict[str, Any]]:
    gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    missing_selected = [gene for gene in selected_genes if gene not in gene_to_idx]
    if missing_selected:
        raise ValueError(f"selected genes missing from dataset: {missing_selected[:5]}")
    if reference_genes is None:
        reference_positions = list(range(len(gene_names)))
        resolved_reference_genes = list(gene_names)
    else:
        resolved_reference_genes = []
        for gene in reference_genes:
            if gene in gene_to_idx and gene not in resolved_reference_genes:
                resolved_reference_genes.append(gene)
        if not resolved_reference_genes:
            raise ValueError("reference gene list has no overlap with the dataset")
        reference_positions = [gene_to_idx[gene] for gene in resolved_reference_genes]

    matrix = ensure_dense(select_matrix(adata, layer))
    if cell_indices is not None:
        matrix = matrix[np.asarray(cell_indices, dtype=np.int64)]
    sampled_indices = subsample_cells(matrix, max_cells=max_cells, seed=seed)
    matrix = matrix[sampled_indices]
    reference_counts = matrix[:, reference_positions].sum(axis=1)
    valid_mask = reference_counts > 0
    if not np.any(valid_mask):
        raise ValueError("no cells with positive reference counts after sampling")
    matrix = matrix[valid_mask]
    reference_counts = np.asarray(reference_counts[valid_mask], dtype=np.float64)
    target_positions = [gene_to_idx[gene] for gene in selected_genes]
    batch = ObservationBatch(
        gene_names=list(selected_genes),
        counts=np.asarray(matrix[:, target_positions], dtype=np.float64),
        reference_counts=reference_counts,
    )
    batch.check_shape()
    S = mean_reference_count(reference_counts)
    metadata = {
        "n_cells_total": int(
            adata.n_obs if cell_indices is None else len(cell_indices)
        ),
        "n_cells_used": int(batch.n_cells),
        "n_reference_genes": int(len(reference_positions)),
        "n_fit_genes": int(len(selected_genes)),
        "mean_reference_count": float(S),
    }
    return batch, float(S), metadata


def resolve_scopes(
    adata: ad.AnnData, args: argparse.Namespace
) -> list[tuple[str, np.ndarray | None]]:
    scopes: list[tuple[str, np.ndarray | None]] = []
    if args.include_global or not args.label:
        scopes.append(("global", None))
    if args.label_key is None:
        if args.label:
            raise ValueError("--label-key is required when using --label")
        return scopes
    if args.label_key not in adata.obs.columns:
        raise KeyError(f"obs column {args.label_key!r} does not exist")
    labels = np.asarray(adata.obs[args.label_key].astype(str)).reshape(-1)
    selected_labels = list(dict.fromkeys(args.label))
    for label in selected_labels:
        indices = np.flatnonzero(labels == label).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"label {label!r} has no cells in {args.label_key!r}")
        scopes.append((f"label:{label}", indices))
    return scopes


def build_figure(
    selected_genes: list[str],
    scope_results: dict[str, tuple[Any, Any]],
    output_path: Path,
) -> None:
    scope_names = list(scope_results)
    n_genes = len(selected_genes)
    n_scopes = len(scope_names)
    fig, axes = plt.subplots(
        n_genes,
        n_scopes,
        figsize=(5.0 * n_scopes, 3.6 * n_genes),
        squeeze=False,
    )
    for row_idx, gene_name in enumerate(selected_genes):
        row_max = 0.0
        for scope_name in scope_names:
            fit_result, em_result = scope_results[scope_name]
            row_max = max(
                row_max,
                float(
                    np.max(
                        np.asarray(fit_result.priors.mu_grid[row_idx], dtype=np.float64)
                    )
                ),
                float(
                    np.max(
                        np.asarray(em_result.priors.mu_grid[row_idx], dtype=np.float64)
                    )
                ),
            )
        row_max = max(row_max, 1e-12)
        for col_idx, scope_name in enumerate(scope_names):
            ax = axes[row_idx][col_idx]
            fit_result, em_result = scope_results[scope_name]
            p_grid_fit = np.asarray(
                fit_result.priors.mu_grid[row_idx], dtype=np.float64
            )
            w_fit = np.asarray(fit_result.priors.weights[row_idx], dtype=np.float64)
            p_grid_em = np.asarray(em_result.priors.mu_grid[row_idx], dtype=np.float64)
            w_em_raw = np.asarray(
                em_result.posterior_average[row_idx], dtype=np.float64
            )
            w_em = np.asarray(em_result.priors.weights[row_idx], dtype=np.float64)
            ax.plot(p_grid_fit, w_fit, lw=2.2, color="#1d4ed8", label="gradient")
            ax.plot(
                p_grid_em,
                w_em_raw,
                lw=1.8,
                ls="--",
                color="#7c3aed",
                label="em raw",
            )
            ax.plot(p_grid_em, w_em, lw=2.2, color="#c2410c", label="em smooth")
            ax.set_xlim(0.0, row_max)
            if row_idx == 0:
                ax.set_title(scope_name)
            ax.set_xlabel("mu")
            ax.set_ylabel(f"{gene_name}\nprior mass" if col_idx == 0 else "prior mass")
            ax.grid(alpha=0.2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    selected_genes = resolve_selected_genes(args)
    reference_genes = (
        None
        if args.reference_genes is None
        else read_gene_list(args.reference_genes.expanduser().resolve())
    )
    intro = Table(show_header=False, box=None)
    intro.add_row("Input", str(input_path))
    intro.add_row("Output", str(output_path))
    intro.add_row("Genes", ", ".join(selected_genes))
    intro.add_row(
        "Max cells", str(args.max_cells) if args.max_cells is not None else "All"
    )
    intro.add_row("Label key", args.label_key or "None")
    intro.add_row("Labels", ", ".join(args.label) if args.label else "None")
    intro.add_row("Device", args.device)
    console.print(Panel(intro, title="Test EM", border_style="cyan"))

    with console.status("Loading data and building observation batch..."):
        adata = ad.read_h5ad(input_path)
        scopes = resolve_scopes(adata, args)
    config = PriorFitConfig(
        grid_size=args.grid_size,
        sigma_bins=args.sigma_bins,
        lr=args.lr,
        n_iter=args.n_iter,
        cell_chunk_size=args.cell_chunk_size,
        torch_dtype=args.torch_dtype,
    )

    scope_results: dict[str, tuple[Any, Any]] = {}
    scope_metadata: dict[str, dict[str, Any]] = {}
    scope_timing: dict[str, dict[str, float | int]] = {}
    default_S: float | None = None
    for scope_index, (scope_name, cell_indices) in enumerate(scopes, start=1):
        with console.status(f"Building batch for {scope_name}..."):
            batch, scope_default_S, metadata = build_observation_batch(
                adata,
                selected_genes=selected_genes,
                reference_genes=reference_genes,
                layer=args.layer,
                max_cells=args.max_cells,
                seed=args.random_seed + scope_index,
                cell_indices=cell_indices,
            )
        if default_S is None:
            default_S = scope_default_S
        with console.status(f"Fitting gradient priors for {scope_name}..."):
            gradient_start = perf_counter()
            fit_result = fit_gene_priors(
                batch,
                S=float(scope_default_S if args.S is None else args.S),
                config=config,
                device=args.device,
            )
            gradient_elapsed = perf_counter() - gradient_start
        with console.status(f"Fitting EM priors for {scope_name}..."):
            em_start = perf_counter()
            em_result = fit_gene_priors_em(
                batch,
                S=float(scope_default_S if args.S is None else args.S),
                config=config,
                device=args.device,
                tol=args.em_tol,
            )
            em_elapsed = perf_counter() - em_start
        scope_results[scope_name] = (fit_result, em_result)
        scope_metadata[scope_name] = metadata
        scope_timing[scope_name] = {
            "gradient_sec": float(gradient_elapsed),
            "em_sec": float(em_elapsed),
            "gradient_steps": int(len(fit_result.loss_history)),
            "em_steps": int(len(em_result.loss_history)),
        }

    if default_S is None:
        raise RuntimeError("no fitting scopes were resolved")
    resolved_S = float(default_S if args.S is None else args.S)
    with console.status("Rendering comparison figure..."):
        build_figure(selected_genes, scope_results, output_path)

    summary = Table(title="EM Comparison Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Genes", str(len(selected_genes)))
    summary.add_row("Scopes", ", ".join(scope_results))
    summary.add_row(
        "Cells used",
        "; ".join(
            f"{scope}:{meta['n_cells_used']}" for scope, meta in scope_metadata.items()
        ),
    )
    summary.add_row(
        "Reference genes", str(next(iter(scope_metadata.values()))["n_reference_genes"])
    )
    summary.add_row("S", f"{resolved_S:.4f}")
    summary.add_row(
        "Gradient time",
        "; ".join(
            f"{scope}:{timing['gradient_sec']:.2f}s/{timing['gradient_steps']} iters"
            for scope, timing in scope_timing.items()
        ),
    )
    summary.add_row(
        "EM time",
        "; ".join(
            f"{scope}:{timing['em_sec']:.2f}s/{timing['em_steps']} iters"
            for scope, timing in scope_timing.items()
        ),
    )
    summary.add_row(
        "Gradient final loss",
        "; ".join(
            f"{scope}:{results[0].final_loss:.6f}"
            for scope, results in scope_results.items()
        ),
    )
    summary.add_row(
        "EM final loss",
        "; ".join(
            f"{scope}:{results[1].final_loss:.6f}"
            for scope, results in scope_results.items()
        ),
    )
    summary.add_row("Output", str(output_path))
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(Panel(str(exc), title="test_em failed", border_style="red"))
        raise SystemExit(1) from exc
