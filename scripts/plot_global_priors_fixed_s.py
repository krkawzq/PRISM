#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, cast

import anndata as ad
import matplotlib
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.model import ObservationBatch, PriorFitConfig, fit_gene_priors


@dataclass(frozen=True, slots=True)
class DatasetFitResult:
    dataset_label: str
    dataset_path: Path
    gene_names: list[str]
    p_grid: np.ndarray
    mu_grid: np.ndarray
    prior_weights: np.ndarray
    mean_reference_count: float
    S: float
    n_cells_used: int
    best_loss: float
    final_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and compare prior curves across one or more datasets under a fixed shared S."
    )
    parser.add_argument("h5ad", nargs="+", type=Path)
    parser.add_argument("--ranked-genes", type=Path, required=True)
    parser.add_argument("--reference-genes", type=Path, required=True)
    parser.add_argument(
        "--eligible-genes",
        type=Path,
        default=None,
        help="Optional gene list that further constrains which ranked genes may be fit.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--fixed-s", type=float, default=1e4)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--sigma-bins", type=float, default=1.0)
    parser.add_argument("--align-loss-weight", type=float, default=1.0)
    parser.add_argument("--torch-dtype", type=str, default="float64")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--cell-chunk-size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="cosine")
    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    genes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not genes:
        raise ValueError(f"gene list is empty: {path}")
    return genes


def select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def slice_matrix(
    matrix, gene_positions: list[int], cell_indices: np.ndarray | None = None
) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=np.float64)
    return np.asarray(subset, dtype=np.float64)


def compute_reference_counts(
    matrix, gene_positions: list[int], cell_indices: np.ndarray | None = None
) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
    if sparse.issparse(subset):
        totals = np.asarray(subset.sum(axis=1)).reshape(-1)
    else:
        totals = np.asarray(subset, dtype=np.float64).sum(axis=1)
    return np.asarray(totals, dtype=np.float64)


def sample_indices(n_items: int, n_select: int | None, seed: int) -> np.ndarray:
    if n_select is None or n_select >= n_items:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(np.arange(n_items, dtype=np.int64), size=n_select, replace=False)
    )


def choose_fit_genes(
    dataset_gene_names: list[str],
    ranked_genes: list[str],
    eligible_genes: set[str] | None,
    top_k: int,
) -> list[str]:
    gene_set = set(dataset_gene_names)
    chosen: list[str] = []
    for gene in ranked_genes:
        if gene not in gene_set:
            continue
        if eligible_genes is not None and gene not in eligible_genes:
            continue
        chosen.append(gene)
        if len(chosen) >= top_k:
            break
    if not chosen:
        raise ValueError(
            "no fit genes remain after intersecting ranked genes with the dataset"
        )
    return chosen


def fit_dataset(
    dataset_path: Path,
    *,
    ranked_genes: list[str],
    reference_genes: list[str],
    eligible_genes: set[str] | None,
    top_k: int,
    n_samples: int | None,
    seed: int,
    layer: str | None,
    fixed_s: float,
    config: PriorFitConfig,
    device: str,
) -> DatasetFitResult:
    adata = ad.read_h5ad(dataset_path)
    matrix = select_matrix(adata, layer)
    gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    fit_genes = choose_fit_genes(gene_names, ranked_genes, eligible_genes, top_k)
    ref_positions = [
        gene_to_idx[name] for name in reference_genes if name in gene_to_idx
    ]
    if not ref_positions:
        raise ValueError(f"reference genes have no overlap with {dataset_path}")
    sampled = sample_indices(int(adata.n_obs), n_samples, seed)
    reference_counts = compute_reference_counts(matrix, ref_positions, sampled)
    valid_mask = reference_counts > 0
    if int(np.count_nonzero(valid_mask)) == 0:
        raise ValueError(
            f"reference counts are zero for all sampled cells in {dataset_path}"
        )
    sampled = sampled[valid_mask]
    reference_counts = reference_counts[valid_mask]
    fit_positions = [gene_to_idx[name] for name in fit_genes]
    counts = slice_matrix(matrix, fit_positions, sampled)
    result = fit_gene_priors(
        ObservationBatch(
            gene_names=fit_genes,
            counts=counts,
            reference_counts=reference_counts,
        ),
        S=fixed_s,
        config=config,
        device=device,
    )
    priors = result.priors.batched()
    return DatasetFitResult(
        dataset_label=dataset_path.stem,
        dataset_path=dataset_path,
        gene_names=fit_genes,
        p_grid=np.asarray(priors.p_grid, dtype=np.float64),
        mu_grid=np.asarray(priors.mu_grid, dtype=np.float64),
        prior_weights=np.asarray(priors.weights, dtype=np.float64),
        mean_reference_count=float(np.mean(reference_counts)),
        S=float(fixed_s),
        n_cells_used=int(reference_counts.shape[0]),
        best_loss=float(result.best_loss),
        final_loss=float(result.final_loss),
    )


def write_summary(results: list[DatasetFitResult], outdir: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "dataset_label": result.dataset_label,
                "dataset_path": str(result.dataset_path),
                "n_genes": len(result.gene_names),
                "n_cells_used": result.n_cells_used,
                "S": result.S,
                "mean_reference_count": result.mean_reference_count,
                "best_loss": result.best_loss,
                "final_loss": result.final_loss,
            }
        )
    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (outdir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: list[DatasetFitResult], outdir: Path) -> None:
    genes = results[0].gene_names
    for gene_idx, gene_name in enumerate(genes):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        for result in results:
            axes[0].plot(
                result.mu_grid[gene_idx],
                result.prior_weights[gene_idx],
                lw=2.0,
                label=result.dataset_label,
            )
            axes[1].plot(
                result.p_grid[gene_idx],
                result.prior_weights[gene_idx],
                lw=2.0,
                label=result.dataset_label,
            )
        axes[0].set_title(f"{gene_name}: prior mass over mu")
        axes[0].set_xlabel("mu")
        axes[0].set_ylabel("weight")
        axes[1].set_title(f"{gene_name}: prior mass over p")
        axes[1].set_xlabel("p")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"{gene_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    ranked_genes = read_gene_list(args.ranked_genes.expanduser().resolve())
    reference_genes = read_gene_list(args.reference_genes.expanduser().resolve())
    eligible_genes = (
        None
        if args.eligible_genes is None
        else set(read_gene_list(args.eligible_genes.expanduser().resolve()))
    )
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else Path("output")
        / f"global_prior_compare_{'_'.join(path.stem for path in args.h5ad)}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    config = PriorFitConfig(
        grid_size=args.grid_size,
        sigma_bins=args.sigma_bins,
        align_loss_weight=args.align_loss_weight,
        lr=args.lr,
        n_iter=args.n_iter,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip=None,
        init_temperature=args.init_temperature,
        cell_chunk_size=args.cell_chunk_size,
        optimizer=cast(Any, args.optimizer),
        scheduler=cast(Any, args.scheduler),
        torch_dtype=cast(Any, args.torch_dtype),
    )
    results = [
        fit_dataset(
            path.expanduser().resolve(),
            ranked_genes=ranked_genes,
            reference_genes=reference_genes,
            eligible_genes=eligible_genes,
            top_k=args.top_k,
            n_samples=args.n_samples,
            seed=args.seed,
            layer=args.layer,
            fixed_s=args.fixed_s,
            config=config,
            device=args.device,
        )
        for path in args.h5ad
    ]
    write_summary(results, outdir)
    plot_results(results, outdir)
    print(f"saved {outdir}")


if __name__ == "__main__":
    main()
