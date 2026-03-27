#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import anndata as ad
import matplotlib
import numpy as np
import torch
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.model import (
    GeneBatch,
    PriorEngine,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
)
from prism.model import DTYPE_NP
from prism.server.services.datasets import (
    resolve_gene_query,
    select_matrix,
    slice_gene_counts,
)


@dataclass(frozen=True, slots=True)
class DatasetFitResult:
    dataset_label: str
    dataset_path: Path
    overlap_gene_count: int
    n_cells_total: int
    n_cells_used: int
    gene_names: list[str]
    support: np.ndarray
    prior_weights: np.ndarray
    grid_max: np.ndarray
    best_loss: float
    final_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit global prior curves for a small gene subset with fixed S and compare one or more datasets."
        )
    )
    parser.add_argument(
        "h5ad", nargs="+", type=Path, help="One or more input h5ad files."
    )
    parser.add_argument(
        "--ranked-genes",
        type=Path,
        default=None,
        help="Ordered gene list used to choose the fitting top-k genes.",
    )
    parser.add_argument(
        "--genes",
        type=Path,
        default=None,
        help="Deprecated alias of --ranked-genes.",
    )
    parser.add_argument(
        "--overlap-genes",
        type=Path,
        default=None,
        help="Optional overlap gene list used only to constrain which ranked genes are eligible.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Only fit the first K eligible genes from --ranked-genes.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Randomly subsample this many cells per dataset for fitting.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for cell sampling."
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer name. Defaults to X.",
    )
    parser.add_argument(
        "--fixed-s",
        type=float,
        default=1e4,
        help="Fixed shared S used for all fits.",
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Only use control-group cells for fitting.",
    )
    parser.add_argument(
        "--control-key",
        type=str,
        default="perturbation",
        help="obs column used to identify control cells.",
    )
    parser.add_argument(
        "--control-value",
        type=str,
        default="control",
        help="obs value treated as control when --control-only is set.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to output/global_prior_compare_<dataset names>.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--sigma-bins", type=float, default=1.0)
    parser.add_argument("--align-loss-weight", type=float, default=1.0)
    parser.add_argument("--torch-dtype", type=str, default="float64")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--init-temperature", type=float, default=1.0)
    parser.add_argument("--cell-chunk-size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument(
        "--gene-batch-size",
        type=int,
        default=100,
        help="Number of genes fit jointly per chunk to limit GPU memory.",
    )
    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    genes = [gene for gene in genes if gene]
    if not genes:
        raise ValueError(f"gene list is empty: {path}")
    return genes


def resolve_ranked_genes_path(args: argparse.Namespace) -> Path:
    if args.ranked_genes is not None and args.genes is not None:
        raise ValueError("use either --ranked-genes or --genes, not both")
    ranked = args.ranked_genes if args.ranked_genes is not None else args.genes
    if ranked is None:
        raise ValueError("--ranked-genes is required")
    return ranked.expanduser().resolve()


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def resolve_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA unavailable; falling back to cpu")
        return "cpu"
    return requested


def compute_totals(matrix) -> np.ndarray:
    try:
        totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    except AttributeError:
        totals = np.asarray(matrix[:, :], dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP)


def counts_fit_totals(counts_matrix: np.ndarray, totals: np.ndarray) -> bool:
    counts_np = np.asarray(counts_matrix, dtype=np.float64)
    totals_np = np.asarray(totals, dtype=np.float64).reshape(-1)
    return bool(np.all(counts_np <= totals_np[:, None] + 1e-12))


def sample_indices(n_items: int, n_select: int | None, seed: int) -> np.ndarray:
    if n_select is None or n_select >= n_items:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_items, size=n_select, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def sample_from_candidates(
    candidate_indices: np.ndarray, n_select: int | None, seed: int
) -> np.ndarray:
    candidate_idx_np = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_idx_np.ndim != 1:
        candidate_idx_np = candidate_idx_np.reshape(-1)
    if candidate_idx_np.size == 0:
        raise ValueError("no candidate cells available for sampling")
    local = sample_indices(candidate_idx_np.size, n_select, seed)
    return np.sort(candidate_idx_np[local])


def resolve_gene_name_map(
    adata: ad.AnnData, candidate_genes: list[str]
) -> dict[str, str]:
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_names_lower = tuple(str(name).lower() for name in gene_names.tolist())
    gene_to_idx = {str(name): int(idx) for idx, name in enumerate(gene_names.tolist())}
    gene_lower_to_idx: dict[str, int] = {}
    for idx, name in enumerate(gene_names_lower):
        gene_lower_to_idx.setdefault(name, idx)

    resolved: dict[str, str] = {}
    seen: set[str] = set()
    for gene in candidate_genes:
        try:
            idx = resolve_gene_query(
                gene,
                gene_names,
                gene_names_lower,
                gene_to_idx,
                gene_lower_to_idx,
            )
        except Exception:
            continue
        resolved_name = str(gene_names[idx])
        if resolved_name in seen:
            continue
        seen.add(resolved_name)
        resolved[gene] = resolved_name
    return resolved


def resolve_gene_indices(
    adata: ad.AnnData, candidate_genes: list[str]
) -> dict[str, int]:
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_names_lower = tuple(str(name).lower() for name in gene_names.tolist())
    gene_to_idx = {str(name): int(idx) for idx, name in enumerate(gene_names.tolist())}
    gene_lower_to_idx: dict[str, int] = {}
    for idx, name in enumerate(gene_names_lower):
        gene_lower_to_idx.setdefault(name, idx)

    resolved: dict[str, int] = {}
    seen: set[int] = set()
    for gene in candidate_genes:
        try:
            idx = resolve_gene_query(
                gene,
                gene_names,
                gene_names_lower,
                gene_to_idx,
                gene_lower_to_idx,
            )
        except Exception:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        resolved[gene] = int(idx)
    return resolved


def slice_gene_matrix(matrix, gene_names: list[str], adata: ad.AnnData) -> np.ndarray:
    var_names = np.asarray(adata.var_names.astype(str))
    lookup = {str(name): int(idx) for idx, name in enumerate(var_names.tolist())}
    columns = [slice_gene_counts(matrix, lookup[name]) for name in gene_names]
    return np.column_stack(columns).astype(DTYPE_NP, copy=False)


def dense_sampled_rows(matrix, row_indices: np.ndarray, *, desc: str) -> np.ndarray:
    row_idx_np = np.asarray(row_indices, dtype=np.int64)
    progress = tqdm(total=2, desc=desc, leave=True, unit="step")
    try:
        sampled = matrix[row_idx_np, :]
        progress.update(1)
        if hasattr(sampled, "toarray"):
            dense = sampled.toarray()
        else:
            dense = np.asarray(sampled)
        progress.update(1)
    finally:
        progress.close()
    return np.asarray(dense, dtype=DTYPE_NP)


def iter_gene_slices(n_genes: int, chunk_size: int) -> list[slice]:
    if chunk_size < 1:
        raise ValueError(f"gene_batch_size must be >= 1, got {chunk_size}")
    return [
        slice(start, min(start + chunk_size, n_genes))
        for start in range(0, n_genes, chunk_size)
    ]


def weights_to_density(support: np.ndarray, weights: np.ndarray) -> np.ndarray:
    support_np = np.asarray(support, dtype=np.float64)
    weights_np = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if support_np.size == 1:
        return np.ones_like(weights_np, dtype=np.float64)
    step = float(np.median(np.diff(support_np)))
    step = max(step, 1e-12)
    density = weights_np / step
    area = float(np.trapezoid(density, support_np))
    if area <= 0:
        density = np.ones_like(support_np, dtype=np.float64)
        area = float(np.trapezoid(density, support_np))
    return density / max(area, 1e-12)


def curve_quantile(support: np.ndarray, density: np.ndarray, q: float) -> float:
    support_np = np.asarray(support, dtype=np.float64)
    density_np = np.clip(np.asarray(density, dtype=np.float64), 0.0, None)
    if support_np.size <= 1:
        return float(support_np[-1])
    dx = np.diff(support_np)
    mid_mass = 0.5 * (density_np[:-1] + density_np[1:]) * dx
    total = float(np.sum(mid_mass))
    if total <= 0:
        return float(support_np[-1])
    cdf = np.concatenate([[0.0], np.cumsum(mid_mass) / total])
    return float(np.interp(q, cdf, support_np))


def prior_mean(support: np.ndarray, weights: np.ndarray) -> float:
    density = weights_to_density(support, weights)
    return float(np.trapezoid(np.asarray(support, dtype=np.float64) * density, support))


def fit_dataset(
    *,
    path: Path,
    gene_names: list[str],
    overlap_genes: list[str],
    n_samples: int | None,
    seed: int,
    layer: str | None,
    fixed_s: float,
    device: str,
    grid_size: int,
    sigma_bins: float,
    align_loss_weight: float,
    torch_dtype: str,
    lr: float,
    n_iter: int,
    lr_min_ratio: float,
    grad_clip: float | None,
    init_temperature: float,
    cell_chunk_size: int,
    optimizer: str,
    scheduler: str,
    gene_batch_size: int,
    control_only: bool,
    control_key: str,
    control_value: str,
) -> DatasetFitResult:
    adata = ad.read_h5ad(path, backed="r")
    progress = None
    try:
        matrix = select_matrix(adata, layer)
        overlap_idx_map = resolve_gene_indices(adata, overlap_genes)
        if len(overlap_idx_map) != len(overlap_genes):
            missing = [gene for gene in overlap_genes if gene not in overlap_idx_map]
            raise ValueError(
                f"dataset {path.stem} is missing overlap genes required for Nc: {missing[:5]}"
            )
        if control_only:
            if control_key not in adata.obs.columns:
                raise KeyError(f"control key {control_key!r} not found in obs")
            control_mask = (
                np.asarray(adata.obs[control_key].astype(str)).reshape(-1)
                == control_value
            )
            candidate_idx = np.flatnonzero(control_mask)
            if candidate_idx.size == 0:
                raise ValueError(
                    f"dataset {path.stem} has no control cells for {control_key}={control_value!r}"
                )
            idx = sample_from_candidates(candidate_idx, n_samples, seed)
        else:
            idx = sample_indices(adata.n_obs, n_samples, seed)
        sampled_dense = dense_sampled_rows(matrix, idx, desc=f"dense rows {path.stem}")
        overlap_indices = np.asarray(list(overlap_idx_map.values()), dtype=np.int64)
        overlap_matrix = sampled_dense[:, overlap_indices]
        totals = np.asarray(np.sum(overlap_matrix, axis=1), dtype=DTYPE_NP)
        var_names = np.asarray(adata.var_names.astype(str))
        gene_lookup = {str(name): int(i) for i, name in enumerate(var_names.tolist())}
        selected_gene_indices = np.asarray(
            [gene_lookup[name] for name in gene_names], dtype=np.int64
        )
        if np.all(np.isin(selected_gene_indices, overlap_indices)):
            overlap_col_lookup = {
                int(gene_idx): pos
                for pos, gene_idx in enumerate(overlap_indices.tolist())
            }
            counts_matrix = overlap_matrix[
                :,
                [
                    overlap_col_lookup[int(gene_idx)]
                    for gene_idx in selected_gene_indices
                ],
            ].astype(DTYPE_NP, copy=False)
        else:
            counts_matrix = sampled_dense[:, selected_gene_indices].astype(
                DTYPE_NP, copy=False
            )
        if not counts_fit_totals(counts_matrix, totals):
            raise ValueError(
                "selected gene counts exceed overlap-based totals; ensure fitted genes are within overlap genes"
            )
        if np.any(counts_matrix > fixed_s + 1e-12):
            max_count = float(np.max(counts_matrix))
            raise ValueError(
                f"counts exceed fixed S={fixed_s}; max selected count is {max_count}"
            )

        gene_slices = iter_gene_slices(len(gene_names), gene_batch_size)
        support_chunks: list[np.ndarray] = []
        prior_weight_chunks: list[np.ndarray] = []
        grid_max_chunks: list[np.ndarray] = []
        best_losses: list[float] = []
        final_losses: list[float] = []
        chunk_progress = tqdm(
            total=len(gene_slices),
            desc=f"gene chunks {path.stem}",
            leave=True,
            unit="chunk",
        )
        try:
            for chunk_idx, gene_slice in enumerate(gene_slices, start=1):
                chunk_gene_names = gene_names[gene_slice]
                chunk_counts = counts_matrix[:, gene_slice]
                batch = GeneBatch(
                    gene_names=chunk_gene_names,
                    counts=chunk_counts,
                    totals=totals,
                )
                engine = PriorEngine(
                    chunk_gene_names,
                    setting=PriorEngineSetting(
                        grid_size=grid_size,
                        sigma_bins=sigma_bins,
                        align_loss_weight=align_loss_weight,
                        torch_dtype=torch_dtype,
                    ),
                    device=device,
                )
                progress = tqdm(
                    total=n_iter,
                    desc=f"fit {path.stem} {chunk_idx}/{len(gene_slices)}",
                    leave=True,
                    unit="iter",
                )

                last_step = 0

                def on_fit_progress(
                    step: int,
                    total: int,
                    loss_value: float,
                    nll_value: float,
                    align_value: float,
                    _best_updated: bool,
                ) -> None:
                    nonlocal last_step
                    if progress.total != total:
                        progress.total = total
                    delta = max(step - last_step, 0)
                    if delta:
                        progress.update(delta)
                        last_step = step
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        nll=f"{nll_value:.4f}",
                        jsd=f"{align_value:.4f}",
                    )

                report = engine.fit_report(
                    batch,
                    s_hat=fixed_s,
                    training_cfg=PriorEngineTrainingConfig(
                        lr=lr,
                        n_iter=n_iter,
                        lr_min_ratio=lr_min_ratio,
                        grad_clip=grad_clip,
                        init_temperature=init_temperature,
                        cell_chunk_size=cell_chunk_size,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    ),
                    progress_callback=on_fit_progress,
                )
                if last_step < progress.total:
                    progress.update(progress.total - last_step)
                progress.close()
                progress = None
                support_chunks.append(np.asarray(report.support, dtype=np.float64))
                prior_weight_chunks.append(
                    np.asarray(report.prior_weights, dtype=np.float64)
                )
                grid_max_chunks.append(np.asarray(report.grid_max, dtype=np.float64))
                best_losses.append(float(report.best_loss))
                final_losses.append(float(report.final_loss))
                chunk_progress.update(1)
        finally:
            chunk_progress.close()
        return DatasetFitResult(
            dataset_label=path.stem,
            dataset_path=path,
            overlap_gene_count=len(overlap_genes),
            n_cells_total=int(adata.n_obs),
            n_cells_used=int(len(idx)),
            gene_names=list(gene_names),
            support=np.concatenate(support_chunks, axis=0),
            prior_weights=np.concatenate(prior_weight_chunks, axis=0),
            grid_max=np.concatenate(grid_max_chunks, axis=0),
            best_loss=float(np.mean(best_losses)),
            final_loss=float(np.mean(final_losses)),
        )
    except Exception:
        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass
        raise
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def save_summary(results: list[DatasetFitResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "gene",
                "n_cells_total",
                "n_cells_used",
                "overlap_gene_count",
                "grid_max",
                "prior_mean",
                "best_loss",
                "final_loss",
            ]
        )
        for result in results:
            for idx, gene_name in enumerate(result.gene_names):
                support = np.asarray(result.support[idx], dtype=np.float64)
                weights = np.asarray(result.prior_weights[idx], dtype=np.float64)
                writer.writerow(
                    [
                        result.dataset_label,
                        gene_name,
                        result.n_cells_total,
                        result.n_cells_used,
                        result.overlap_gene_count,
                        float(result.grid_max[idx]),
                        prior_mean(support, weights),
                        result.best_loss,
                        result.final_loss,
                    ]
                )


def save_curve_csv(results: list[DatasetFitResult], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "gene", "support", "density"])
        for result in results:
            for idx, gene_name in enumerate(result.gene_names):
                support = np.asarray(result.support[idx], dtype=np.float64)
                density = weights_to_density(support, result.prior_weights[idx])
                for x, y in zip(support, density, strict=True):
                    writer.writerow(
                        [result.dataset_label, gene_name, float(x), float(y)]
                    )


def plot_curves(
    results: list[DatasetFitResult],
    gene_names: list[str],
    out_dir: Path,
    *,
    genes_per_figure: int = 10,
) -> list[Path]:
    figure_paths: list[Path] = []
    cmap = plt.get_cmap("tab10", max(len(results), 1))
    gene_slices = iter_gene_slices(len(gene_names), genes_per_figure)

    for fig_idx, gene_slice in enumerate(gene_slices, start=1):
        chunk_gene_names = gene_names[gene_slice]
        n_genes = len(chunk_gene_names)
        fig, axes = plt.subplots(
            n_genes, 1, figsize=(12, max(2.8 * n_genes, 4.0)), squeeze=False
        )

        for local_idx, gene_name in enumerate(chunk_gene_names):
            gene_idx = gene_slice.start + local_idx
            ax = axes[local_idx, 0]
            gene_x_max = 0.0
            for ds_idx, result in enumerate(results):
                support = np.asarray(result.support[gene_idx], dtype=np.float64)
                density = weights_to_density(support, result.prior_weights[gene_idx])
                gene_x_max = max(gene_x_max, curve_quantile(support, density, 0.95))
                ax.plot(
                    support,
                    density,
                    lw=2.0,
                    alpha=0.95,
                    color=cmap(ds_idx),
                    label=f"{result.dataset_label} | n_samples={result.n_cells_used}",
                )
                ax.axvline(
                    float(result.grid_max[gene_idx]),
                    color=cmap(ds_idx),
                    lw=0.9,
                    ls="--",
                    alpha=0.45,
                )
            ax.set_title(gene_name)
            ax.set_xlabel("mu support")
            ax.set_ylabel("F_g")
            ax.set_xlim(0.0, max(gene_x_max * 1.05, 1.0))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper right")
        fig.suptitle(
            f"Global prior curves with fixed S | genes {gene_slice.start + 1}-{gene_slice.stop}",
            fontsize=14,
        )
        fig.tight_layout()
        out_path = out_dir / f"global_prior_curves_part_{fig_idx:02d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        figure_paths.append(out_path)

    return figure_paths


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.device)
    paths = [path.expanduser().resolve() for path in args.h5ad]
    if len(paths) < 1:
        raise ValueError("at least one dataset is required")
    ranked_genes_path = resolve_ranked_genes_path(args)
    ranked_genes = read_gene_list(ranked_genes_path)
    if args.overlap_genes is None:
        raise ValueError("--overlap-genes is required to define Nc totals consistently")
    overlap_genes_path = args.overlap_genes.expanduser().resolve()
    overlap_genes = read_gene_list(overlap_genes_path)
    overlap_gene_set = set(overlap_genes)

    dataset_labels = [sanitize_label(path.stem) for path in paths]
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else (
            PROJECT_ROOT / "output" / f"global_prior_compare_{'_'.join(dataset_labels)}"
        ).resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    resolved_by_dataset: list[dict[str, str]] = []
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        try:
            resolved_by_dataset.append(resolve_gene_name_map(adata, ranked_genes))
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
    eligible_ranked_genes = [gene for gene in ranked_genes if gene in overlap_gene_set]
    common_tokens = [
        gene
        for gene in eligible_ranked_genes
        if all(gene in resolved for resolved in resolved_by_dataset)
    ]
    if not common_tokens:
        raise ValueError(
            "no ranked genes remain after overlap filtering and dataset resolution"
        )
    selected_genes = [
        resolved_by_dataset[0][gene] for gene in common_tokens[: args.top_k]
    ]
    if not selected_genes:
        raise ValueError("top-k selection is empty")

    n_samples = args.n_samples

    results = [
        fit_dataset(
            path=path,
            gene_names=selected_genes,
            overlap_genes=overlap_genes,
            n_samples=n_samples,
            seed=args.seed + idx,
            layer=args.layer,
            fixed_s=args.fixed_s,
            device=resolved_device,
            grid_size=args.grid_size,
            sigma_bins=args.sigma_bins,
            align_loss_weight=args.align_loss_weight,
            torch_dtype=args.torch_dtype,
            lr=args.lr,
            n_iter=args.n_iter,
            lr_min_ratio=args.lr_min_ratio,
            grad_clip=args.grad_clip,
            init_temperature=args.init_temperature,
            cell_chunk_size=args.cell_chunk_size,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            gene_batch_size=args.gene_batch_size,
            control_only=args.control_only,
            control_key=args.control_key,
            control_value=args.control_value,
        )
        for idx, path in enumerate(paths)
    ]

    (outdir / "selected_genes.txt").write_text(
        "\n".join(selected_genes) + "\n", encoding="utf-8"
    )
    save_summary(results, outdir / "summary.csv")
    save_curve_csv(results, outdir / "curve_points.csv")
    figure_paths = plot_curves(results, selected_genes, outdir, genes_per_figure=10)
    run_payload = {
        "datasets": [str(path) for path in paths],
        "ranked_genes": str(ranked_genes_path),
        "overlap_genes": str(overlap_genes_path),
        "top_k": args.top_k,
        "selected_genes": selected_genes,
        "n_ranked_genes": len(ranked_genes),
        "n_overlap_genes": len(overlap_genes),
        "n_eligible_ranked_genes": len(eligible_ranked_genes),
        "n_samples": n_samples,
        "seed": args.seed,
        "fixed_s": args.fixed_s,
        "device": resolved_device,
        "layer": args.layer,
        "gene_batch_size": args.gene_batch_size,
        "control_only": args.control_only,
        "control_key": args.control_key,
        "control_value": args.control_value,
        "genes_per_figure": 10,
        "figure_paths": [str(path) for path in figure_paths],
        "totals_definition": "sum of counts over overlap genes",
    }
    (outdir / "run_config.json").write_text(
        json.dumps(run_payload, indent=2), encoding="utf-8"
    )

    print(f"datasets       : {[path.stem for path in paths]}")
    print(f"ranked genes   : {ranked_genes_path}")
    print(f"overlap genes  : {overlap_genes_path}")
    print(f"selected genes : {selected_genes}")
    print(f"n_samples      : {n_samples if n_samples is not None else 'all'}")
    print(f"device         : {resolved_device}")
    print(f"gene batch size: {args.gene_batch_size}")
    if args.control_only:
        print(f"control filter : {args.control_key}={args.control_value}")
    print(f"figures        : {len(figure_paths)}")
    for result in results:
        print(
            f"{result.dataset_label}: overlap_genes={result.overlap_gene_count} total={result.n_cells_total} used={result.n_cells_used}"
        )
    print(f"saved          : {outdir}")


if __name__ == "__main__":
    main()
