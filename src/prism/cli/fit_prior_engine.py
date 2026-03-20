from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anndata as ad
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy import sparse
from tqdm.auto import tqdm

from prism.model import (
    GeneBatch,
    PriorEngine,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
    fit_pool_scale,
)
from prism.model._typing import DTYPE_NP, OptimizerName, SchedulerName

console = Console()
DEFAULT_ENGINE_SETTING = PriorEngineSetting()
DEFAULT_TRAINING_CONFIG = PriorEngineTrainingConfig()


def fit_prior_engine(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Argument(..., help="Output checkpoint pickle path."),
    layer: str | None = typer.Option(None, help="AnnData layer name to use."),
    device: str = typer.Option("cuda", help="Torch device, e.g. cpu or cuda."),
    batch_size: int = typer.Option(64, min=1, help="Genes per fit batch."),
    max_genes: int | None = typer.Option(
        None, min=1, help="Fit only the first N selected genes."
    ),
    gene_start: int = typer.Option(0, min=0, help="Start gene index, inclusive."),
    gene_end: int | None = typer.Option(None, min=0, help="End gene index, exclusive."),
    r: float = typer.Option(
        0.05, min=1e-12, max=1.0, help="Capture efficiency used to convert rS to S."
    ),
    grid_size: int = typer.Option(
        DEFAULT_ENGINE_SETTING.grid_size, min=2, help="Prior grid size."
    ),
    torch_dtype: str = typer.Option(
        DEFAULT_ENGINE_SETTING.torch_dtype,
        help="Torch dtype for fitting: float64 or float32.",
    ),
    sigma_bins: float = typer.Option(
        DEFAULT_ENGINE_SETTING.sigma_bins,
        min=0.0,
        help="Gaussian smoothing sigma in bins.",
    ),
    align_loss_weight: float = typer.Option(
        DEFAULT_ENGINE_SETTING.align_loss_weight, min=0.0, help="JSD alignment weight."
    ),
    lr: float = typer.Option(
        DEFAULT_TRAINING_CONFIG.lr, min=1e-12, help="Optimizer learning rate."
    ),
    n_iter: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.n_iter, min=1, help="Optimization iterations."
    ),
    lr_min_ratio: float = typer.Option(
        DEFAULT_TRAINING_CONFIG.lr_min_ratio,
        min=0.0,
        help="Minimum lr ratio for scheduler.",
    ),
    grad_clip: float | None = typer.Option(
        DEFAULT_TRAINING_CONFIG.grad_clip, min=0.0, help="Optional gradient clip."
    ),
    init_temperature: float = typer.Option(
        DEFAULT_TRAINING_CONFIG.init_temperature,
        min=1e-12,
        help="Cold-start initialization temperature.",
    ),
    cell_chunk_size: int = typer.Option(
        DEFAULT_TRAINING_CONFIG.cell_chunk_size,
        min=1,
        help="Cells per likelihood chunk during fitting.",
    ),
    optimizer: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.optimizer, help="Optimizer name."
    ),
    scheduler: str = typer.Option(
        DEFAULT_TRAINING_CONFIG.scheduler, help="Scheduler name."
    ),
    pool_max_iter: int = typer.Option(120, min=1, help="Pool scale EM max iterations."),
    pool_tol: float = typer.Option(1e-6, min=1e-16, help="Pool scale EM tolerance."),
    pool_n_quad: int = typer.Option(128, min=2, help="Pool scale quadrature points."),
    use_posterior_mu: bool = typer.Option(
        False, help="Use posterior softargmax point estimate for pool scale."
    ),
    pool_softargmax_temperature: float = typer.Option(
        0.05,
        min=1e-12,
        help="Softargmax temperature when posterior pool estimate is enabled.",
    ),
    rank: int = typer.Option(0, min=0, help="Current worker rank."),
    world_size: int = typer.Option(1, min=1, help="Total number of workers."),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.resolve()
    output_path = output_path.resolve()

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    totals = _compute_totals(matrix)
    all_gene_names = np.asarray(adata.var_names.astype(str))
    selected_gene_names, resolved_gene_start, resolved_gene_end = _select_gene_names(
        all_gene_names,
        gene_start=gene_start,
        gene_end=gene_end,
        max_genes=max_genes,
    )
    if rank >= world_size:
        raise ValueError(
            f"rank 必须满足 rank < world_size，收到 {rank} >= {world_size}"
        )

    console.print("[bold cyan]Fitting[/bold cyan] pool scale")
    pool_estimate = fit_pool_scale(
        totals,
        max_iter=pool_max_iter,
        tol=pool_tol,
        n_quad=pool_n_quad,
        use_posterior_mu=use_posterior_mu,
        softargmax_temperature=pool_softargmax_temperature,
    )
    s_hat = float(pool_estimate.point_eta / r)
    _print_pool_summary(pool_estimate.mu, pool_estimate.sigma, s_hat)

    setting = PriorEngineSetting(
        grid_size=grid_size,
        torch_dtype=cast(Any, torch_dtype),
        sigma_bins=sigma_bins,
        align_loss_weight=align_loss_weight,
    )
    training_cfg = PriorEngineTrainingConfig(
        lr=lr,
        n_iter=n_iter,
        lr_min_ratio=lr_min_ratio,
        grad_clip=grad_clip,
        init_temperature=init_temperature,
        cell_chunk_size=cell_chunk_size,
        optimizer=cast(OptimizerName, optimizer),
        scheduler=cast(SchedulerName, scheduler),
    )
    batch_ranges = _split_gene_batches(
        n_genes=len(selected_gene_names),
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )
    if not batch_ranges:
        raise ValueError("当前 rank 没有分配到任何基因批次")
    local_gene_names = np.concatenate(
        [selected_gene_names[start:end] for _, start, end in batch_ranges]
    )
    resolved_device = _resolve_fit_device(
        requested_device=device,
        n_cells=int(adata.n_obs),
        batch_size=batch_size,
        grid_size=setting.grid_size,
        torch_dtype=setting.torch_dtype,
        cell_chunk_size=training_cfg.cell_chunk_size,
    )
    _print_dataset_summary(
        n_cells=int(adata.n_obs),
        total_genes=int(adata.n_vars),
        selected_genes=len(selected_gene_names),
        local_genes=int(local_gene_names.size),
        layer=layer,
        device=resolved_device,
        batch_size=batch_size,
        cell_chunk_size=training_cfg.cell_chunk_size,
        rank=rank,
        world_size=world_size,
    )
    engine = PriorEngine(
        local_gene_names.tolist(), setting=setting, device=resolved_device
    )

    fit_history: list[dict[str, Any]] = []
    for local_batch_index, (global_batch_index, start, end) in enumerate(
        tqdm(
            batch_ranges,
            desc=f"fit rank {rank}/{world_size}",
            unit="batch",
            dynamic_ncols=True,
        ),
        start=1,
    ):
        batch_gene_names = selected_gene_names[start:end]
        batch_counts = _slice_gene_counts(
            matrix,
            resolved_gene_start + start,
            resolved_gene_start + end,
        )
        gene_batch = GeneBatch(
            gene_names=batch_gene_names.tolist(),
            counts=batch_counts,
            totals=totals,
        )
        fit_summary = engine.fit(gene_batch, s_hat=s_hat, training_cfg=training_cfg)
        fit_history.append(
            {
                "local_batch_index": local_batch_index,
                "global_batch_index": global_batch_index,
                **asdict(fit_summary),
            }
        )

    elapsed_sec = perf_counter() - start_time
    checkpoint = {
        "engine": engine,
        "pool_estimate": pool_estimate,
        "s_hat": s_hat,
        "h5ad_path": str(h5ad_path),
        "layer": layer,
        "gene_start": resolved_gene_start,
        "gene_end": resolved_gene_end,
        "gene_names": local_gene_names.tolist(),
        "global_gene_names": selected_gene_names.tolist(),
        "setting": asdict(setting),
        "training_config": asdict(training_cfg),
        "fit_history": fit_history,
        "n_cells": int(adata.n_obs),
        "elapsed_sec": elapsed_sec,
        "rank": rank,
        "world_size": world_size,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(checkpoint, fh)

    console.print(f"[bold green]Saved[/bold green] checkpoint to {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")
    return 0


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} 不存在")
    return adata.layers[layer]


def _compute_totals(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def _select_gene_names(
    gene_names: np.ndarray,
    *,
    gene_start: int,
    gene_end: int | None,
    max_genes: int | None,
) -> tuple[np.ndarray, int, int]:
    n_genes = len(gene_names)
    if gene_start < 0 or gene_start >= n_genes:
        raise ValueError(f"gene_start 超出范围: {gene_start}")

    resolved_end = n_genes if gene_end is None else min(gene_end, n_genes)
    if resolved_end <= gene_start:
        raise ValueError(
            f"gene_end 必须大于 gene_start，收到 {resolved_end} <= {gene_start}"
        )

    selected = gene_names[gene_start:resolved_end]
    if max_genes is not None:
        selected = selected[:max_genes]
        resolved_end = gene_start + len(selected)

    return selected, gene_start, resolved_end


def _slice_gene_counts(matrix, start: int, end: int) -> np.ndarray:
    subset = matrix[:, start:end]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=DTYPE_NP)
    return np.asarray(subset, dtype=DTYPE_NP)


def _split_gene_batches(
    *,
    n_genes: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> list[tuple[int, int, int]]:
    all_ranges = [
        (batch_index, start, min(start + batch_size, n_genes))
        for batch_index, start in enumerate(range(0, n_genes, batch_size))
    ]
    return [item for item in all_ranges if item[0] % world_size == rank]


def _print_dataset_summary(
    *,
    n_cells: int,
    total_genes: int,
    selected_genes: int,
    local_genes: int,
    layer: str | None,
    device: str,
    batch_size: int,
    cell_chunk_size: int,
    rank: int,
    world_size: int,
) -> None:
    table = Table(title="Dataset")
    table.add_column("Cells", justify="right")
    table.add_column("Genes", justify="right")
    table.add_column("Selected", justify="right")
    table.add_column("Local", justify="right")
    table.add_column("Layer")
    table.add_column("Device")
    table.add_column("Batch", justify="right")
    table.add_column("CellChunk", justify="right")
    table.add_column("Rank", justify="right")
    table.add_row(
        str(n_cells),
        str(total_genes),
        str(selected_genes),
        str(local_genes),
        layer or "X",
        device,
        str(batch_size),
        str(cell_chunk_size),
        f"{rank}/{world_size}",
    )
    console.print(table)


def _print_pool_summary(mu: float, sigma: float, s_hat: float) -> None:
    table = Table(title="Pool Scale")
    table.add_column("mu", justify="right")
    table.add_column("sigma", justify="right")
    table.add_column("s_hat", justify="right")
    table.add_row(f"{mu:.4f}", f"{sigma:.4f}", f"{s_hat:.4f}")
    console.print(table)


def _resolve_fit_device(
    *,
    requested_device: str,
    n_cells: int,
    batch_size: int,
    grid_size: int,
    torch_dtype: str,
    cell_chunk_size: int,
) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device

    bytes_per_value = 8 if torch_dtype == "float64" else 4
    effective_cells = min(n_cells, cell_chunk_size)
    estimated_bytes = effective_cells * batch_size * grid_size * bytes_per_value * 4
    estimated_gib = estimated_bytes / (1024**3)
    if estimated_gib > 8.0:
        console.print(
            "[yellow]Warning:[/yellow] estimated likelihood cache is too large for efficient GPU fitting; "
            f"falling back to CPU (est. cache {estimated_gib:.2f} GiB)."
        )
        return "cpu"
    return requested_device
